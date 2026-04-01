import argparse
import os
import sys
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime

from isaaclab.app import AppLauncher

# 1. Parse Args
parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
parser.add_argument("--task", type=str, default="quadloco-a1-flat-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of robots to evaluate.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps.")
parser.add_argument("--algorithm", type=str, default="sac", choices=["ppo", "sac"])
parser.add_argument("--ml_framework", type=str, default="jax", choices=["jax", "torch"])
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Launch Isaac Sim
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports after App Launch (Crucial)
from stable_baselines3.common.vec_env import VecNormalize
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
import QuadLoco 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@hydra_task_config(args_cli.task, f"sb3_{args_cli.algorithm.lower()}_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    
    # Override env config
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Select Algorithm Class
    if args_cli.ml_framework == "torch":
        from stable_baselines3 import PPO, SAC
    else:
        from sbx import PPO, SAC
    RLAlgorithm = PPO if args_cli.algorithm.lower() == "ppo" else SAC

    # --- PEEK ACTION SPACE ---
    detected_action_range = 1.0
    if os.path.exists(args_cli.checkpoint):
        try:
            model_data = RLAlgorithm.load(args_cli.checkpoint, device="cpu")
            detected_action_range = float(model_data.action_space.high[0])
            print(f"[INFO] Auto-detected action range: {detected_action_range}")
            del model_data
        except Exception as e:
            print(f"[WARNING] Peek failed: {e}")

    # Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # Apply Action Space Fix
    env.action_space = gym.spaces.Box(
        low=-detected_action_range, high=detected_action_range, 
        shape=env.action_space.shape, dtype=np.float32
    )

    # Load Normalization
    vec_norm_path = args_cli.checkpoint.replace("model.zip", "model_vecnormalize.pkl")
    if os.path.exists(vec_norm_path):
        print(f"[INFO] Loading normalization: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    # Load Model
    print(f"[INFO] Loading model: {args_cli.checkpoint}")
    model = RLAlgorithm.load(args_cli.checkpoint, env=env, device="cuda")

    # --- EVAL LOOP ---
    num_steps, num_envs = args_cli.eval_steps, args_cli.num_envs
    history = {
        "rew": torch.zeros((num_steps, num_envs), device="cuda"),
        "done": torch.zeros((num_steps, num_envs), device="cuda"),
        "vel_err": torch.zeros((num_steps, num_envs), device="cuda"),
    }

    print(f"--- Starting Eval for {num_envs} robots ---")
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0] # Handle SB3 wrapper vs Gymnasium reset

    for t in range(num_steps):
        with torch.inference_mode():
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)

            history["rew"][t] = torch.as_tensor(rewards, device="cuda")
            history["done"][t] = torch.as_tensor(dones, device="cuda")
            
            if "log" in infos and "lin_vel_error" in infos["log"]:
                history["vel_err"][t] = torch.norm(infos["log"]["lin_vel_error"], dim=-1)

    # --- STATS ---
    mean_rew = torch.mean(torch.sum(history["rew"], dim=0)).item()
    survival = (1.0 - torch.max(history["done"], dim=0)[0].float().mean()).item() * 100
    mean_vel_err = torch.mean(history["vel_err"][int(num_steps*0.8):]).item()

    print("\n" + "="*30)
    print(f"Survival Rate:   {survival:.2f}%")
    print(f"Mean Reward:     {mean_rew:.2f}")
    print(f"Lin Vel Error:   {mean_vel_err:.4f} m/s")
    print("="*30)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()