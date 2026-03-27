import argparse
import sys
import gymnasium as gym
import numpy as np
import sbx
import os
from isaaclab.app import AppLauncher

# 1. Setup Parser for YOUR arguments first
parser = argparse.ArgumentParser(description="Load and Play Saved SAC Model")
parser.add_argument("--task", type=str, default="quadloco-a1-flat-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .zip file")
parser.add_argument("--action_range", type=float, default=3.0, help="Action range")

# Add IsaacLab specific args to your parser
AppLauncher.add_app_launcher_args(parser)

# Use parse_known_args to split your args from Hydra's overrides
args_cli, hydra_args = parser.parse_known_args()

# IMPORTANT: Clear sys.argv for Hydra so it doesn't see --checkpoint
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after Isaac Launch ---
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import QuadLoco.tasks

@hydra_task_config(args_cli.task, "sb3_sac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 2. Setup Environment
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)
    
    # Match the action range used during tuning
    env.action_space = gym.spaces.Box(
        low=-args_cli.action_range, 
        high=args_cli.action_range, 
        shape=(12,), 
        dtype=np.float32
    )

    # 3. Load the Model
    print(f"\n--- Loading Model: {args_cli.checkpoint} ---")
    model = sbx.SAC.load(args_cli.checkpoint, env=env)

    # 4. Play Loop
    obs = env.reset()
    print("--- Starting Visualization (Deterministic=True) ---")
    try:
        while simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, resets, extras = env.step(action)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()