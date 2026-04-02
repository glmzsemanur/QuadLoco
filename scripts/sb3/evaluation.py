"""
Final Evaluation Script for Isaac Lab + SB3/SBX.
Verified for Istanbul Technical University (ITU) - Graduation Phase.
- Robust Action Space Peeking
- Manual Normalization Stats Injection
- Global Scraper for all Isaac Metrics
- 100% Robot Population Sanity Checks
"""

import argparse
import os
import sys
import torch
import numpy as np
import gymnasium as gym
import logging
import pickle
from datetime import datetime

from isaaclab.app import AppLauncher

# 1. Parse Args
parser = argparse.ArgumentParser(description="Evaluate a trained RL agent.")
parser.add_argument("--task", type=str, default="isaaclab-a1-flat-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of robots to evaluate.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps.")
parser.add_argument("--algorithm", type=str, required=True, default=None, choices=["ppo", "sac"])
parser.add_argument("--ml_framework", type=str, default="jax", choices=["jax", "torch"])
parser.add_argument("--plot_action_dist", action="store_true", default=False, help="Create a PDF plot for the action distribution.")
parser.add_argument("--plot_vel_error", action="store_true", default=False, help="Create a PDF plot for the velocity error distribution.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# --- PREALLOCATION & LOGGING SETTINGS FOR JAX ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("absl").setLevel(logging.WARNING)

# Launch Isaac Sim
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports after App Launch
from stable_baselines3.common.vec_env import VecNormalize
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
import QuadLoco 

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

    # Create Environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)

    # --- SPACE PEEK ---
    if os.path.exists(args_cli.checkpoint):
        try:
            from stable_baselines3.common.save_util import load_from_zip_file
            data, _, _ = load_from_zip_file(args_cli.checkpoint)
            if "action_space" in data:
                env.action_space = data["action_space"]
                print(f"[INFO] Auto-detected Action Space: {env.action_space}")
            if "observation_space" in data:
                env.observation_space = data["observation_space"]
                print(f"[INFO] Auto-detected Observation Space: {env.observation_space}")
        except Exception as e:
            print(f"[WARNING] Space peek failed: {e}")

    # --- NORMALIZATION INJECTION ---
    vec_norm_path = args_cli.checkpoint.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vec_norm_path):
        vec_norm_path = args_cli.checkpoint.replace(".zip", ".pkl")
    
    if os.path.exists(vec_norm_path):
        print(f"[INFO] Injecting stats from: {vec_norm_path}")
        env = VecNormalize(env, training=False, norm_reward=False)
        try:
            with open(vec_norm_path, "rb") as f:
                saved_stats = pickle.load(f)
                env.obs_rms = saved_stats.obs_rms
                env.epsilon = saved_stats.epsilon
                env.clip_obs = saved_stats.clip_obs
                if hasattr(saved_stats, 'ret_rms'): env.ret_rms = saved_stats.ret_rms
                print("[SUCCESS] Stats injected.")
        except Exception as e:
            print(f"[ERROR] Injection failed: {e}")

    # Load Model
    model = RLAlgorithm.load(args_cli.checkpoint, env=env, device="cuda")

    # --- EVALUATION LOOP (THE SCRAPER) ---
    num_steps, num_envs = args_cli.eval_steps, args_cli.num_envs
    all_step_logs = []
    rewards_history = torch.zeros((num_steps, num_envs), device="cuda")
    action_history = []

    print(f"--- Evaluating {num_envs} Robots for {num_steps} steps ---")
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    for t in range(num_steps):
        with torch.inference_mode():
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            action_history.append(actions)

            rewards_history[t] = torch.as_tensor(rewards, device="cuda")

            # Scrape Isaac Data
            step_data = {}
            unwrapped_env = env.unwrapped
            if hasattr(unwrapped_env, "extras") and "log" in unwrapped_env.extras:
                for key, value in unwrapped_env.extras["log"].items():
                    step_data[key] = value.clone() if isinstance(value, torch.Tensor) else value
            
            step_data["_phys_height"] = unwrapped_env.scene["robot"].data.root_pos_w[:, 2].clone()
            all_step_logs.append(step_data)

    # --- POST-PROCESSING & SANITY CHECKS ---
    print("\n" + "!"*20 + " SANITY CHECKS " + "!"*20)
    print(f"Data Shape Check (Rewards): {rewards_history.shape}") # (1000, 4096)
    
    available_keys = all_step_logs[0].keys() if all_step_logs else []
    final_stats_dict = {}

    # 1. Survival (Base Contact)
    contact_key = 'Episode_Termination/base_contact'
    if contact_key in available_keys:
        contact_history = torch.stack([torch.as_tensor(s[contact_key], device="cuda") for s in all_step_logs])
        ever_fell = torch.any(contact_history > 0.5, dim=0)
        survival_rate = (1.0 - ever_fell.float().mean()).item() * 100
    else:
        all_heights = torch.stack([torch.as_tensor(s["_phys_height"], device="cuda") for s in all_step_logs])
        ever_fell = torch.any(all_heights < 0.22, dim=0)
        survival_rate = (1.0 - ever_fell.float().mean()).item() * 100
    
    final_stats_dict["Survival_Rate_Pct"] = survival_rate

    # 2. Velocity Error
    vel_err_key = 'Metrics/base_velocity/error_vel_xy'
    if vel_err_key in available_keys:
        vel_errors = torch.stack([torch.as_tensor(s[vel_err_key], device="cuda") for s in all_step_logs])
        final_stats_dict[vel_err_key] = torch.mean(vel_errors).item()
        
        # --- VELOCITY ERROR PLOTS ---
        if args_cli.plot_vel_error:
            print("\n🎉 Generating velocity error plots...")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme()
            
            vel_errors_np = vel_errors.cpu().numpy()
            
            # Handle both 1D (already averaged across envs) and 2D (per-env) shapes
            if vel_errors_np.ndim == 1:
                mean_error_per_step = vel_errors_np
            else:
                mean_error_per_step = vel_errors_np.mean(axis=1)
                
            flat_errors = vel_errors_np.flatten()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(mean_error_per_step, color="red", linewidth=2)
            ax.set_title("Mean Velocity Error Per Step", fontsize=14)
            ax.set_xlabel("Step", fontsize=12)
            ax.set_ylabel("Mean Error", fontsize=12)
            ax.set_ylim(0, 0.3)
            
            plt.tight_layout()
            error_plot_save_path = os.path.join(os.path.dirname(args_cli.checkpoint), f"eval_vel_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            plt.savefig(error_plot_save_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"📊 Velocity error plot saved to: {error_plot_save_path}")

    # 3. Dynamic Loop for all other keys
    for key in available_keys:
        if key in [contact_key, vel_err_key, "_phys_height"]: continue
        try:
            data_stack = torch.stack([torch.as_tensor(s[key], device="cuda") for s in all_step_logs])
            # We log the raw mean for the report
            final_stats_dict[key.replace("/", "_")] = torch.mean(data_stack).item()
        except: continue

    # Comparison Mode: Total Reward
    total_rewards_per_robot = torch.sum(rewards_history, dim=0)
    final_stats_dict["Cumulative_Reward_Mean"] = torch.mean(total_rewards_per_robot).item()
    
    # --- ACTION SPACE PERCENTILES ---
    action_history_np = np.stack(action_history) # (num_steps, num_envs, action_dim)
    flat_actions = action_history_np.reshape(-1, action_history_np.shape[-1])
    low_pct = np.percentile(flat_actions, 2.5, axis=0)
    high_pct = np.percentile(flat_actions, 97.5, axis=0)
    
    low_str = ", ".join([f"{x:.2f}" for x in low_pct])
    high_str = ", ".join([f"{x:.2f}" for x in high_pct])
    
    print("\n" + "="*50)
    print("Action Space Boundaries (2.5th to 97.5th Percentiles)")
    print(f"low=np.array([{low_str}]),")
    print(f"high=np.array([{high_str}])")
    
    final_stats_dict["Action_Space_Low"] = f"[{low_str}]"
    final_stats_dict["Action_Space_High"] = f"[{high_str}]"

    # --- CONTINUOUS ACTION DISTRIBUTION PLOT ---
    if args_cli.plot_action_dist:
        print("\n🎉 Generating continuous Gaussian plots for evaluated actions...")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        n_actions = flat_actions.shape[1]
        
        for i in range(n_actions):
            # kde=True generates the continuous Probability Density Function
            sns.kdeplot(flat_actions[:, i], fill=True, alpha=0.2, label=f"Joint {i}", bw_adjust=1.5)
            
        plt.title("Action Distribution (Continuous Gaussian PDF) during Evaluation", fontsize=16)
        plt.xlabel("Action Value (Position)", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)
        plt.legend(loc='best')
        plt.tight_layout()
        
        plot_save_path = os.path.join(os.path.dirname(args_cli.checkpoint), f"eval_action_distributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        plt.savefig(plot_save_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"📊 Action distribution plot saved to: {plot_save_path}")

    print(f"Verify Non-Zero Data:      {torch.any(rewards_history > 0).item()}")
    print(f"Best Robot Total Reward:   {torch.max(total_rewards_per_robot).item():.2f}")
    print(f"Worst Robot Total Reward:  {torch.min(total_rewards_per_robot).item():.2f}")
    print("!"*55)

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("           COMPLETE EVALUATION SUMMARY")
    print("="*50)
    for k, v in final_stats_dict.items():
        if isinstance(v, (float, int)):
            print(f"{k:<40}: {v:.6f}")
        else:
            print(f"{k:<40}: {v}")
    print("="*50)

    save_path = os.path.join(os.path.dirname(args_cli.checkpoint), "full_eval_report.txt")
    with open(save_path, "w") as f:
        f.write(f"Report Generated: {datetime.now()}\n")
        for k, v in final_stats_dict.items():
            if isinstance(v, (float, int)):
                f.write(f"{k}: {v:.6f}\n")
            else:
                f.write(f"{k}: {v}\n")
    print(f"[INFO] Report saved to: {save_path}")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()