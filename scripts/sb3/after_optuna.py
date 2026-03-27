import argparse
import sys
import os
import time
import random
import logging
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime

from isaaclab.app import AppLauncher

# Standard Argument Parser
parser = argparse.ArgumentParser(description="Manual SAC Run")
parser.add_argument("--task", type=str, default="quadloco-a1-flat-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment.")
parser.add_argument("--action_range", type=float, default=3.0, help="Action range for SAC.")
parser.add_argument("--wandb_project", type=str, default="sac_manual_verification", help="WandB project name.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Handle Hydra conflict
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- JAX DEBUG & MEMORY PREVENTERS ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60" # Recommended for stability
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import jax.nn
import sbx
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
import QuadLoco.tasks

# Hyperparameter conversion
def to_hyperparams(params: dict) -> dict:
    return {
        "gamma": params["gamma"],
        "learning_rate": params["learning_rate"],
        "ent_coef": f"auto_{params['ent_coef_init']}",
        "batch_size": 2**params["batch_size_pow"],
        "tau": params["tau"],
        "train_freq": 2**params["train_freq_pow"],
        "gradient_steps": 2**params["gradient_steps_pow"],
        "policy_delay": 2**params["policy_delay_pow"],
        "policy_kwargs": {"net_arch": [128, 128, 128], "activation_fn": jax.nn.elu},
        "seed": args_cli.seed,
    }

class TimeoutCallback(BaseCallback):
    def __init__(self, timeout: int):
        super().__init__()
        self.timeout = timeout
        self.start_time = time.time()
    def _on_step(self) -> bool:
        return (time.time() - self.start_time) < self.timeout

@hydra_task_config(args_cli.task, "sb3_sac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # 1. Setup Seed
    if args_cli.seed is None or args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs

    # 2. MANUAL PARAMETERS (Enter Trial 72 values here)
    params = {
        "gamma": 0.9766639514032978,            # Change these to match your trial
        "learning_rate": 0.0018441165124470785, 
        "ent_coef_init": 0.01417841501679298, 
        "batch_size_pow": 10,     # 2^10 = 1024
        "train_freq_pow": 3,      # 2^0 = 1
        "gradient_steps_pow": 8,  # 2^0 = 1
        "policy_delay_pow": 2, 
        "tau": 0.007335540238399933,
        "buffer_size": 8_000_000,
        "learning_starts": 100,
        "policy_kwargs": {"layer_norm": True, "net_arch": [128, 128, 128], "activation_fn": jax.nn.elu},
        "seed": 42
    }
    
    hyperparams = to_hyperparams(params)
    run_name = f"manual_verify_{datetime.now().strftime('%H%M%S')}"
    
    print(f"\n[DEBUG] Starting Manual Run: {run_name}")
    print(f"[DEBUG] Params: LR={hyperparams['learning_rate']}, Batch={hyperparams['batch_size']}, Ent_Init={hyperparams['ent_coef']}")

    # 3. WandB Setup
    run = wandb.init(
        project=args_cli.wandb_project,
        name=run_name,
        config={**hyperparams, "mode": "manual_verification"},
        sync_tensorboard=True,
        reinit=True,
        settings=wandb.Settings(console="off")
    )

    # 4. Environment & Agent
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)
    env.action_space = gym.spaces.Box(low=-args_cli.action_range, high=args_cli.action_range, shape=(12,), dtype=np.float32)

    print(f"[DEBUG] Initializing SBX SAC Model...")
    model = sbx.SAC("MlpPolicy", env, verbose=0, **hyperparams, tensorboard_log="optuna/runs/mama")

    # 5. Train
    try:
        # Run for 5 minutes (300s) to replicate the original duration
        timeout_seconds = 30000 
        print(f"[DEBUG] Beginning {timeout_seconds}s training run...")
        
        model.learn(
            total_timesteps=int(5e7), 
            callback=[TimeoutCallback(timeout_seconds), WandbCallback(verbose=1)], 
            progress_bar=True, 
            log_interval=100
        )
        
        print(f"[DEBUG] Training finished. Running Final Evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=32)
        print(f"[RESULT] Final Reward: {mean_reward:.4f} +/- {std_reward:.4f}")
        
    except Exception as e:
        print(f"[ERROR] Run failed: {e}")
    finally:
        env.close()
        run.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()