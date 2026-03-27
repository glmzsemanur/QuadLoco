import argparse
import sys
import os
import time
import random
import logging
import numpy as np
import torch
import gymnasium as gym
import optuna
from datetime import datetime

from isaaclab.app import AppLauncher

# Standard Argument Parser
parser = argparse.ArgumentParser(description="Tuner: 1 Trial per Call")
parser.add_argument("--task", type=str, default="quadloco-a1-flat-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment.")
parser.add_argument("--action_range", type=float, default=3.0, help="Action range for SAC.")
parser.add_argument("--wandb_project", type=str, default="sac_tuning_third", help="WandB project name.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Handle Hydra conflict
sys.argv = [sys.argv[0]] + hydra_args

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- JAX DEBUG & MEMORY PREVENTERS ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60" 
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
        "policy_kwargs": {"layer_norm": True, "net_arch": [128, 128, 128], "activation_fn": jax.nn.elu},
        "seed": 42,
        "buffer_size": 8_000_000,
        "learning_starts": 100,
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

    # 2. Optuna Setup
    study_name = f"tuning_sac_{args_cli.task}_third"
    storage_url = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_url, 
        direction="maximize", 
        load_if_exists=True
    )

    trial = study.ask() 
    print(f"\n[DEBUG] Starting Trial {trial.number} | Time Limit: 5 Minutes")
    
    params = {
        "gamma": trial.suggest_float("gamma", 0.975, 0.995),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.002, log=True),
        "ent_coef_init": trial.suggest_float("ent_coef_init", 0.001, 0.02, log=True),
        "batch_size_pow": trial.suggest_int("batch_size_pow", 7, 12),
        "train_freq_pow": trial.suggest_int("train_freq_pow", 0, 3),
        "gradient_steps_pow": trial.suggest_int("gradient_steps_pow", 0, 10),
        "policy_delay_pow": trial.suggest_int("policy_delay_pow", 0, 5),
        "tau": trial.suggest_float("tau", 0.001, 0.05, log=True),
    }
    hyperparams = to_hyperparams(params)
    print(f"[DEBUG] Params Sampled: LR={hyperparams['learning_rate']:.6f}, Batch={hyperparams['batch_size']}, GradSteps={hyperparams['gradient_steps']}")

    # 3. WandB Setup (Minimized output)
    run = wandb.init(
        project=args_cli.wandb_project,
        name=f"trial_{trial.number}_{args_cli.task}",
        config={**hyperparams, "trial_num": trial.number},
        sync_tensorboard=True,
        reinit="finish_previous",
        settings=wandb.Settings(console="off")
    )

    # 4. Environment & Agent
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env, fast_variant=True)
    env.action_space = gym.spaces.Box(low=-args_cli.action_range, high=args_cli.action_range, shape=(12,), dtype=np.float32)

    print(f"[DEBUG] Initializing SBX SAC Model...")
    model = sbx.SAC("MlpPolicy", env, verbose=0, 
                    **hyperparams, tensorboard_log=f"optuna/runs/{trial.number}") # tensorboard_log removed to prevent massive event file uploads

    # 5. Train
    try:
        print(f"[DEBUG] Beginning 5-minute training burst...")
        model.learn(total_timesteps=int(5e7), callback=[TimeoutCallback(60*5), WandbCallback(verbose=0)], progress_bar=True, log_interval=1000)
        
        print(f"[DEBUG] Training finished. Running Evaluation...")
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=128, return_episode_rewards=False)

        # 1. Tell Optuna
        study.tell(trial, mean_reward) 
        
        # 2. Print success and write to file
        result_line = f"Trial {trial.number} Result: {mean_reward:.4f} +/- {std_reward:.4f} | Params: {trial.params}"
        print(f"[SUCCESS] {result_line}")
        with open(f"{study_name}_results.txt", "a") as f:
            f.write(result_line + "\n")
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} encountered an error: {e}")
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    finally:
        env.close()
        run.finish()
        print(f"[DEBUG] Trial {trial.number} Cleanup Complete.")

if __name__ == "__main__":
    main()
    simulation_app.close()