import argparse
import sys
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
import random
import logging
import numpy as np
import torch
import gymnasium as gym
import optuna
from datetime import datetime
from contextlib import redirect_stdout # Simple, non-intrusive logging

from isaaclab.app import AppLauncher

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Gated Stability Tuner")
parser.add_argument("--task", type=str, default="quadloco-a1-flat-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--action_range", type=float, default=3.0, help="Action range.")
parser.add_argument("--wandb_project", type=str, default="sac_gated_tuning", help="WandB project.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
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

# --- 2. Stability Callbacks ---
class GatedTimeoutCallback(BaseCallback):
    def __init__(self, timeout_seconds: int):
        super().__init__()
        self.timeout = timeout_seconds
        self.start_time = time.time()
    def _on_step(self) -> bool:
        return (time.time() - self.start_time) < self.timeout

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

@hydra_task_config(args_cli.task, "sb3_sac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs

    study_name = f"tuning_sac_{args_cli.task}_gated"
    storage_url = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=storage_url, direction="maximize", load_if_exists=True)

    trial = study.ask()
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

    # --- Start Recording to File ---
    with open("tuner_results.txt", "a", buffering=1) as f:
            f.write(f"\n[TRIAL {trial.number}] START ---\n")
            
            run = wandb.init(
                project=args_cli.wandb_project,
                name=f"trial_{trial.number}",
                config={**hyperparams, "trial_num": trial.number},
                sync_tensorboard=True,
                reinit="finish_previous",
                settings=wandb.Settings(console="off")
            )
            env = gym.make(args_cli.task, cfg=env_cfg)
            env = Sb3VecEnvWrapper(env, fast_variant=True)
            env.action_space = gym.spaces.Box(low=-args_cli.action_range, high=args_cli.action_range, shape=(12,), dtype=np.float32)
            
            model = sbx.SAC("MlpPolicy", env, verbose=0, **hyperparams, tensorboard_log=f"optuna/runs3/{trial.number}")

            try:
                # PHASE 1 (4 min)
                f.write(f"[TRIAL {trial.number}] Phase 1: 4m Training...\n")
                model.learn(log_interval=1000, total_timesteps=int(5e7), progress_bar=True, callback=[GatedTimeoutCallback(240), WandbCallback(verbose=0)], reset_num_timesteps=False)
                mean_4, std_4 = evaluate_policy(model, env, n_eval_episodes=4096)
                f.write(f"[TRIAL {trial.number}] Result @ 4m: {mean_4:.2f}, {std_4:.2f}\n")

                if mean_4 > 15:
                    # PHASE 2 (6 min total)
                    f.write(f"[TRIAL {trial.number}] Phase 2: +2m Check...\n")
                    model.learn(log_interval=1000, total_timesteps=int(5e7), progress_bar=True, callback=[GatedTimeoutCallback(120), WandbCallback(verbose=0)], reset_num_timesteps=False)
                    mean_6, std_6 = evaluate_policy(model, env, n_eval_episodes=4096)
                    f.write(f"[TRIAL {trial.number}] Result @ 6m: {mean_6:.2f}, {std_6:.2f}\n")

                    if mean_6 > 30:
                        # PHASE 3 (8 min total)
                        f.write(f"[TRIAL {trial.number}] Phase 3: Final +2m Check...\n")
                        model.learn(log_interval=1000, total_timesteps=int(5e7), progress_bar=True, callback=[GatedTimeoutCallback(120), WandbCallback(verbose=0)], reset_num_timesteps=False)
                        mean_8, std_8 = evaluate_policy(model, env, n_eval_episodes=4096)
                        f.write(f"[TRIAL {trial.number}] Result @ 8m: {mean_8:.2f}, {std_8:.2f}\n")
                        os.makedirs("best_models", exist_ok=True)
                        model.save(f"best_models/trial_{trial.number}_6")
                        f.write(f"!!! SAVED TRIAL {trial.number} !!!\n")
                        if mean_8 > 30:
                            os.makedirs("best_models", exist_ok=True)
                            model.save(f"best_models/trial_{trial.number}_8")
                            f.write(f"!!! SAVED TRIAL {trial.number} !!!\n")
                            study.tell(trial, mean_8)
                        else:
                            study.tell(trial, mean_8)
                    else:
                        study.tell(trial, mean_6)
                else:
                    study.tell(trial, mean_4)

            except Exception as e:
                f.write(f"[ERROR] Trial {trial.number}: {e}\n")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            finally:
                env.close()
                run.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()