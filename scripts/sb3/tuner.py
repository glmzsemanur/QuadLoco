import argparse
import sys
import os
import time
import random
import logging
import numpy as np
import optax
import torch
import gymnasium as gym
import optuna
from datetime import datetime
from contextlib import redirect_stdout

from isaaclab.app import AppLauncher

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Optuna Pruning Tuner")
parser.add_argument("--task", type=str, default="quadloco-a1-flat-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=4096, help="Environments.")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--action_range", type=float, default=3.0, help="Action range.")
parser.add_argument("--wandb_project", type=str, default="sac_staired512_2tuning", help="WandB project.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax.nn
import sbx
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
import QuadLoco.tasks

from optuna.pruners import BasePruner
class StaircasePruner(BasePruner):
    def __init__(self):
        self.gates = {0: 20.0, 1: 32.0, 2: 32.0}
    def prune(self, study, trial):
        step = trial.last_step
        if step is None or step not in self.gates:
            return False
        return trial.intermediate_values[step] < self.gates[step]
    
class TrialEvalCallback(BaseCallback):
    def __init__(self, trial, eval_env, log_file):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.log_file = log_file
        self.start_time = time.time()
        self.checkpoints = [200, 300, 480] 
        self.current_checkpoint_idx = 0
        self.is_pruned = False
        self.last_mean_reward = 0.0
        self.last_std_reward = 0.0

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed >= 500: # Hard stop at 500 seconds (8m20s) to prevent runaway trials
            self.log_file.write(f"--- REACHED MAX TIME (9m) --- Ending Trial {self.trial.number} successfully.\n")
            return False # This stops the model.learn loop
        if self.current_checkpoint_idx < len(self.checkpoints):
            if elapsed >= self.checkpoints[self.current_checkpoint_idx]:
                # Evaluate
                mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=4096)    
                self.last_mean_reward = mean_reward
                self.last_std_reward = std_reward
                # Report index (0, 1, 2)
                self.trial.report(mean_reward, self.current_checkpoint_idx)
                self.log_file.write(f"[GATE {self.current_checkpoint_idx}] Reward: {mean_reward:.2f} +/- {std_reward:.2f} at {elapsed:.0f}s\n")
                
                if self.trial.should_prune():
                    self.log_file.write(f"!!! PRUNED !!! Failed Gate {self.current_checkpoint_idx}\n")
                    self.is_pruned = True
                    self.trial.study.tell(self.trial, mean_reward)
                    return False 
                
                self.current_checkpoint_idx += 1
        return True

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
        "policy_kwargs": {"layer_norm": True, "net_arch": [512, 256, 128], "activation_fn": jax.nn.elu, "optimizer_class": optax.adamw},
        "seed": 42,
        "buffer_size": 8_000_000,
        "learning_starts": 100,
    }

@hydra_task_config(args_cli.task, "sb3_sac_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs

    study_name = args_cli.wandb_project
    storage_url = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_url, 
        direction="maximize", 
        load_if_exists=True,
        pruner=StaircasePruner()
    )

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

    with open(f"optuna/{args_cli.wandb_project}/results.txt", "a", buffering=1) as f:
        f.write(f"\n[TRIAL {trial.number}] STARTING ---\n")
        
        run = wandb.init(project=args_cli.wandb_project, sync_tensorboard=True, name=f"trial_{trial.number}", config=hyperparams)
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = Sb3VecEnvWrapper(env, fast_variant=True)
        env.action_space = gym.spaces.Box(low=-args_cli.action_range, high=args_cli.action_range, shape=(12,), dtype=np.float32)
        
        model = sbx.SAC("MlpPolicy", env, verbose=0, **hyperparams, tensorboard_log=f"optuna/{args_cli.wandb_project}/{trial.number}")

        eval_cb = TrialEvalCallback(trial, env, f)
        
        try:
            # We set a high timestep count; the callback will stop it based on time
            model.learn(
                log_interval=1000,
                total_timesteps=int(1e8), 
                callback=[eval_cb, WandbCallback(verbose=0)], 
                progress_bar=True
            )

            if not eval_cb.is_pruned:
                # Save stable high performers
                final_reward = eval_cb.last_mean_reward
                if final_reward > 30:
                    os.makedirs(f"optuna/{args_cli.wandb_project}/successes", exist_ok=True)
                    model.save(f"optuna/{args_cli.wandb_project}/successes/trial_{trial.number}_FINAL")

                study.tell(trial, final_reward)
                f.write(f"[TRIAL {trial.number}] SUCCESS: {final_reward:.2f}, {eval_cb.last_std_reward:.2f}\n | Params: {trial.params}")
            else:
                # If pruned, Optuna already knows the state via trial.report()
                pass

        except Exception as e:
            f.write(f"[ERROR] Trial {trial.number}: {e}\n")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        finally:
            print("Waiting for WandB to finish syncing files...")
            wandb.finish()
            env.close()
            run.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()