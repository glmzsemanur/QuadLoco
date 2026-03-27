import subprocess #  ["ppo", "torch", 100], ["ppo", "jax", 100], 
# for algorithm, ml_framework, action_range in [["sac", "jax", 5], ["sac", "jax", 4], ["sac", "jax", 3], ["sac", "jax", 2], ["sac", "jax", 1]]:
for num_envs in [256, 512, 1028, 2048, 4096]:
    print(f"\n=== Starting training")
    cmd = [
        "python",
        "train.py",
        "--task=quadloco-a1-rough-v0",
        "--headless",
        f"--algorithm=sac",
        f"--ml_framework=jax",
        f"--action_range=3",
        f"--wandb_name=sac_action_range_3_{num_envs}_320grad",
        f"--num_envs={num_envs}"
    ]
    subprocess.run(cmd, check=True)
    print(f"\n=== Finished training ===\n")
print("\nAll trainings completed.")