import subprocess

# List of configurations to run: [task_name, algorithm, action_range]
runs = [
    # ["isaaclab-a1-flat-v0", "ppo", 100],
    ["isaaclab-a1-flat-v0", "sac", 1],
]

for task_name, algorithm, action_range in runs:
    # Derive wandb_name automatically (e.g. quadloco-a1-rough-v0 -> rough)
    task_suffix = task_name.split('-')[2] if '-' in task_name else task_name
    wandb_name = f"{algorithm}_action_{action_range}_{task_suffix}"

    print(f"\n=== Starting training: {wandb_name} ===")
    cmd = [
        "python",
        "train.py",
        f"--task={task_name}",
        "--headless",
        f"--algorithm={algorithm}",
        "--ml_framework=jax",
        f"--action_range={action_range}",
        f"--wandb_name={wandb_name}"
    ]
    subprocess.run(cmd, check=True)
    print(f"\n=== Finished training: {wandb_name} ===\n")

print("\nAll trainings completed.")