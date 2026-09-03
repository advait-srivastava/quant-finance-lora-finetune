import subprocess

cmd = [
    "python", "-m", "mlx_lm.lora",
    "--model", "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "--data", "./data",
    "--train",
    "--batch-size", "1",
    "--lora-layers", "8",
    "--iters", "500",
    "--learning-rate", "1e-5",
    "--adapter-path", "./fo-adapter",
    "--seed", "42",
]

print("Starting MLX LoRA fine-tuning...")
subprocess.run(cmd)
print("Done! Adapter saved to ./fo-adapter")
