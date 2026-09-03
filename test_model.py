import os

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ADAPTER_PATH=./fo-adapter-unsloth to test the Unsloth-trained adapter.
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./fo-adapter")
print(f"Adapter: {ADAPTER_PATH}\n")

model, tokenizer = load(
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    adapter_path=ADAPTER_PATH
)

questions = [
    "How does the Heston model correlation parameter affect the implied volatility skew?",
    "Explain gamma risk for a short BANKNIFTY straddle on expiry day.",
    "What is the variance risk premium and how do you harvest it on NSE?",
    "How does the Carr-Madan FFT method work for option pricing?",
]

for q in questions:
    messages = [{"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # mlx_lm >= 0.2x dropped generate(temp=...); temperature goes via a sampler.
    response = generate(
        model, tokenizer, prompt=prompt, max_tokens=512,
        sampler=make_sampler(temp=0.7),
    )
    print(f"\nQ: {q}")
    print(f"A: {response}")
    print("=" * 60)
