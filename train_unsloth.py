"""LoRA fine-tune on the quant-finance Q&A set using Unsloth.

On Apple Silicon, Unsloth dispatches FastLanguageModel -> FastMLXModel and
trains through MLXTrainer, not TRL's SFTTrainer. So load_in_4bit here is real
MLX quantization (not bitsandbytes), which is what makes the 3B fit in 8GB.

Defaults mirror train_mlx.py's base model so the two adapters are comparable.
Override with MODEL=... (e.g. Qwen/Qwen2.5-1.5B-Instruct for a bf16 run).
"""
import os

from unsloth import FastLanguageModel
from unsloth_zoo.mlx.trainer import (
    MLXTrainer,
    MLXTrainingConfig,
    train_on_responses_only,
)
from datasets import load_dataset

MODEL = os.environ.get("MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 640))  # longest example is 562 tok
EPOCHS = int(os.environ.get("EPOCHS", 2))
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./fo-adapter-unsloth")

print(f"Base model : {MODEL}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,  # keep native dtype; auto-picks bf16 on bf16-capable chips
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    use_gradient_checkpointing="mlx",
    max_seq_length=MAX_SEQ_LEN,
    random_state=42,
)

dataset = load_dataset(
    "json",
    data_files={"train": "data/train.jsonl", "validation": "data/valid.jsonl"},
)


def to_text(batch):
    return {
        "text": [
            tokenizer.apply_chat_template(m, tokenize=False)
            for m in batch["messages"]
        ]
    }


dataset = dataset.map(to_text, batched=True, remove_columns=["messages"])
print(f"Train: {len(dataset['train'])}  Valid: {len(dataset['validation'])}")

trainer = MLXTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=MLXTrainingConfig(
        # Peak was only 3.06 GB of 8 GB at bs=1, so batch the real sequences
        # instead of accumulating. Effective batch stays 4 either way.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=EPOCHS,
        max_steps=-1,
        learning_rate=2e-4,
        warmup_steps=10,
        lr_scheduler_type="linear",
        optim="adamw",
        weight_decay=0.01,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        logging_steps=5,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        output_dir="outputs-unsloth",
        report_to="none",
        seed=42,
    ),
)

# Mask the prompt so loss lands only on the assistant's answer.
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

print("Starting Unsloth LoRA fine-tuning...")
stats = trainer.train()
print(stats)

# save_lora_adapters() alone writes only Unsloth's own config schema, which
# mlx_lm.load() can't read (it needs fine_tune_type/num_layers/lora_parameters
# and otherwise dies on a missing `num_layers`). Passing them here keeps the
# adapter loadable by plain mlx_lm / test_model.py. scale must be alpha/r —
# mlx_lm defaults to 20.0, which would apply the adapter 20x too strong.
model.save_lora_adapters(
    ADAPTER_PATH,
    adapter_config={
        "fine_tune_type": "lora",
        "num_layers": model.args.num_hidden_layers,
        "lora_parameters": {
            "rank": 16,
            "scale": 16 / 16,  # lora_alpha / r
            "dropout": 0.0,
            "keys": [
                "self_attn.q_proj", "self_attn.k_proj",
                "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ],
        },
    },
)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"Done! Adapter saved to {ADAPTER_PATH}")
