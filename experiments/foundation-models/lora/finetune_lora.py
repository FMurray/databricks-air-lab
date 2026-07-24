"""LoRA fine-tune (UAT W3) — the "Ops rapid fine-tuning" archetype, GA surface only.

Dry-run default is a small model on A10 so the plumbing (HF download, peft wiring, MLflow,
loss trajectory) is proven cheaply; the acceptance run overrides to a Mistral-class model on
8xH100. Pass = loss decreased over training + measured wall-clock printed (feeds the
right-sizing note: did it need the GPUs it got?).

Env knobs: MODEL_ID (default Qwen/Qwen2.5-0.5B), MAX_STEPS (default 60), LORA_R (default 16).
"""
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60"))
LORA_R = int(os.environ.get("LORA_R", "16"))

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=32, target_modules="all-linear",
                                         task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"RESULT model={MODEL_ID} trainable_params={trainable}", flush=True)

ds = load_dataset("tatsu-lab/alpaca", split="train[:2000]")


def fmt(ex):
    text = f"### Instruction:\n{ex['instruction']}\n### Response:\n{ex['output']}"
    out = tok(text, truncation=True, max_length=512, padding="max_length")
    out["labels"] = out["input_ids"].copy()
    return out


ds = ds.map(fmt, remove_columns=ds.column_names)
print("PHASE data_ready", flush=True)

args = TrainingArguments(
    output_dir="/tmp/lora-out", max_steps=MAX_STEPS, per_device_train_batch_size=4,
    bf16=True, logging_steps=10, report_to="mlflow", save_strategy="no",
)
t0 = time.time()
trainer = Trainer(model=model, args=args, train_dataset=ds)
result = trainer.train()
wall = time.time() - t0

history = [h["loss"] for h in trainer.state.log_history if "loss" in h]
assert len(history) >= 2, "no loss history logged"
assert history[-1] < history[0], f"loss did not decrease: {history[0]:.3f} -> {history[-1]:.3f}"
print(f"RESULT lora=PASS steps={MAX_STEPS} wall_seconds={wall:.0f} "
      f"loss_first={history[0]:.3f} loss_last={history[-1]:.3f} "
      f"gpus={torch.cuda.device_count()}", flush=True)
