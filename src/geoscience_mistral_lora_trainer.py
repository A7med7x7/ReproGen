import os
import sys
import argparse
import time
import shutil
import traceback
import random
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
)

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.mlflow_log import log_git, log_gpu, log_python

def fatal(msg: str, code: int = 2):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(code)

def safe_mkdir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def print_env():
    print("Python:", sys.version.split()[0])
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    try:
        print("CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            try:
                print(f"  - [{i}]", torch.cuda.get_device_name(i))
            except Exception:
                pass
    except Exception:
        pass

def load_tokenizer_with_fallback(model_name: str, cache_dir: str | None = None):
    cache_dir = cache_dir or os.environ.get("HF_HOME")
    last_exc = None
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print("Fast tokenizer failed:", e); last_exc = e
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception as e:
        print("Slow tokenizer failed:", e); last_exc = e
    if cache_dir:
        hub_root = os.path.join(cache_dir, "hub")
        if os.path.isdir(hub_root):
            removed = False
            for entry in os.listdir(hub_root):
                if entry.startswith("models--" + model_name.replace("/", "--")) or entry.startswith("models--" + model_name.split("/")[0]):
                    cand = os.path.join(hub_root, entry)
                    try:
                        print("Removing cached model path:", cand)
                        shutil.rmtree(cand)
                        removed = True
                    except Exception as rr:
                        print("Could not remove", cand, rr)
            if removed:
                return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    raise RuntimeError(f"Failed to load tokenizer for {model_name}") from last_exc

# data prep
def prepare_dataset(tokenizer: AutoTokenizer, dataset_name: str, max_length: int, subset_pct: float = 100.0):
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    preferred = [c for c in ds.column_names if c.lower() in ("text","input","prompt","question","context","instruction")]
    text_col = preferred[0] if preferred else ds.column_names[0]
    print("Using text column:", text_col)
    if subset_pct < 100.0:
        n = max(1, int(len(ds) * (subset_pct / 100.0)))
        ds = ds.select(range(n)); print("Using subset:", n)
    def tok_batch(examples: Dict[str, Any]):
        texts = [str(t) for t in examples[text_col]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
    tokenized = ds.map(tok_batch, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch", columns=["input_ids","attention_mask"])
    print("Tokenized dataset size:", len(tokenized))
    return tokenized

# collator
class CausalDataCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# training loop 
def train_loop(
    model,
    tokenizer,
    train_dataset,
    output_dir: str,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 8,
    epochs: int = 1,
    lr: float = 1e-4,
    logging_steps: int = 10,
    save_every_steps: int = 500,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
):
    safe_mkdir(output_dir)

    collator = CausalDataCollator(tokenizer)
    loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    if len(loader) == 0:
        raise RuntimeError("DataLoader is empty! Check your dataset and batch size.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    total_update_steps = max(1, (len(loader) // grad_accum_steps) * epochs)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_update_steps
    )

    scaler = torch.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

    model.train()
    device0 = torch.device("cuda:0") if torch.cuda.is_available() and torch.cuda.device_count() > 0 else torch.device("cpu")
    print("Training start - device0 for inputs:", device0)

    global_step = 0
    running_loss = 0.0
    t0_global = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        for step, batch in enumerate(loader):
            # move batch to device
            input_ids = batch["input_ids"].to(device0, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device0, non_blocking=True)
            labels = batch["labels"].to(device0, non_blocking=True)

            try:
                with torch.amp.autocast("cuda", enabled=use_amp and torch.cuda.is_available(), dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
            except Exception as e:
                print("Exception during forward:", e)
                traceback.print_exc()
                raise

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.item() * grad_accum_steps)

            if (step + 1) % grad_accum_steps == 0:
                # optimizer step
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

                # logging
                if global_step % logging_steps == 0:
                    avg_loss = running_loss / (logging_steps * grad_accum_steps)
                    elapsed = time.time() - t0_global
                    gpu_info = []
                    for dev in range(torch.cuda.device_count()):
                        try:
                            alloc = torch.cuda.memory_allocated(dev)
                            reserved = torch.cuda.memory_reserved(dev)
                            gpu_info.append(f"GPU{dev} alloc={alloc//1024**2}MB res={reserved//1024**2}MB")
                        except Exception:
                            gpu_info.append(f"GPU{dev} unknown")
                    print(f"[E{epoch+1}] global_step {global_step} avg_loss {avg_loss:.4f} elapsed {elapsed:.1f}s {' | '.join(gpu_info)}")
                    mlflow.log_metric("train_loss", avg_loss, step=global_step)
                    running_loss = 0.0
                    t0_global = time.time()

                # save adapter periodically
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    adapter_dir = Path(output_dir) / f"adapter_step_{global_step}"
                    safe_mkdir(adapter_dir)
                    print("Saving LoRA adapter to", adapter_dir)
                    model.save_pretrained(adapter_dir)
                    mlflow.log_artifacts(str(adapter_dir), artifact_path="adapters")

        epoch_time = time.time() - epoch_start
        mlflow.log_metric("epoch_time", epoch_time, step=epoch + 1)
        print(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")

    # final save
    adapter_dir = Path(output_dir) / "adapter_final"
    safe_mkdir(adapter_dir)
    print("🦎 Saving final LoRA adapter to", adapter_dir)
    model.save_pretrained(adapter_dir)
    mlflow.log_artifacts(str(adapter_dir), artifact_path="adapters")
    return adapter_dir

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Open-Orca/Mistral-7B-OpenOrca")
    p.add_argument("--dataset_name", type=str, default="GeoGPT-Research-Project/GeoGPT-QA")
    p.add_argument("--output_dir", type=str, default="/home/jovyan/data/adapters")
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train_subset_pct", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_memory_gpu0", type=str, default="14GiB")
    p.add_argument("--max_memory_gpu1", type=str, default="14GiB")
    p.add_argument("--max_memory_cpu", type=str, default="120GiB")
    return p.parse_args()

# main
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print_env()
    safe_mkdir(args.output_dir)

    if BitsAndBytesConfig is None:
        fatal("BitsAndBytesConfig not available. Install compatible transformers+bitsandbytes.")

    # tokenizer + pad handling
    print("🦎 Loading tokenizer:", args.model_name)
    try:
        tokenizer = load_tokenizer_with_fallback(args.model_name, cache_dir=os.environ.get("HF_HOME"))
    except Exception:
        traceback.print_exc(); fatal("Tokenizer load failed.")

    pad_added = False
    if tokenizer.pad_token is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Set tokenizer.pad_token = eos_token (ok for causal LM).")
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            pad_added = True
            print("Added pad_token '<pad>'; will resize embeddings after model load.")

    # QLoRA config
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    max_memory = {0: args.max_memory_gpu0, 1: args.max_memory_gpu1, "cpu": args.max_memory_cpu}
    print("🦎 max_memory mapping:", max_memory)

    print("🦎 Loading model (QLoRA 4-bit)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_conf,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
        )
    except Exception as e:
        print("Model load failed:", e, file=sys.stderr); traceback.print_exc()
        fatal("Failed to load model with QLoRA settings.")

    if pad_added:
        try:
            model.resize_token_embeddings(len(tokenizer)); print("Resized embeddings to", len(tokenizer))
        except Exception as e:
            print("Warning: resize_token_embeddings failed:", e)

    print("🦎 odel hf_device_map:", getattr(model, "hf_device_map", None))

    model.config.use_cache = False
    print("🦎 Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    print("🦎 Applying LoRA adapters...")
    peft_conf = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_conf)

    mlflow.set_experiment("Mistral-instruct-LoRA")
    with mlflow.start_run(log_system_metrics=True):
        log_git()
        log_gpu()
        log_python()

        mlflow.log_params({
            "model_name": args.model_name,
            "dataset": args.dataset_name,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "seq_len": args.max_seq_length,
            "epochs": args.epochs,
            "max_memory": max_memory,
        })

        tokenized = prepare_dataset(tokenizer, args.dataset_name, max_length=args.max_seq_length, subset_pct=args.train_subset_pct)
        print("🦎 Dataset examples:", len(tokenized))

        adapter_dir = train_loop(
            model=model,
            tokenizer=tokenizer,
            train_dataset=tokenized,
            output_dir=args.output_dir,
            per_device_batch_size=args.per_device_train_batch_size,
            grad_accum_steps=args.gradient_accumulation_steps,
            epochs=args.epochs,
            lr=args.learning_rate,
            logging_steps=args.logging_steps,
            save_every_steps=args.save_every_steps,
        )

        print("🦎 Training complete. Adapter saved to:", adapter_dir)
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapters")


if __name__ == "__main__":
    main()