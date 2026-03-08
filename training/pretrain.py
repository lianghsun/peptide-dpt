"""Phase 1: Pretraining gemma-3-1b from scratch on SELFIES corpus.

Uses Gemma-3-1b architecture with random initialization and
our custom SELFIES vocabulary.

Usage:
    python -m training.pretrain --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from tokenizer.selfies_tokenizer import SelfiesTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SelfiesDataset(Dataset):
    def __init__(self, jsonl_path: str, max_length: int = 256):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                self.records.append(r)
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        input_ids = r["input_ids"][: self.max_length]
        labels = r["labels"][: self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        n = b["input_ids"].size(0)
        pad = max_len - n
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def build_model(config_dict: dict, vocab_size: int) -> AutoModelForCausalLM:
    log.info(f"Loading Gemma-3-1b config from {config_dict['model']['architecture']}...")
    cfg = AutoConfig.from_pretrained(
        config_dict["model"]["architecture"],
        trust_remote_code=True,
    )
    cfg.vocab_size = vocab_size
    cfg.pad_token_id = 0
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2

    log.info("Initializing model with random weights (from scratch)...")
    model = AutoModelForCausalLM.from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model parameters: {n_params:.1f}M")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tokenizer = SelfiesTokenizer.load(cfg["tokenizer"]["path"])
    log.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    train_ds = SelfiesDataset(cfg["data"]["train_path"], cfg["tokenizer"]["max_length"])
    val_ds = SelfiesDataset(cfg["data"]["val_path"], cfg["tokenizer"]["max_length"])
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    model = build_model(cfg, tokenizer.vocab_size)

    t = cfg["training"]
    training_args = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        fp16=t.get("fp16", False),
        bf16=t.get("bf16", True),
        logging_steps=t["logging_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t["eval_steps"],
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        deepspeed=t.get("deepspeed", None),
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name", "pretrain"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, pad_id=tokenizer.pad_token_id),
    )

    log.info("Starting pretraining...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(f"{t['output_dir']}/best")
    log.info(f"Model saved → {t['output_dir']}/best")


if __name__ == "__main__":
    main()
