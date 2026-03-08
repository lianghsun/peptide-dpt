"""Phase 2: PSMA-specific supervised fine-tuning.

Loads Phase 1 checkpoint and fine-tunes on PSMA dataset
with activity-conditioned generation.

Usage:
    python -m training.finetune_psma --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import logging

import yaml
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tokenizer.selfies_tokenizer import SelfiesTokenizer
from training.dataset import SelfiesDataset, collate_fn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tokenizer = SelfiesTokenizer.load(cfg["tokenizer"]["path"])
    log.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    checkpoint = cfg["model"]["checkpoint"]
    log.info(f"Loading model from {checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    # Resize embeddings if vocab changed (potency tokens added)
    if model.config.vocab_size != tokenizer.vocab_size:
        log.info(
            f"Resizing embeddings: {model.config.vocab_size} → {tokenizer.vocab_size}"
        )
        model.resize_token_embeddings(tokenizer.vocab_size)

    max_len = cfg["tokenizer"].get("max_length", 256)
    train_ds = SelfiesDataset(cfg["data"]["train_path"], split="train", max_length=max_len)
    val_ds   = SelfiesDataset(cfg["data"]["val_path"],   split="validation", max_length=max_len)
    log.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

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
        bf16=t.get("bf16", True),
        logging_steps=t["logging_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t["eval_steps"],
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        deepspeed=t.get("deepspeed", None),
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name", "finetune"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, pad_id=tokenizer.pad_token_id),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=t.get("early_stopping_patience", 5)
            )
        ],
    )

    log.info("Starting PSMA fine-tuning...")
    trainer.train()
    trainer.save_model(f"{t['output_dir']}/best")
    log.info(f"Model saved → {t['output_dir']}/best")


if __name__ == "__main__":
    main()
