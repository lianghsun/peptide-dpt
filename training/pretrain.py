"""Phase 1: Pretraining gemma-3-1b from scratch on SELFIES corpus.

Uses Gemma-3-1b architecture with random initialization and
our custom SELFIES vocabulary.

Data path formats (auto-detected):
  data/processed/pretrain_train.jsonl          → local JSONL
  data/processed/pretrain/                     → HF dataset dir (snapshot_download)
  lianghsun/peptidomimetic-pretrain            → HF Hub repo ID

Usage:
    # Single GPU
    python -m training.pretrain --config configs/pretrain.yaml

    # Multi-GPU (B200x8)
    torchrun --nproc_per_node=8 -m training.pretrain --config configs/pretrain_b200.yaml
"""

from __future__ import annotations

import argparse
import logging

import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from tokenizer.selfies_tokenizer import SelfiesTokenizer
from training.dataset import SelfiesDataset, collate_fn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


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

    max_len = cfg["tokenizer"]["max_length"]
    train_ds = SelfiesDataset(cfg["data"]["train_path"], split="train", max_length=max_len)
    val_ds   = SelfiesDataset(cfg["data"]["val_path"],   split="validation", max_length=max_len)
    log.info(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}")

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
