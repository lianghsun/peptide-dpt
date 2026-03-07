"""Prepare pretraining dataset (Phase 1).

Combines all SELFIES sources, deduplicates, splits train/val,
and writes JSONL files for the Trainer.

Each line: {"selfies": "...", "input_ids": [...], "labels": [...]}
Labels = input_ids shifted by 1 (CLM).

Usage:
    python -m data.process.prepare_pretrain \
        --selfies_csvs data/processed/chembl_psma_selfies.csv \
                       data/processed/pdb_selfies.csv \
                       data/processed/bindingdb_selfies.csv \
        --vocab tokenizer/selfies_vocab.json \
        --output_dir data/processed \
        --val_ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from tokenizer.selfies_tokenizer import SelfiesTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MAX_LENGTH = 256
SEED = 42


def load_all_selfies(csv_paths: list[str]) -> list[str]:
    all_selfies = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "selfies" in df.columns:
            all_selfies.extend(df["selfies"].dropna().tolist())
    # Deduplicate
    unique = list(dict.fromkeys(all_selfies))
    log.info(f"Total unique SELFIES: {len(unique)} (from {len(all_selfies)} raw)")
    return unique


def make_clm_record(selfies_str: str, tokenizer: SelfiesTokenizer) -> dict | None:
    ids = tokenizer.encode(selfies_str, add_special_tokens=True, max_length=MAX_LENGTH)
    if len(ids) < 3:
        return None
    return {
        "selfies": selfies_str,
        "input_ids": ids[:-1],   # everything except last EOS as input
        "labels": ids[1:],        # shifted right (CLM target)
    }


def write_jsonl(records: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    log.info(f"Wrote {len(records)} records → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfies_csvs", nargs="+", required=True)
    parser.add_argument("--vocab", default="tokenizer/selfies_vocab.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    tokenizer = SelfiesTokenizer.load(args.vocab)
    log.info(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    all_selfies = load_all_selfies(args.selfies_csvs)

    records = []
    for s in tqdm(all_selfies, desc="Tokenizing"):
        rec = make_clm_record(s, tokenizer)
        if rec:
            records.append(rec)

    random.seed(SEED)
    random.shuffle(records)

    n_val = max(1, int(len(records) * args.val_ratio))
    val_records = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, f"{args.output_dir}/pretrain_train.jsonl")
    write_jsonl(val_records, f"{args.output_dir}/pretrain_val.jsonl")

    log.info(f"Train: {len(train_records)}, Val: {len(val_records)}")


if __name__ == "__main__":
    main()
