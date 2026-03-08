"""Shared dataset loader supporting both JSONL and HuggingFace parquet formats.

Auto-detection logic:
  *.jsonl            → read line by line
  directory/         → load_from_disk (HF snapshot_download)
  user/repo-name     → load_dataset from HF Hub
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def _load_records(path: str, split: str = "train") -> list[dict]:
    p = Path(path)

    # Case 1: JSONL file
    if p.suffix == ".jsonl":
        records = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        log.info(f"Loaded {len(records):,} records from JSONL: {path}")
        return records

    # Case 2: Local HuggingFace dataset directory (from snapshot_download)
    if p.is_dir():
        from datasets import load_from_disk
        ds = load_from_disk(str(p))
        # DatasetDict → pick split; Dataset → use directly
        if hasattr(ds, "keys"):
            ds = ds[split]
        records = list(ds)
        log.info(f"Loaded {len(records):,} records from HF disk: {path} (split={split})")
        return records

    # Case 3: HuggingFace Hub repo ID (e.g. "lianghsun/peptidomimetic-pretrain")
    from datasets import load_dataset
    ds = load_dataset(path, split=split)
    records = list(ds)
    log.info(f"Loaded {len(records):,} records from HF Hub: {path} (split={split})")
    return records


class SelfiesDataset(Dataset):
    """Works with JSONL or HF parquet. Each record must have input_ids + labels."""

    def __init__(self, path: str, split: str = "train", max_length: int = 256):
        self.records = _load_records(path, split=split)
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
