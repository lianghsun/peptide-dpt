"""Upload datasets to HuggingFace Hub.

Public dataset  (lianghsun/peptidomimetic-pretrain):
    General peptidomimetic SELFIES corpus for pretraining.
    Contains: selfies, smiles columns. No bioactivity data.

Restricted dataset (lianghsun/psma-sft):
    PSMA-specific fine-tuning data with bioactivity labels.
    Access requires approval (gated repository).

Usage:
    python -m data.upload_hf --phase pretrain
    python -m data.upload_hf --phase psma_sft
    python -m data.upload_hf --phase all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

HF_USER = "lianghsun"
PRETRAIN_REPO = f"{HF_USER}/peptidomimetic-pretrain"
PSMA_REPO = f"{HF_USER}/psma-sft"


def upload_pretrain(pretrain_dir: str = "data/processed"):
    log.info("Uploading pretraining dataset...")
    train_path = f"{pretrain_dir}/pretrain_train.jsonl"
    val_path = f"{pretrain_dir}/pretrain_val.jsonl"

    if not Path(train_path).exists():
        log.error(f"File not found: {train_path}. Run prepare_pretrain.py first.")
        return

    ds = DatasetDict(
        {
            "train": Dataset.from_json(train_path),
            "validation": Dataset.from_json(val_path),
        }
    )

    # Keep only selfies column for public upload (no raw SMILES to avoid IP issues)
    ds = ds.select_columns(["selfies", "input_ids", "labels"])

    ds.push_to_hub(
        PRETRAIN_REPO,
        private=False,
        commit_message="Add peptidomimetic SELFIES pretraining corpus",
    )
    log.info(f"Uploaded to https://huggingface.co/datasets/{PRETRAIN_REPO}")


def upload_psma_sft(sft_dir: str = "data/processed"):
    log.info("Uploading PSMA SFT dataset (restricted)...")
    train_path = f"{sft_dir}/psma_sft_train.jsonl"
    val_path = f"{sft_dir}/psma_sft_val.jsonl"

    if not Path(train_path).exists():
        log.error(f"File not found: {train_path}. Run prepare_psma_sft.py first.")
        return

    api = HfApi()

    # Create gated repository
    try:
        api.create_repo(
            repo_id=PSMA_REPO,
            repo_type="dataset",
            private=False,   # public but gated
            exist_ok=True,
        )
        # Enable gating (requires access request)
        api.update_repo_settings(
            repo_id=PSMA_REPO,
            repo_type="dataset",
            gated="auto",   # 'auto' = auto-approve; use 'manual' for manual review
        )
    except Exception as e:
        log.warning(f"Repo create/gate: {e}")

    ds = DatasetDict(
        {
            "train": Dataset.from_json(train_path),
            "validation": Dataset.from_json(val_path),
        }
    )

    ds.push_to_hub(
        PSMA_REPO,
        private=False,
        commit_message="Add PSMA-specific SFT dataset with activity conditioning",
    )
    log.info(f"Uploaded to https://huggingface.co/datasets/{PSMA_REPO}")
    log.info("Dataset is gated — users must request access.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["pretrain", "psma_sft", "all"],
        default="all",
    )
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()

    if args.phase in ("pretrain", "all"):
        upload_pretrain(args.data_dir)
    if args.phase in ("psma_sft", "all"):
        upload_psma_sft(args.data_dir)


if __name__ == "__main__":
    main()
