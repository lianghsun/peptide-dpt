"""Prepare PSMA-specific supervised fine-tuning dataset (Phase 2).

Merges ChEMBL, PDB ligand, and BindingDB PSMA data.
Applies activity-conditioning: prepends a potency token to each sequence.

Potency bins (nM):
    <10    → [VERY_POTENT]
    10–100 → [POTENT]
    100–1000 → [MODERATE]
    >1000  → [WEAK]
    unknown → (no conditioning token)

Output format (JSONL):
    {"selfies": "...", "prompt": "[POTENT]", "input_ids": [...], "labels": [...]}

Usage:
    python -m data.process.prepare_psma_sft \
        --selfies_csvs data/processed/chembl_psma_selfies.csv \
                       data/processed/pdb_selfies.csv \
                       data/processed/bindingdb_psma_selfies.csv \
        --vocab tokenizer/selfies_vocab.json \
        --output_dir data/processed
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
VAL_RATIO = 0.1

POTENCY_TOKENS = {
    "very_potent": "[VERY_POTENT]",   # < 10 nM
    "potent": "[POTENT]",             # 10–100 nM
    "moderate": "[MODERATE]",         # 100–1000 nM
    "weak": "[WEAK]",                 # > 1000 nM
}


def _potency_bin(value_nM: float | None) -> str | None:
    if value_nM is None or pd.isna(value_nM):
        return None
    if value_nM < 10:
        return "very_potent"
    if value_nM < 100:
        return "potent"
    if value_nM < 1000:
        return "moderate"
    return "weak"


def load_psma_data(csv_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "selfies" not in df.columns:
            log.warning(f"{p}: no 'selfies' column, skipping")
            continue
        # Normalise affinity column
        for col in ["value_nM", "affinity_nM", "IC50", "Ki", "Kd"]:
            if col in df.columns:
                df["affinity_nM_norm"] = pd.to_numeric(df[col], errors="coerce")
                break
        else:
            df["affinity_nM_norm"] = None
        dfs.append(df[["selfies", "affinity_nM_norm"]].copy())

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["selfies"])
    combined = combined.drop_duplicates(subset=["selfies"])
    log.info(f"PSMA dataset: {len(combined)} unique compounds")
    return combined


def make_sft_record(
    selfies_str: str,
    tokenizer: SelfiesTokenizer,
    potency_bin: str | None,
) -> dict | None:
    # Potency conditioning: prepend special token as first token after BOS
    # The tokenizer vocab includes POTENCY_TOKENS as SELFIES won't contain them,
    # so we handle them as literal prefix strings in the raw SELFIES.
    # Strategy: if potency known, prepend token string before SELFIES encoding.
    # Since our tokenizer is token-based, we handle this at the string level:
    # full_selfies = "[POTENT]" + selfies_str (treated as separate token).

    # Build ids manually to support potency prefix
    mol_ids = tokenizer.encode(selfies_str, add_special_tokens=False, max_length=MAX_LENGTH - 3)
    if len(mol_ids) < 2:
        return None

    potency_token = POTENCY_TOKENS.get(potency_bin) if potency_bin else None
    potency_id = tokenizer.token2id.get(potency_token) if potency_token else None

    if potency_id is not None:
        input_ids = [tokenizer.bos_token_id, potency_id] + mol_ids + [tokenizer.eos_token_id]
    else:
        input_ids = [tokenizer.bos_token_id] + mol_ids + [tokenizer.eos_token_id]

    return {
        "selfies": selfies_str,
        "potency_bin": potency_bin,
        "prompt": potency_token or "",
        "input_ids": input_ids[:-1],
        "labels": input_ids[1:],
    }


def extend_tokenizer_with_potency(tokenizer: SelfiesTokenizer, vocab_path: str):
    """Add potency tokens to vocab if not present, and save."""
    changed = False
    for tok in POTENCY_TOKENS.values():
        if tok not in tokenizer.token2id:
            new_id = tokenizer.vocab_size
            tokenizer._selfies_vocab[tok] = new_id
            tokenizer._selfies_id2token[new_id] = tok
            changed = True
    if changed:
        tokenizer.save(vocab_path)
        log.info(f"Extended tokenizer with potency tokens → {vocab_path}")


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
    args = parser.parse_args()

    tokenizer = SelfiesTokenizer.load(args.vocab)
    extend_tokenizer_with_potency(tokenizer, args.vocab)

    df = load_psma_data(args.selfies_csvs)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building SFT records"):
        bin_ = _potency_bin(row["affinity_nM_norm"])
        rec = make_sft_record(row["selfies"], tokenizer, bin_)
        if rec:
            records.append(rec)

    random.seed(SEED)
    random.shuffle(records)

    n_val = max(1, int(len(records) * VAL_RATIO))
    write_jsonl(records[n_val:], f"{args.output_dir}/psma_sft_train.jsonl")
    write_jsonl(records[:n_val], f"{args.output_dir}/psma_sft_val.jsonl")

    # Save known SMILES for diversity reward (GRPO phase)
    import selfies as sf
    known_smiles = []
    for s in df["selfies"].dropna():
        try:
            smi = sf.decoder(s)
            if smi:
                known_smiles.append(smi)
        except Exception:
            continue
    out_smiles = Path(args.output_dir) / "psma_known_smiles.txt"
    out_smiles.write_text("\n".join(known_smiles))
    log.info(f"Saved {len(known_smiles)} reference SMILES → {out_smiles}")


if __name__ == "__main__":
    main()
