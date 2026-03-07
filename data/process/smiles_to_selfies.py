"""Convert SMILES → SELFIES and build tokenizer vocabulary.

Usage:
    python -m data.process.smiles_to_selfies \
        --input data/raw/chembl_psma.csv \
        --smiles_col canonical_smiles \
        --output data/processed/chembl_psma_selfies.csv

Also builds tokenizer/selfies_vocab.json from the full corpus.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import selfies as sf
from tqdm import tqdm

from tokenizer.selfies_tokenizer import SelfiesTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def smiles_to_selfies_safe(smiles: str) -> str | None:
    try:
        s = sf.encoder(smiles)
        # Validate round-trip
        recovered = sf.decoder(s)
        if recovered is None:
            return None
        return s
    except Exception:
        return None


def convert_file(
    input_path: str,
    smiles_col: str,
    output_path: str,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    log.info(f"Input: {len(df)} rows from {input_path}")

    selfies_col = []
    for smi in tqdm(df[smiles_col], desc="SMILES→SELFIES"):
        selfies_col.append(smiles_to_selfies_safe(str(smi)) if pd.notna(smi) else None)

    df["selfies"] = selfies_col
    valid = df.dropna(subset=["selfies"])
    log.info(f"Valid conversions: {len(valid)}/{len(df)} ({100*len(valid)/len(df):.1f}%)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(output_path, index=False)
    log.info(f"Saved → {output_path}")
    return valid


def build_vocab_from_files(csv_paths: list[str], vocab_out: str = "tokenizer/selfies_vocab.json"):
    """Merge multiple SELFIES CSV files and build vocabulary."""
    all_selfies = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "selfies" in df.columns:
            all_selfies.extend(df["selfies"].dropna().tolist())

    log.info(f"Building vocab from {len(all_selfies)} SELFIES strings...")
    tok = SelfiesTokenizer.from_corpus(all_selfies)
    # Extend with full alphabet to cover unseen tokens
    full_alphabet = sorted(sf.get_semantic_robust_alphabet())
    next_id = tok.vocab_size
    for sym in full_alphabet:
        if sym not in tok.token2id:
            tok.token2id[sym] = next_id
            tok.id2token[next_id] = sym
            next_id += 1

    tok.save(vocab_out)
    log.info(f"Vocab size: {tok.vocab_size} → {vocab_out}")
    return tok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--smiles_col", default="canonical_smiles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--build_vocab", action="store_true")
    parser.add_argument("--vocab_out", default="tokenizer/selfies_vocab.json")
    args = parser.parse_args()

    convert_file(args.input, args.smiles_col, args.output)

    if args.build_vocab:
        build_vocab_from_files([args.output], args.vocab_out)
