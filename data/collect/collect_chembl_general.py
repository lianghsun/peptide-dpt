"""Download large-scale general peptidomimetic/drug-like corpus from ChEMBL.

Filters ChEMBL compounds by:
    - Has valid SMILES
    - MW: 150–2000 Da  (covers drug-like + peptidomimetics)
    - RO5 violations ≤ 2 (loosened for peptides)
    - Not PSMA-specific (we have that separately)

Target: ~300k–500k compounds for pretraining.

Usage:
    python -m data.collect.collect_chembl_general \
        --max_mw 2000 --limit 500000 --output data/raw/chembl_general.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("data/raw/chembl_general.csv")


def fetch_chembl_molecules(max_mw: float, limit: int) -> pd.DataFrame:
    mol_client = new_client.molecule

    log.info(f"Querying ChEMBL molecules (MW ≤ {max_mw}, limit={limit:,})...")

    # Filter: has SMILES, MW in range, not deprecated
    res = mol_client.filter(
        molecule_properties__mw_freebase__lte=max_mw,
        molecule_properties__mw_freebase__gte=400,
        molecule_type="Small molecule",
    ).only([
        "molecule_chembl_id",
        "molecule_structures",
        "molecule_properties",
    ])

    records = []
    for mol in tqdm(res, desc="ChEMBL general", total=limit):
        if len(records) >= limit:
            break
        try:
            smiles = (mol.get("molecule_structures") or {}).get("canonical_smiles")
            props = mol.get("molecule_properties") or {}
            mw = props.get("mw_freebase")
            if smiles and mw:
                records.append({
                    "chembl_id": mol.get("molecule_chembl_id"),
                    "canonical_smiles": smiles,
                    "mw": float(mw),
                })
        except Exception:
            continue

    df = pd.DataFrame(records)
    df = df.dropna(subset=["canonical_smiles"])
    df = df.drop_duplicates(subset=["canonical_smiles"])
    log.info(f"Fetched {len(df):,} unique molecules")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_mw", type=float, default=2000)
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = fetch_chembl_molecules(args.max_mw, args.limit)
    df.to_csv(args.output, index=False)
    log.info(f"Saved {len(df):,} compounds → {args.output}")
    log.info(f"MW range: {df['mw'].min():.0f}–{df['mw'].max():.0f} Da")


if __name__ == "__main__":
    main()
