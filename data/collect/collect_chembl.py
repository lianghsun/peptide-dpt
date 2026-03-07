"""Collect GCPII/PSMA inhibitor data from ChEMBL.

Target: CHEMBL3231 (Glutamate carboxypeptidase II / PSMA)
Outputs: data/raw/chembl_psma.csv
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PSMA_CHEMBL_ID = "CHEMBL3231"
OUTPUT_PATH = Path("data/raw/chembl_psma.csv")
ACTIVITY_TYPES = {"IC50", "Ki", "Kd", "inhibition"}


def fetch_psma_activities() -> pd.DataFrame:
    log.info(f"Querying ChEMBL target {PSMA_CHEMBL_ID}...")
    activity_client = new_client.activity

    records = []
    for atype in ACTIVITY_TYPES:
        log.info(f"  Fetching {atype} data...")
        try:
            res = activity_client.filter(
                target_chembl_id=PSMA_CHEMBL_ID,
                standard_type=atype,
            ).only(
                [
                    "molecule_chembl_id",
                    "canonical_smiles",
                    "standard_type",
                    "standard_value",
                    "standard_units",
                    "pchembl_value",
                    "assay_chembl_id",
                    "document_chembl_id",
                ]
            )
            for r in tqdm(res, desc=atype):
                if r.get("canonical_smiles"):
                    records.append(r)
        except Exception as e:
            log.warning(f"  Failed to fetch {atype}: {e}")

    df = pd.DataFrame(records)
    log.info(f"Total records: {len(df)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df = df[df["standard_value"].apply(lambda x: _is_numeric(x))]
    df["standard_value"] = df["standard_value"].astype(float)
    # Keep only nM-range: standard_units == 'nM' or convert uM
    df = df[df["standard_units"].isin(["nM", "uM", "pM"])]
    df["value_nM"] = df.apply(_to_nM, axis=1)
    df = df.dropna(subset=["value_nM"])
    df = df[df["value_nM"] > 0]
    df = df.drop_duplicates(subset=["canonical_smiles"])
    return df.reset_index(drop=True)


def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _to_nM(row) -> float | None:
    v = float(row["standard_value"])
    u = row["standard_units"]
    if u == "nM":
        return v
    elif u == "uM":
        return v * 1000
    elif u == "pM":
        return v / 1000
    return None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_psma_activities()
    df = clean(df)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(df)} compounds → {OUTPUT_PATH}")
    log.info(df[["canonical_smiles", "standard_type", "value_nM"]].head())


if __name__ == "__main__":
    main()
