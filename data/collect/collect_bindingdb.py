"""Collect PSMA/GCPII data from BindingDB via UniProt ID Q04609.

Downloads the BindingDB TSV dump for GCPII and extracts
ligand SMILES + binding affinity data.
Outputs: data/raw/bindingdb_psma.csv
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

UNIPROT_ID = "Q04609"
OUTPUT_PATH = Path("data/raw/bindingdb_psma.csv")

# BindingDB REST API for UniProt-based query
BDB_API = (
    "https://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprots"
    "?uniprot={uid}&cutoff=10000&response=application/json"
)


def fetch_by_api(uniprot_id: str) -> pd.DataFrame:
    """Try BindingDB REST API."""
    url = BDB_API.format(uid=uniprot_id)
    log.info(f"Querying BindingDB API for {uniprot_id}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        affinities = data.get("affinities", [])
        records = []
        for entry in affinities:
            smiles = entry.get("smile") or entry.get("smiles")
            if not smiles:
                continue
            records.append(
                {
                    "smiles": smiles,
                    "monomerid": entry.get("monomerid"),
                    "IC50": entry.get("IC50"),
                    "Ki": entry.get("Ki"),
                    "Kd": entry.get("Kd"),
                    "EC50": entry.get("EC50"),
                    "kon": entry.get("kon"),
                    "koff": entry.get("koff"),
                    "target_name": entry.get("target_name"),
                    "doi": entry.get("doi"),
                }
            )
        df = pd.DataFrame(records)
        log.info(f"API returned {len(df)} records")
        return df
    except Exception as e:
        log.warning(f"API failed: {e}")
        return pd.DataFrame()


def best_affinity_nM(row: pd.Series) -> tuple[str, float] | tuple[None, None]:
    """Pick the best available affinity value, convert to nM."""
    for col in ["Ki", "IC50", "Kd", "EC50"]:
        val = row.get(col)
        if val and val not in ("", ">10000", None):
            try:
                v = float(str(val).replace(">", "").replace("<", "").strip())
                return col, v
            except ValueError:
                continue
    return None, None


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["smiles"])
    df = df[df["smiles"].str.len() > 5]

    rows = []
    for _, row in df.iterrows():
        atype, val = best_affinity_nM(row)
        rows.append(
            {
                "smiles": row["smiles"],
                "affinity_type": atype,
                "affinity_nM": val,
                "target_name": row.get("target_name", "PSMA"),
                "doi": row.get("doi"),
            }
        )
    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["smiles"])
    return out.reset_index(drop=True)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_by_api(UNIPROT_ID)
    df = clean(df)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(df)} compounds → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
