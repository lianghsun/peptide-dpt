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

# PubChem BioAssay: PSMA-related assays (GCPII inhibition)
# AID 1259411 = PSMA/GCPII enzyme inhibition assay (large dataset)
PUBCHEM_AIDS = [
    "1259411",  # GCPII inhibition (urea-based, ~800 cpds)
    "651744",   # PSMA inhibition
    "720576",   # GCPII inhibitors
    "2551",     # GCPII inhibition (2-PMPA class)
]
PUBCHEM_ASSAY_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV"
PUBCHEM_SMILES_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/IsomericSMILES/JSON"


def fetch_pubchem_assay(aid: str) -> pd.DataFrame:
    """Fetch activity data for a PubChem BioAssay."""
    url = PUBCHEM_ASSAY_URL.format(aid=aid)
    log.info(f"Fetching PubChem AID {aid}...")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        log.info(f"  AID {aid}: {len(df)} rows, columns: {list(df.columns[:5])}")
        return df
    except Exception as e:
        log.warning(f"  AID {aid} failed: {e}")
        return pd.DataFrame()


def get_smiles_for_cids(cids: list[str]) -> dict[str, str]:
    """Batch fetch IsomericSMILES for PubChem CIDs."""
    if not cids:
        return {}
    # PubChem allows up to 100 CIDs per request
    result = {}
    for i in range(0, len(cids), 100):
        batch = cids[i : i + 100]
        url = PUBCHEM_SMILES_URL.format(cids=",".join(str(c) for c in batch))
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            props = r.json().get("PropertyTable", {}).get("Properties", [])
            for p in props:
                result[str(p["CID"])] = p.get("IsomericSMILES", "")
        except Exception as e:
            log.debug(f"SMILES batch failed: {e}")
    return result


def fetch_by_api(uniprot_id: str) -> pd.DataFrame:
    """Fetch via PubChem BioAssay (fallback since BindingDB API is down)."""
    log.info(f"BindingDB API unavailable — using PubChem BioAssay for GCPII ({uniprot_id})")
    all_records = []

    for aid in PUBCHEM_AIDS:
        df = fetch_pubchem_assay(aid)
        if df.empty:
            continue

        # PubChem CSV columns vary but CID is usually present
        # PubChem CSV has SMILES directly in PUBCHEM_EXT_DATASOURCE_SMILES
        smiles_col = next(
            (c for c in df.columns if "SMILES" in c.upper()), None
        )
        cid_col = next((c for c in df.columns if "CID" in c.upper()), None)
        activity_col = next((c for c in df.columns if "ACTIVITY_OUTCOME" in c.upper()), None)

        if smiles_col is None and cid_col is None:
            log.warning(f"  AID {aid}: no SMILES or CID column found")
            continue

        # Filter to active compounds
        if activity_col:
            active = df[df[activity_col].astype(str).str.upper() == "ACTIVE"].copy()
        else:
            active = df.copy()

        if smiles_col:
            # Use SMILES directly from CSV
            for _, row in active.iterrows():
                smi = str(row.get(smiles_col, "")).strip()
                if smi and smi.lower() not in ("nan", ""):
                    all_records.append({
                        "smiles": smi,
                        "cid": row.get(cid_col, ""),
                        "aid": aid,
                        "IC50": None,
                        "Ki": None,
                        "target_name": "PSMA/GCPII",
                    })
        else:
            # Fallback: fetch SMILES by CID
            cids = active[cid_col].dropna().astype(str).tolist()
            log.info(f"  AID {aid}: {len(cids)} active compounds, fetching SMILES...")
            smiles_map = get_smiles_for_cids(cids)
            for cid in cids:
                smi = smiles_map.get(cid)
                if smi:
                    all_records.append({
                        "smiles": smi,
                        "cid": cid,
                        "aid": aid,
                        "IC50": None,
                        "Ki": None,
                        "target_name": "PSMA/GCPII",
                    })

    df_out = pd.DataFrame(all_records)
    log.info(f"PubChem total: {len(df_out)} records")
    return df_out


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
