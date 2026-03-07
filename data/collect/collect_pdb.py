"""Collect GCPII co-crystal ligand SMILES from RCSB PDB.

Queries all structures for UniProt Q04609 (GCPII/PSMA), extracts
ligand SMILES from PDB Chemical Component Dictionary (CCD).
Outputs: data/raw/pdb_psma_ligands.csv
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

UNIPROT_ID = "Q04609"
OUTPUT_PATH = Path("data/raw/pdb_psma_ligands.csv")

# Known GCPII PDB entries with their key ligands
KNOWN_ENTRIES = {
    "8BOW": "QYF",  # PSMA-617
    "8BO8": None,   # P17
    "8BOL": None,   # P18
    "1Z8L": "GCP",  # glutamate
    "2JBJ": "PMP",  # 2-PMPA
    "4LQG": None,   # CTT1056
    "4NGN": None,   # urea-based
    "4NGP": None,   # urea-based
    "6S1X": None,
    "6H7Z": None,
    "6HKZ": None,
    "6HKJ": None,
}

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA = "https://data.rcsb.org/rest/v1/core"
CCD_SMILES = "https://files.rcsb.org/ligands/download/{ccd_id}_ideal.sdf"


def search_uniprot_structures(uniprot_id: str) -> list[str]:
    """Return all PDB entry IDs associated with a UniProt accession."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
        "request_options": {"results_verbosity": "compact", "return_all_hits": True},
    }
    r = requests.post(RCSB_SEARCH, json=query, timeout=30)
    r.raise_for_status()
    data = r.json()
    result_set = data.get("result_set", [])
    # result_set is a list of strings (entry IDs), not dicts
    if result_set and isinstance(result_set[0], str):
        return result_set
    return [h["identifier"] for h in result_set]


def get_entry_ligands(pdb_id: str) -> list[dict]:
    """Get non-polymer ligand info for a PDB entry."""
    url = f"{RCSB_DATA}/entry/{pdb_id.upper()}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"{pdb_id}: failed to fetch entry data: {e}")
        return []

    # Get non-polymer entity IDs
    nonpoly = data.get("rcsb_entry_container_identifiers", {}).get(
        "non_polymer_entity_ids", []
    )
    ligands = []
    for eid in nonpoly:
        entity_url = f"{RCSB_DATA}/nonpolymer_entity/{pdb_id.upper()}/{eid}"
        try:
            er = requests.get(entity_url, timeout=10)
            er.raise_for_status()
            ed = er.json()
            ccd_id = (
                ed.get("pdbx_entity_nonpoly", {}).get("comp_id")
                or ed.get("rcsb_nonpolymer_entity", {}).get("pdbx_description")
            )
            if ccd_id and ccd_id not in {"HOH", "ZN", "CA", "CL", "NAG", "FUC", "MAN"}:
                ligands.append({"pdb_id": pdb_id, "ccd_id": ccd_id, "entity_id": eid})
        except Exception:
            continue
    return ligands


def get_smiles_from_ccd(ccd_id: str) -> str | None:
    """Fetch isomeric SMILES from RCSB CCD REST API."""
    url = f"{RCSB_DATA}/chemcomp/{ccd_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Try descriptors
        descriptors = data.get("pdbx_chem_comp_descriptor", [])
        for d in descriptors:
            if d.get("type") == "SMILES_CANONICAL" and d.get("program") == "OpenEye OEToolkits":
                return d.get("descriptor")
        for d in descriptors:
            if "SMILES" in d.get("type", ""):
                return d.get("descriptor")
    except Exception as e:
        log.debug(f"CCD {ccd_id}: {e}")
    return None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Searching PDB for UniProt {UNIPROT_ID}...")
    pdb_ids = search_uniprot_structures(UNIPROT_ID)
    # Merge with known entries
    all_ids = list(set(pdb_ids) | set(KNOWN_ENTRIES.keys()))
    log.info(f"Found {len(all_ids)} PDB entries")

    records = []
    for pdb_id in tqdm(all_ids, desc="Fetching ligands"):
        for lig in get_entry_ligands(pdb_id):
            smiles = get_smiles_from_ccd(lig["ccd_id"])
            if smiles:
                lig["smiles"] = smiles
                records.append(lig)

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["smiles"])
    df.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(df)} ligands → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
