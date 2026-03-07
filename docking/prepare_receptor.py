"""Prepare 8BOW receptor PDB for docking.

Steps:
1. Download 8BOW from RCSB
2. Remove PSMA-617 (QYF) and all non-protein HETATM records
3. Keep only chain A (monomer, since Gnina handles this fine)
4. Compute binding pocket center from QYF centroid
5. Save receptor PDB and box center JSON

Usage:
    python -m docking.prepare_receptor --output_dir docking/
"""

from __future__ import annotations

import argparse
import json
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PDB_ID = "8BOW"
LIGAND_RESNAME = "QYF"
KEEP_CHAIN = "A"

# HETATM residues to keep (ions critical for catalysis are kept for receptor)
KEEP_HETATM = {"ZN", "CA"}

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str) -> str:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    log.info(f"Downloading {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def extract_ligand_coords(pdb_text: str, resname: str) -> np.ndarray:
    """Return (N, 3) array of ligand atom coordinates."""
    coords = []
    for line in pdb_text.splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == resname:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue
    if not coords:
        raise ValueError(f"No atoms found for ligand {resname}")
    return np.array(coords)


def clean_pdb(pdb_text: str, keep_chain: str, ligand_resname: str) -> str:
    """Remove ligand and solvent HETATM, keep only specified chain."""
    lines_out = []
    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record == "ATOM":
            chain = line[21]
            if chain == keep_chain:
                lines_out.append(line)
        elif record == "HETATM":
            resname = line[17:20].strip()
            chain = line[21]
            if resname in KEEP_HETATM and chain == keep_chain:
                lines_out.append(line)
            # Skip: ligand, water, glycans
        elif record in {"TER", "END", "REMARK", "SEQRES", "CRYST1", "ORIGX", "SCALE"}:
            lines_out.append(line)
    lines_out.append("END")
    return "\n".join(lines_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="docking")
    parser.add_argument("--pdb_id", default=PDB_ID)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_text = download_pdb(args.pdb_id)

    # Extract ligand centroid for box center
    try:
        lig_coords = extract_ligand_coords(pdb_text, LIGAND_RESNAME)
        center = lig_coords.mean(axis=0).tolist()
        log.info(f"Binding pocket center (from {LIGAND_RESNAME}): {center}")
    except ValueError as e:
        log.warning(f"{e} — using fallback center (0,0,0)")
        center = [0.0, 0.0, 0.0]

    # Save box config
    box_config = {
        "pdb_id": args.pdb_id,
        "ligand_resname": LIGAND_RESNAME,
        "center_x": center[0],
        "center_y": center[1],
        "center_z": center[2],
        "box_size": 22.0,
    }
    box_path = out_dir / "box_config.json"
    box_path.write_text(json.dumps(box_config, indent=2))
    log.info(f"Box config → {box_path}")

    # Clean and save receptor
    cleaned = clean_pdb(pdb_text, KEEP_CHAIN, LIGAND_RESNAME)
    receptor_path = out_dir / f"{args.pdb_id}_receptor.pdb"
    receptor_path.write_text(cleaned)
    log.info(f"Receptor PDB → {receptor_path}")

    # Save raw original too
    raw_path = out_dir / f"{args.pdb_id}_raw.pdb"
    raw_path.write_text(pdb_text)
    log.info(f"Raw PDB → {raw_path}")


if __name__ == "__main__":
    main()
