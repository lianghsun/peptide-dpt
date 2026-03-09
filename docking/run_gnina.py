"""Gnina docking wrapper.

Converts SMILES to 3D SDF (via RDKit), runs Gnina, parses score.

Requirements:
    - gnina binary in PATH (https://github.com/gnina/gnina/releases)
    - docking/8BOW_receptor.pdb  (from prepare_receptor.py)
    - docking/box_config.json

Usage:
    from docking.run_gnina import GninaDocking
    docker = GninaDocking()
    score = docker("CCO")   # returns docking score in kcal/mol (negative = better)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

DEFAULT_RECEPTOR = "docking/8BOW_receptor.pdb"
DEFAULT_BOX_CONFIG = "docking/box_config.json"
GNINA_BIN = os.environ.get("GNINA_BIN", "gnina")


def smiles_to_3d_sdf(smiles: str, out_path: str) -> bool:
    """Generate 3D conformer from SMILES and write SDF. Returns True on success."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback to random coordinates
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result != 0:
        return False
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    writer = Chem.SDWriter(out_path)
    writer.write(mol)
    writer.close()
    return True


def parse_gnina_score(stdout: str) -> float | None:
    """Parse CNNscore or minimizedAffinity from Gnina stdout."""
    for line in stdout.splitlines():
        # Gnina output: "   1     -8.5     0.95     0.85"
        # columns: rank, affinity, CNNscore, CNNaffinity
        line = line.strip()
        if line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])  # minimizedAffinity kcal/mol
                except ValueError:
                    continue
    return None


class GninaDocking:
    def __init__(
        self,
        receptor: str = DEFAULT_RECEPTOR,
        box_config: str = DEFAULT_BOX_CONFIG,
        exhaustiveness: int = 8,
        gnina_bin: str = GNINA_BIN,
    ):
        self.receptor = receptor
        self.exhaustiveness = exhaustiveness
        self.gnina_bin = gnina_bin

        if not shutil.which(gnina_bin):
            log.warning(
                f"'{gnina_bin}' not found in PATH. "
                "Install from https://github.com/gnina/gnina/releases"
            )

        with open(box_config) as f:
            bc = json.load(f)
        self.cx = bc["center_x"]
        self.cy = bc["center_y"]
        self.cz = bc["center_z"]
        self.box_size = bc["box_size"]

    def __call__(self, smiles: str) -> float | None:
        """Dock SMILES against PSMA receptor. Returns kcal/mol (None on failure)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_sdf = os.path.join(tmpdir, "ligand.sdf")
            out_sdf = os.path.join(tmpdir, "out.sdf")

            if not smiles_to_3d_sdf(smiles, ligand_sdf):
                log.debug(f"3D generation failed: {smiles[:30]}...")
                return None

            cmd = [
                self.gnina_bin,
                "--receptor", self.receptor,
                "--ligand", ligand_sdf,
                "--center_x", str(self.cx),
                "--center_y", str(self.cy),
                "--center_z", str(self.cz),
                "--size_x", str(self.box_size),
                "--size_y", str(self.box_size),
                "--size_z", str(self.box_size),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", "1",
                "--out", out_sdf,
                "--quiet",
            ]

            # Force gnina onto CPU so it doesn't compete with the LLM for VRAM.
            # Each torchrun rank already occupies one GPU; gnina runs fast
            # enough on CPU for single-ligand docking (~5-30 s per call).
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120, env=env
                )
                score = parse_gnina_score(result.stdout)
                if score is None:
                    log.debug(f"Could not parse score. stdout: {result.stdout[:200]}")
                return score
            except subprocess.TimeoutExpired:
                log.warning("Gnina timed out")
                return None
            except FileNotFoundError:
                log.error(f"Gnina binary '{self.gnina_bin}' not found")
                return None

    def batch(self, smiles_list: list[str], n_workers: int = 4) -> list[float | None]:
        """Batch docking with multiprocessing."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(self.__call__, smiles_list))
