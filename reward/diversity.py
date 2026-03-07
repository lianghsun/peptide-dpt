"""Diversity reward: encourages exploration away from known PSMA binders.

Uses Tanimoto distance to the nearest known PSMA binder.
Reward = average min-distance to reference set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

log = logging.getLogger(__name__)


def _smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


class DiversityReward:
    def __init__(
        self,
        reference_smiles_path: str,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.ref_fps = self._load_references(reference_smiles_path)
        log.info(f"Loaded {len(self.ref_fps)} reference fingerprints")

    def _load_references(self, path: str) -> list:
        fps = []
        smiles_list = Path(path).read_text().strip().splitlines()
        for smi in smiles_list:
            fp = _smiles_to_fp(smi.strip(), self.radius, self.n_bits)
            if fp is not None:
                fps.append(fp)
        return fps

    def __call__(self, smiles: str) -> float:
        """Return diversity reward in [0, 1].

        0 = identical to a known PSMA binder (bad for exploration)
        1 = maximally different from all known binders
        """
        if not self.ref_fps:
            return 0.5  # no reference, neutral

        fp = _smiles_to_fp(smiles, self.radius, self.n_bits)
        if fp is None:
            return 0.0

        sims = DataStructs.BulkTanimotoSimilarity(fp, self.ref_fps)
        max_sim = max(sims) if sims else 0.0
        # Diversity = 1 - max_similarity
        # But we also don't want completely random → soft target around 0.3–0.6 similarity
        # Use bell-shaped reward centered at sim=0.4 (novel but related)
        target_sim = 0.4
        diversity = 1.0 - abs(max_sim - target_sim) / target_sim
        return float(max(0.0, min(1.0, diversity)))

    def batch(self, smiles_list: list[str]) -> list[float]:
        return [self(s) for s in smiles_list]
