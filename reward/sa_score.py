"""Synthetic Accessibility (SA) Score reward.

Uses RDKit's SA score implementation.
Returns a normalized reward in [0, 1] where 1 = most synthesizable.
"""

from __future__ import annotations

import math
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys

# Load RDKit SA score module
_sa_score_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
if _sa_score_path not in sys.path:
    sys.path.append(_sa_score_path)

try:
    import sascorer
    _HAS_SASCORER = True
except ImportError:
    _HAS_SASCORER = False


def sa_score(smiles: str) -> float | None:
    """Return SA score in [1, 10] (lower = more synthesizable). None if invalid."""
    if not _HAS_SASCORER:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return None


def sa_reward(smiles: str) -> float:
    """Normalized SA reward in [0, 1]. 1 = most synthesizable (score ≈ 1)."""
    score = sa_score(smiles)
    if score is None:
        return 0.0
    # SA score range: 1 (easy) to 10 (hard)
    # Map to [0, 1]: reward = (10 - score) / 9
    return (10.0 - score) / 9.0


def batch_sa_reward(smiles_list: list[str]) -> list[float]:
    return [sa_reward(s) for s in smiles_list]
