"""Docking reward wrapper for GRPO.

Converts SELFIES → SMILES → docking score → normalized reward.

Reward normalization:
    Raw Gnina score is in kcal/mol (negative = better binding).
    We clip to [-12, 0] and normalize to [0, 1].
    score = -12 → reward = 1.0 (very strong binding)
    score =   0 → reward = 0.0 (no binding)
"""

from __future__ import annotations

import logging

import selfies as sf

from docking.run_gnina import GninaDocking

log = logging.getLogger(__name__)

SCORE_MIN = -12.0   # kcal/mol — strong binder
SCORE_MAX = 0.0     # kcal/mol — no binding


def _normalize_score(score: float) -> float:
    """Map [SCORE_MIN, SCORE_MAX] → [1, 0]."""
    clipped = max(SCORE_MIN, min(SCORE_MAX, score))
    return (SCORE_MAX - clipped) / (SCORE_MAX - SCORE_MIN)


class DockingReward:
    def __init__(self, **gnina_kwargs):
        self.docker = GninaDocking(**gnina_kwargs)

    def selfies_to_smiles(self, selfies_str: str) -> str | None:
        try:
            smi = sf.decoder(selfies_str)
            return smi
        except Exception:
            return None

    def __call__(self, selfies_str: str) -> float:
        """Return docking reward in [0, 1]. 0 on any failure."""
        smiles = self.selfies_to_smiles(selfies_str)
        if smiles is None:
            return 0.0

        score = self.docker(smiles)
        if score is None:
            return 0.0

        reward = _normalize_score(score)
        log.debug(f"Docking: {smiles[:30]}... score={score:.2f} reward={reward:.3f}")
        return reward

    def batch(self, selfies_list: list[str], n_workers: int = 4) -> list[float]:
        smiles_list = [self.selfies_to_smiles(s) or "" for s in selfies_list]
        raw_scores = self.docker.batch(smiles_list, n_workers=n_workers)
        rewards = []
        for score in raw_scores:
            if score is None:
                rewards.append(0.0)
            else:
                rewards.append(_normalize_score(score))
        return rewards
