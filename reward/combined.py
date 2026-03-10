"""Combined reward function for GRPO.

R_total = w_dock * R_docking + w_sa * R_sa + w_div * R_diversity

All individual rewards are in [0, 1].
"""

from __future__ import annotations

import logging

import selfies as sf

from reward.docking import DockingReward
from reward.sa_score import sa_reward
from reward.diversity import DiversityReward

log = logging.getLogger(__name__)


class CombinedReward:
    def __init__(
        self,
        receptor_pdb: str,
        box_config: str,
        reference_smiles_path: str,
        weights: dict[str, float] | None = None,
        exhaustiveness: int = 2,
    ):
        self.weights = weights or {"docking": 0.60, "sa_score": 0.25, "diversity": 0.15}
        assert abs(sum(self.weights.values()) - 1.0) < 1e-6, "Weights must sum to 1"

        self.docking = DockingReward(
            receptor=receptor_pdb,
            box_config=box_config,
            exhaustiveness=exhaustiveness,
        )
        self.diversity = DiversityReward(reference_smiles_path)

    def __call__(self, selfies_str: str) -> float:
        """Compute combined reward for a single SELFIES string."""
        # SELFIES → SMILES (needed by SA and diversity)
        try:
            smiles = sf.decoder(selfies_str)
        except Exception:
            smiles = None

        r_dock = self.docking(selfies_str)
        r_sa = sa_reward(smiles) if smiles else 0.0
        r_div = self.diversity(smiles) if smiles else 0.0

        total = (
            self.weights["docking"] * r_dock
            + self.weights["sa_score"] * r_sa
            + self.weights["diversity"] * r_div
        )

        log.debug(
            f"dock={r_dock:.3f} sa={r_sa:.3f} div={r_div:.3f} → total={total:.3f}"
        )
        return total

    def batch(self, selfies_list: list[str], n_workers: int = 4) -> list[float]:
        """Compute rewards in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(self.__call__, selfies_list))

    @classmethod
    def from_config(cls, cfg: dict) -> "CombinedReward":
        reward_cfg = cfg["reward"]
        return cls(
            receptor_pdb=reward_cfg["docking"]["receptor_pdb"],
            box_config="docking/box_config.json",
            reference_smiles_path=reward_cfg["diversity"]["reference_smiles_path"],
            weights=reward_cfg["weights"],
            exhaustiveness=reward_cfg["docking"].get("exhaustiveness", 2),
        )
