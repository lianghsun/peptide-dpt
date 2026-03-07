"""Inference: generate PSMA-targeting peptidomimetic candidates.

Loads the GRPO-trained model and generates SELFIES strings,
then decodes to SMILES and ranks by docking score.

Usage:
    python -m inference.generate \
        --checkpoint checkpoints/grpo/best \
        --vocab tokenizer/selfies_vocab.json \
        --n_samples 1000 \
        --potency very_potent \
        --output outputs/candidates.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import selfies as sf
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from tokenizer.selfies_tokenizer import SelfiesTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

POTENCY_TOKEN_MAP = {
    "very_potent": "[VERY_POTENT]",
    "potent": "[POTENT]",
    "moderate": "[MODERATE]",
    "weak": "[WEAK]",
}


def build_prompt(
    tokenizer: SelfiesTokenizer, potency: str = "very_potent"
) -> torch.Tensor:
    potency_tok = POTENCY_TOKEN_MAP.get(potency, "[VERY_POTENT]")
    potency_id = tokenizer.token2id.get(potency_tok, tokenizer.bos_token_id)
    return torch.tensor([[tokenizer.bos_token_id, potency_id]], dtype=torch.long)


def generate_selfies(
    model: AutoModelForCausalLM,
    tokenizer: SelfiesTokenizer,
    n_samples: int,
    potency: str = "very_potent",
    batch_size: int = 32,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    device: str = "cpu",
) -> list[str]:
    model.eval()
    model.to(device)

    prompt = build_prompt(tokenizer, potency).to(device)
    all_selfies = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(n_batches), desc="Generating"):
        bs = min(batch_size, n_samples - len(all_selfies))
        prompt_batch = prompt.expand(bs, -1)

        with torch.no_grad():
            output_ids = model.generate(
                prompt_batch,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for seq in output_ids:
            # Strip prompt
            gen_ids = seq[prompt.size(1):].tolist()
            # Stop at EOS
            if tokenizer.eos_token_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(tokenizer.eos_token_id)]
            selfies_str = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if selfies_str:
                all_selfies.append(selfies_str)

        if len(all_selfies) >= n_samples:
            break

    return all_selfies[:n_samples]


def selfies_to_smiles(selfies_str: str) -> str | None:
    try:
        smi = sf.decoder(selfies_str)
        if smi and Chem.MolFromSmiles(smi):
            return smi
    except Exception:
        pass
    return None


def score_and_filter(
    selfies_list: list[str],
    receptor_pdb: str | None = None,
    box_config: str | None = None,
    reference_smiles_path: str | None = None,
) -> pd.DataFrame:
    from reward.sa_score import sa_score
    from reward.combined import CombinedReward

    records = []
    for s in tqdm(selfies_list, desc="Scoring"):
        smiles = selfies_to_smiles(s)
        if not smiles:
            continue

        sa = sa_score(smiles)
        record = {
            "selfies": s,
            "smiles": smiles,
            "sa_score": sa,
            "docking_score": None,
            "combined_reward": None,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Optionally run docking (slow)
    if receptor_pdb and box_config and reference_smiles_path:
        log.info("Running docking on all candidates (this may take a while)...")
        reward_model = CombinedReward(
            receptor_pdb=receptor_pdb,
            box_config=box_config,
            reference_smiles_path=reference_smiles_path,
        )
        from reward.docking import DockingReward
        docker = DockingReward(receptor=receptor_pdb, box_config=box_config)
        df["docking_score"] = df["selfies"].apply(docker)
        df["combined_reward"] = df["selfies"].apply(reward_model)
        df = df.sort_values("combined_reward", ascending=False)
    else:
        # Sort by SA score as proxy
        df = df.sort_values("sa_score", ascending=True)

    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/grpo/best")
    parser.add_argument("--vocab", default="tokenizer/selfies_vocab.json")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--potency", default="very_potent",
                        choices=list(POTENCY_TOKEN_MAP.keys()))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/candidates.csv")
    parser.add_argument("--run_docking", action="store_true")
    parser.add_argument("--receptor_pdb", default="docking/8BOW_receptor.pdb")
    parser.add_argument("--reference_smiles", default="data/processed/psma_known_smiles.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = SelfiesTokenizer.load(args.vocab)
    log.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    log.info(f"Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    if model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)

    log.info(f"Generating {args.n_samples} samples (potency={args.potency})...")
    selfies_list = generate_selfies(
        model, tokenizer,
        n_samples=args.n_samples,
        potency=args.potency,
        batch_size=args.batch_size,
        temperature=args.temperature,
        device=args.device,
    )
    log.info(f"Generated {len(selfies_list)} SELFIES")

    df = score_and_filter(
        selfies_list,
        receptor_pdb=args.receptor_pdb if args.run_docking else None,
        box_config="docking/box_config.json" if args.run_docking else None,
        reference_smiles_path=args.reference_smiles if args.run_docking else None,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info(f"Saved {len(df)} candidates → {args.output}")
    log.info(df[["smiles", "sa_score"]].head(10).to_string())


if __name__ == "__main__":
    main()
