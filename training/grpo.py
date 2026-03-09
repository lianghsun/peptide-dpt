"""Phase 3: GRPO training with docking-based reward.

Uses TRL's GRPOTrainer. The policy model generates SELFIES strings,
which are evaluated by the combined reward function.

Usage:
    python -m training.grpo --config configs/grpo.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from tokenizer.selfies_tokenizer import SelfiesTokenizer
from reward.combined import CombinedReward

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Prompt tokens used to elicit PSMA-targeted generation
SYSTEM_PROMPT = "[BOS][VERY_POTENT]"


def build_prompt_dataset(tokenizer: SelfiesTokenizer, n_prompts: int = 512) -> Dataset:
    """Build a dataset of conditioning prompts for GRPO rollouts.

    Each example is a prompt that seeds generation.
    We use the potency-conditioned BOS token as the only prompt.
    """
    potency_id = tokenizer.token2id.get("[VERY_POTENT]", tokenizer.bos_token_id)
    prompt_ids = [tokenizer.bos_token_id, potency_id]

    records = [
        {
            "input_ids": prompt_ids,
            "prompt": SYSTEM_PROMPT,
        }
        for _ in range(n_prompts)
    ]
    return Dataset.from_list(records)


def decode_generated_ids(
    generated_ids: list[int], tokenizer: SelfiesTokenizer
) -> str:
    """Decode generated token IDs to SELFIES string."""
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def make_reward_fn(reward_model: CombinedReward, tokenizer: SelfiesTokenizer):
    """Return a reward function compatible with TRL GRPOTrainer."""

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        # completions are raw decoded strings from the model
        rewards = []
        for completion in completions:
            # The completion string is the raw token sequence; treat as SELFIES
            r = reward_model(completion)
            rewards.append(r)
        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tokenizer = SelfiesTokenizer.load(cfg["tokenizer"]["path"])
    log.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Build reward model
    log.info("Initializing reward model (docking + SA + diversity)...")
    reward_model = CombinedReward.from_config(cfg)

    # Build prompt dataset
    prompt_dataset = build_prompt_dataset(tokenizer, n_prompts=512)

    # GRPO config
    g = cfg["grpo"]
    grpo_config = GRPOConfig(
        output_dir=g["output_dir"],
        num_train_epochs=g["num_train_epochs"],
        per_device_train_batch_size=g["per_device_train_batch_size"],
        gradient_accumulation_steps=g["gradient_accumulation_steps"],
        learning_rate=g["learning_rate"],
        lr_scheduler_type=g["lr_scheduler_type"],
        warmup_steps=g["warmup_steps"],
        bf16=g.get("bf16", True),
        logging_steps=g["logging_steps"],
        save_steps=g["save_steps"],
        report_to=g.get("report_to", "none"),
        run_name=g.get("run_name", "grpo"),
        # GRPO-specific
        num_generations=g["group_size"],           # G
        beta=g["kl_coef"],                         # KL penalty coefficient
        epsilon=g["clip_range"],                   # PPO clip range
        max_completion_length=g["max_new_tokens"],
        temperature=g.get("temperature", 1.0),
    )

    # Load fine-tuned model as policy
    from transformers import AutoModelForCausalLM
    checkpoint = cfg["model"]["checkpoint"]
    log.info(f"Loading policy model from {checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    if model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(tokenizer.vocab_size)

    # Reward function wrapper
    reward_fn = make_reward_fn(reward_model, tokenizer)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=prompt_dataset,
        processing_class=tokenizer,  # needed for TRL to decode completions before reward
    )

    log.info("Starting GRPO training...")
    trainer.train()
    trainer.save_model(f"{g['output_dir']}/best")
    log.info(f"GRPO model saved → {g['output_dir']}/best")


if __name__ == "__main__":
    main()
