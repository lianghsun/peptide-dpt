# peptide-dpt

PSMA-targeted peptidomimetic drug discovery via SELFIES-based LLM with GRPO.

Training pipeline: **Pretraining → PSMA Fine-tuning → GRPO (docking reward)**

## Target

**PDB 8BOW** — PSMA/GCPII (Glutamate Carboxypeptidase II) co-crystallized with PSMA-617 at 1.58 Å.

Key binding pocket: S1' glutamate recognition site (Arg210/Asn257/Lys699) + dual Zn²⁺ active center + entrance funnel.

## Molecular Representation

**SELFIES** (Self-Referencing Embedded Strings) — guarantees 100% valid molecule output at inference, eliminating validity reward hacking during GRPO.

## Pipeline

```
Phase 0 — Data
  ├── data/collect/collect_chembl.py     # ChEMBL GCPII (CHEMBL3231)
  ├── data/collect/collect_pdb.py        # PDB co-crystal ligands
  └── data/collect/collect_bindingdb.py  # BindingDB UniProt Q04609

  ├── data/process/smiles_to_selfies.py  # SMILES → SELFIES conversion
  ├── data/process/prepare_pretrain.py   # Build CLM pretraining dataset
  └── data/process/prepare_psma_sft.py  # Build activity-conditioned SFT dataset

Phase 1 — Pretraining (from scratch)
  └── training/pretrain.py               # Gemma-3-1b random init, CLM on SELFIES

Phase 2 — PSMA Fine-tuning
  └── training/finetune_psma.py          # SFT with potency conditioning tokens

Phase 3 — GRPO
  ├── training/grpo.py                   # TRL GRPOTrainer
  └── reward/combined.py                 # R = 0.6*dock + 0.25*SA + 0.15*diversity

Inference
  └── inference/generate.py              # Generate → decode → rank
```

## Reward Function

| Component | Weight | Description |
|---|---|---|
| `R_docking` | 0.60 | Gnina docking score vs 8BOW (normalized kcal/mol → [0,1]) |
| `R_sa` | 0.25 | RDKit SA score (synthesizability) |
| `R_diversity` | 0.15 | Tanimoto distance to known PSMA binders |

SELFIES eliminates the need for a validity reward term (R_validity).

## Datasets (HuggingFace)

| Dataset | Access | Content |
|---|---|---|
| [`lianghsun/peptidomimetic-pretrain`](https://huggingface.co/datasets/lianghsun/peptidomimetic-pretrain) | Public | General SELFIES corpus for pretraining |
| [`lianghsun/psma-sft`](https://huggingface.co/datasets/lianghsun/psma-sft) | Gated (request access) | PSMA-specific SFT with activity labels |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Install gnina: https://github.com/gnina/gnina/releases
```

## Run

```bash
# Step 0: Collect data
python -m data.collect.collect_chembl
python -m data.collect.collect_pdb
python -m data.collect.collect_bindingdb

# Step 0b: Process
python -m data.process.smiles_to_selfies --input data/raw/chembl_psma.csv --smiles_col canonical_smiles --output data/processed/chembl_psma_selfies.csv --build_vocab
python -m data.process.prepare_pretrain --selfies_csvs data/processed/*_selfies.csv --vocab tokenizer/selfies_vocab.json
python -m data.process.prepare_psma_sft --selfies_csvs data/processed/chembl_psma_selfies.csv data/processed/pdb_selfies.csv --vocab tokenizer/selfies_vocab.json

# Step 0c: Prepare receptor
python -m docking.prepare_receptor

# Step 0d: Upload to HuggingFace
python -m data.upload_hf --phase all

# Step 1: Pretrain
python -m training.pretrain --config configs/pretrain.yaml

# Step 2: Fine-tune
python -m training.finetune_psma --config configs/finetune.yaml

# Step 3: GRPO
python -m training.grpo --config configs/grpo.yaml

# Inference
python -m inference.generate --checkpoint checkpoints/grpo/best --n_samples 1000 --run_docking
```

## Key Files

| File | Description |
|---|---|
| `tokenizer/selfies_tokenizer.py` | SELFIES tokenizer with `[PAD]/[BOS]/[EOS]` + activity tokens |
| `reward/combined.py` | Combined reward orchestrator |
| `docking/run_gnina.py` | Gnina subprocess wrapper + score parser |
| `docking/prepare_receptor.py` | 8BOW receptor prep + box center computation |
