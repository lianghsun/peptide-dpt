# Peptide-DPT 訓練狀態 (更新: 2026-03-11)

## 三階段進度

| 階段 | 狀態 | 備註 |
|---|---|---|
| Phase 1: Pretrain | ✅ 完成 | loss 7.48 → 2.51，checkpoint: `checkpoints/pretrain/best` |
| Phase 2: SFT | ✅ 完成（第二次，正確資料）| loss 0.55 → 0.36，early stop @ epoch 5.5，checkpoint: `checkpoints/finetune/best` |
| Phase 3: GRPO | 🔄 進行中 | NCCL timeout crash 後剛修正，等待重新啟動 |

---

## 目前要執行的指令（GRPO）

```bash
cd /data/tmp/peptide-dpt
git pull  # 確保拿到最新修正
torchrun --nproc_per_node=8 -m training.grpo --config configs/grpo_b200.yaml 2>&1 | tee logs/grpo_b200.log
```

---

## 已修正的關鍵 Bug（歷史紀錄）

### SFT 資料路徑錯誤（已修正）
- **問題**：`finetune_b200.yaml` 指向 `psma-sft_train.jsonl`（舊檔，錯誤 vocab）
- **正確檔案**：`data/processed/psma_sft_train.jsonl`（正確 vocab，手動上傳）
- **現狀**：config 已修正，第二次 SFT 使用正確資料

### Vocab ID 錯位（已修正）
- **問題**：JSONL 用舊 corpus-based vocab（[C]=32）編碼，但當前 vocab 是 alphabet-based（[C]=159）
- **修正**：重新執行 `prepare_psma_sft.py`，重新生成 JSONL
- **Vocab 大小**：316（pretrain）→ 320（SFT 後加入 4 個 potency tokens）
  - `[VERY_POTENT]`=316, `[POTENT]`=317, `[MODERATE]`=318, `[WEAK]`=319

### NCCL Timeout（剛修正）
- **問題**：GRPO 每步 dock 128 分子/rank，gnina 慢，30 分鐘 NCCL timeout
- **修正**：`per_device_train_batch_size` 8→2，`group_size` 16→8，`ddp_timeout=7200`（2h）
- **現在每步**：2 × 8 = 16 分子/rank，約 2–5 分鐘

### Gnina Score Parser Bug（已修正）
- **問題**：進度條 `0%  10  20...` 被誤判為分數 10.0
- **修正**：等待 `-----+` 分隔線後才讀分數

### resume_from_checkpoint 錯誤（已修正）
- **問題**：`resume_from_checkpoint=True` 但目錄不存在時 crash
- **修正**：先偵測 `checkpoint-*` 是否存在再決定是否 resume

---

## 重要檔案路徑（B200）

```
/data/tmp/peptide-dpt/
├── checkpoints/
│   ├── pretrain/best/        ← Phase 1 完成
│   └── finetune/best/        ← Phase 2 完成（正確版）
├── configs/
│   ├── grpo_b200.yaml        ← 目前使用的 GRPO config
│   └── finetune_b200.yaml
├── data/processed/
│   ├── psma_sft_train.jsonl  ← 正確 SFT 訓練資料（手動上傳）
│   ├── psma_sft_val.jsonl    ← 正確 SFT 驗證資料（手動上傳）
│   ├── psma-sft_train.jsonl  ← 舊檔（錯誤 vocab，勿使用）
│   └── psma_known_smiles.txt ← GRPO diversity reward 用
├── tokenizer/
│   └── selfies_vocab.json    ← 320 tokens（含 potency tokens）
└── docking/
    ├── 8BOW_receptor.pdb
    └── box_config.json       ← gnina docking box 定義
```

---

## GRPO Config 現況（grpo_b200.yaml）

```yaml
per_device_train_batch_size: 2
group_size: 8
gradient_accumulation_steps: 4   # effective = 2*8*4 = 64 rollouts/step
exhaustiveness: 2                 # gnina 快速模式
max_new_tokens: 200
ddp_timeout: 7200                 # 2 小時（避免 NCCL timeout）
```

---

## SFT 資料品質說明

SFT 資料（ChEMBL + BindingDB GCPII）中 [VERY_POTENT] 樣本（261 筆）包含部分非典型 PSMA 化合物（無 Glu-urea-Lys 藥效基團）。這不影響 GRPO——Gnina docking reward 會直接優化 PSMA 結合，修正 SFT 偏差。

| Potency bin | 數量 |
|---|---|
| very_potent | 261 |
| potent | 579 |
| moderate | 991 |
| weak | 1978 |
| none | 3932 |
| **Total** | **7741** |

---

## Reward 設計

```
R_total = 0.60 * R_docking + 0.25 * R_sa + 0.15 * R_diversity
```

- `R_docking`：Gnina kcal/mol → reward（見 `reward/docking.py`）
- `R_sa`：RDKit SA score（1–10）→ reward
- `R_diversity`：Morgan fingerprint Tanimoto vs. 已知 PSMA SMILES

---

## 訓練指令參考

```bash
# Phase 1 Pretrain（已完成）
torchrun --nproc_per_node=8 -m training.pretrain --config configs/pretrain_b200.yaml

# Phase 2 SFT（已完成）
torchrun --nproc_per_node=8 -m training.finetune_psma --config configs/finetune_b200.yaml

# Phase 3 GRPO（進行中）
torchrun --nproc_per_node=8 -m training.grpo --config configs/grpo_b200.yaml 2>&1 | tee logs/grpo_b200.log
```
