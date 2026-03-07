# Peptide Drug Design for PSMA — Project Plan

## 目標

針對攝護腺癌靶點 **PSMA (GCPII, Glutamate Carboxypeptidase II)**，從頭訓練一個 **gemma-3-1b-pt** 模型，生成具高親和力的 peptide 候選藥物。
訓練流程：**Pretraining → Fine-tuning → GRPO (docking-based reward)**

---

## 靶點分析：PDB 8BOW

| 屬性 | 詳情 |
|---|---|
| 蛋白質 | PSMA / GCPII (Homo sapiens) |
| 分辨率 | 1.58 Å (X-ray diffraction) |
| 生物組裝 | 同源二聚體 (C2 cyclic symmetry) |
| 共結晶配體 | **PSMA-617** (compound QYF) — 臨床用 benchmark ligand |
| 催化活性 | Zn²⁺ 依賴性金屬羧肽酶 (EC 3.4.17.21) |
| 糖基化 | 7 個 N-linked 糖基化位點 |
| 輔助離子 | 2× Zn²⁺, 1× Ca²⁺, Cl⁻ |

### 關鍵結合位點架構

PSMA 的結合腔分為三個連續區域：

1. **S1' 麩胺酸辨識口袋**
   - 關鍵殘基：Arg210、Asn257、Lys699
   - 與 Glu (glutamate) 端形成氫鍵
   - 所有已知高效抑制劑均需保留此交互作用

2. **雙核 Zn²⁺ 催化活性中心**
   - Urea carbonyl 與 Zn²⁺ 形成配位鍵
   - 藥效基團：Glu-urea-Lys (KuE motif) 為核心骨架

3. **入口漏斗 (Entrance Funnel)**
   - 空間寬敞，可容納多種連結基/功能基
   - PSMA-617 的 linker 折疊於此區域

### 設計含義

生成的 peptide 候選物應具備：
- 至少一個帶負電的 Glu/Asp 尾端（對應 S1' 口袋）
- 具尿素（urea）或類似鋅配位基團（若為 peptidomimetic）
- 多樣的中間序列以最佳化 entrance funnel 填充

---

## 訓練策略決策

### 關於 MOSES 小分子 → Peptide Transfer（選項一）

**結論：不建議採用此路徑。**

原因：
- MOSES 資料集來自 ZINC drug-like small molecules，化學語法（SMILES token 分佈、環系統、官能基）與 peptide 有根本性差異
- 小分子 SMILES 的 grammar 不利於 peptide 生成，transfer 只會引入雜訊而非知識
- 模型須同時拋棄「小分子語法」並學習「peptide 語法」，浪費預訓練容量
- 近期文獻（PeptideMTR, PeptideCLM, PepBERT）均直接從 peptide 資料預訓練，無需小分子橋接

### 關於 Peptide Dataset → PSMA Fine-tuning（選項二）

**結論：採用此路徑，為最直接且文獻支持的方案。**

分兩階段：
1. 在大規模 peptide 語料上做 domain pretraining（建立 peptide 化學語法）
2. 在 PSMA-specific 資料集上做 supervised fine-tuning（注入靶向偏好）

### 關於 Docking Score 作為 GRPO Reward（選項三）

**結論：採用，但需多訊號組合。**

純 docking score 作為 reward 容易導致 reward hacking（生成在計算上「得分高」但化學無效的序列）。
建議使用組合 reward（見 GRPO 章節）。

---

## 四階段訓練流程

### Phase 0：資料準備

#### 表示法選擇（已決策）

**FASTA 與「胺基酸序列」的關係**：FASTA 只是一種**檔案格式**（`>header\nSEQUENCE`），其內容本質上仍是胺基酸單字母序列，不影響模型的 token 表示法。所以「用 FASTA」和「用 AA 序列」是同一件事。

真正需要決定的是：**AA 序列 vs SMILES vs HELM**。

| 表示法 | 優點 | 缺點 | 適用場景 |
|---|---|---|---|
| AA 序列 | 最短、最自然、LM 最易學 | 無法表示非天然 AA、urea 基團、環化 | 純天然線性 peptide |
| SMILES | 可表示任何化學結構，包含 peptidomimetic | Token 序列長、**可生成非法字串** | 小分子 / peptidomimetic |
| SELFIES | **100% 保證合法分子**、可表示 peptidomimetic | 比 SMILES 長 ~20–30%、較冷門 | **本專案採用** |
| HELM | 緊湊且可表示環化/修飾 | 較冷門，資料量少 | 巨環 / 修飾 peptide |

**本專案決策：使用 SELFIES**

理由：
- PSMA 的所有高效已知配體（PSMA-617, DCFPyL, 2-PMPA, KuE）**全部是 peptidomimetic**，含 Glu-urea-Lys 骨架，無法用純 AA 序列表示
- 若使用 AA 序列，生成的候選物先天上就無法包含 urea 鋅配位基團，對 PSMA 親和力的上限很低
- SELFIES 相比 SMILES 保證 **100% 輸出合法分子**，模型容量不需浪費在學習「哪些 token 組合非法」
- Gemma-3-1b 從頭訓練容量有限，SELFIES 讓模型可以把全部參數用於學習化學結構偏好
- GRPO reward 可簡化：移除 `R_validity` 項，reward signal 更穩定、梯度更乾淨
- 所有 SMILES 語料可一行程式碼批次轉換為 SELFIES；docking 時 decode 回 SMILES 即可，pipeline 完全相容

**Tokenization 策略**：使用 SELFIES 原生的 `[token]` 格式（`sf.split_selfies()`），每個 `[...]` 為一個 token，而非 character-level。

**後果**：pretraining 語料以 SMILES 資料庫（ChEMBL, ZINC, PDB ligands）為原始來源，統一轉換為 SELFIES 後訓練。

```python
import selfies as sf

# 資料集轉換（一次性）
selfies_str = sf.encoder(smiles)

# Inference 後還原（docking 前）
smiles = sf.decoder(selfies_str)

# Tokenization
tokens = list(sf.split_selfies(selfies_str))
# → ['[C]', '[=O]', '[NH1]', '[C@@H]', ...]
```

---

#### PSMA 領域訓練資料全面 Survey

**重要認知：目前沒有公開的 PSMA peptide 訓練資料集 GitHub repo。** 所有資料需從以下來源人工彙整或程式化萃取。

##### A. 結構資料庫（PDB GCPII 共晶結構）

以下 PDB entry 均含 GCPII + 配體共晶結構，可萃取配體 SMILES + 結合親和力：

| PDB | 配體 | 類型 | 備註 |
|---|---|---|---|
| 8BOW | PSMA-617 (QYF) | Glu-urea | 1.58 Å，benchmark |
| 8BO8 | P17 | Glu-urea 衍生物 | IC50 0.30 nM |
| 8BOL | P18 | Glu-urea 衍生物 | IC50 0.45 nM |
| 1Z8L | glutamate | 天然受質 | 最早 PSMA 結構 |
| 2JBJ | 2-PMPA | phosphonate | IC50 0.9 nM |
| 4LQG | CTT1056 | phosphoramidate | IC50 ~nM |
| 4NGN | urea-based | urea | — |
| 4NGP | urea-based | urea | — |
| 6S1X | KB1160 | — | E424M mutant |
| 6H7Z | RNA 2-65-1 | — | — |
| 6HKZ | RNA 2-49-1 | — | — |
| 6HKJ | RNA 2-19-1 | — | — |

**做法**：用 `pypdb` 或 RCSB REST API 批次下載所有 GCPII（UniProt Q04609）共晶結構，萃取配體 SMILES。

##### B. ChEMBL（最大可程式化取得的資料庫）

- Target：**CHEMBL3231**（Glutamate carboxypeptidase II, GCPII）
- 預估有數百筆 IC50/Ki 活性資料
- 可直接用 `chembl_webresource_client` Python 套件批次下載
```python
from chembl_webresource_client.new_client import new_client
activities = new_client.activity.filter(target_chembl_id='CHEMBL3231')
```
- 包含 phosphonate、urea、thiol、hydroxamate 各類骨架

##### C. BindingDB

- UniProt Q04609 查詢：`https://www.bindingdb.org/rwd/bind/ByTarget.jsp`
- 資料格式：IC50/Ki/Kd，含 SMILES
- 可下載 TSV/SDF 格式，規模預估數十至數百筆

##### D. 文獻人工彙整（最重要的 PSMA 特異性資料）

以下論文含大量化合物表格，需手動或用 LLM 輔助萃取 SMILES + 活性：

| 論文 | 內容 | 化合物數 |
|---|---|---|
| *Metamorphosis of PSMA inhibitors* (PMC8921357, 2022) | 各類骨架綜述，引用 100+ 化合物 | ~100 |
| *Glu-Ureido Inhibitors of PSMA* (J. Nucl. Med. 2018, 58 S2:17S) | KuE 系列 SAR | ~30 |
| *ACS Omega 2024* (10.1021/acsomega.4c10142) | PSMA-617 衍生物 SAR | ~20 |
| *J. Med. Chem. 2021* urea SAR 系列 | urea linker 修飾 | ~40 |
| *EJNMMI Res.* 2018 (PMC6104465) | lipophilic linker urea | ~15 |
| *Prodrugs targeting PSMA* (ACS 2024, 10.1021/acs.jmedchem.4c02626) | prodrug 系列 | ~20 |
| *Theranostic PSMA ligands* (EJNMMI 2022) | multimodal 系列 | ~10 |
| *Paving the way for future PSMA inhibitors* (PMC12474757, 2025) | 最新結構修飾分析 | ~20 |

##### E. Phage Display 篩選得到的 Peptide Sequences（純 AA 序列，可作 pretraining 輔助）

這類資料是真正的「純 peptide」（非 peptidomimetic），可補充 AA 序列語料，但對 PSMA 的親和力較弱（μM 級別）：

| 論文 | 序列 | 親和力 |
|---|---|---|
| PLOS ONE 2013 (10.1371/journal.pone.0068339) | SHSFSVGSGDHSPFT, GRFLTGGTGRLLRIS | 未報告 Ki |
| J. Drug Delivery 2016 (27582001) | GTI peptide | Kd ~8 μM |
| *Disulfide-constrained heptamers* (watermark02) | 多個環狀序列 | μM 級別 |

**注意**：這些 phage display peptide 親和力弱（μM），而 PSMA-617 是 nM 級，差異 ~1000 倍。若目標是高親和力，phage display 序列只能作為多樣性語料，不能作為高品質訓練目標。

##### F. 彙整後預估訓練集規模

| 來源 | 分子數（估計） | 用途 |
|---|---|---|
| ChEMBL GCPII | 300–500 | PSMA SFT 核心 |
| BindingDB Q04609 | 100–300 | PSMA SFT |
| PDB 配體萃取 | 20–50 | 高品質 anchor |
| 文獻人工彙整 | 200–400 | PSMA SFT |
| Phage display AA 序列 | ~50 | 多樣性補充 |
| **PSMA 特異性合計** | **~1000** | PSMA SFT + GRPO 正樣本 |

---

#### 通用 Peptide/Peptidomimetic Pretraining 語料（SMILES 為主）

| 資料集 | 規模 | 來源 |
|---|---|---|
| ChEMBL peptide bioactives (MW 500–2000) | ~10萬 | ChEMBL 篩選 |
| PubChem peptide-like compounds | 數十萬 | PubChem |
| ZINC peptide subset | 數萬 | ZINC20 |
| PDB 所有配體 SMILES | ~35萬 | PDB CCD |
| BindingDB 所有配體 | ~100萬 | BindingDB |

Pretraining 目標：讓模型學習合法 SMILES 語法 + 含 Glu/urea/phosphonate 基團的 peptidomimetic 化學空間。

---

### Phase 1：Domain Pretraining（從頭訓練）

- **模型：** gemma-3-1b-pt (random init)
- **資料：** 大規模 peptide 序列語料（Phase 0 預訓練集）
- **目標：** Next-token prediction (CLM)
- **策略：**
  - 自定義 tokenizer 或複用 Gemma tokenizer（確認胺基酸字母覆蓋）
  - 加入 special tokens：`[BOS]`, `[EOS]`, `[SEP]`, `[PAD]`
  - 考慮加入 peptide 性質條件 token（長度、電荷、親疏水性）
- **驗證指標：** Perplexity、生成 peptide 的有效率、氨基酸組成分佈

---

### Phase 2：PSMA-Specific Supervised Fine-tuning

- **資料：** PSMA/GCPII 已知活性 peptide/peptidomimetic
- **格式：** 條件式生成（可加入 IC50 分段 token 作為 prompt 條件）
- **目標：** 讓模型偏向生成含 PSMA 藥效基團的序列
- **注意：** 此階段資料量小，需 low LR + early stopping 防止遺忘

---

### Phase 3：GRPO（Docking Score Reward）

**環境設置：**

- **結構準備：** 使用 8BOW，移除 PSMA-617 後準備 receptor PDB
- **結合口袋定義：** 以 PSMA-617 質心為中心，box size ~22×22×22 Å（涵蓋 S1' + Zn + entrance funnel）
- **Docking 工具：** AutoDock Vina 或 **Gnina**（推薦 Gnina：支援 CNN scoring，更適合 peptide）
- **序列→結構：** 需在 reward 計算中生成 3D 構型（OpenBabel / RDKit / ESMFold for peptide）

**組合 Reward 函數：**

```
# SELFIES 保證 100% 合法分子 → 移除 R_validity
R_total = w1 * R_docking + w2 * R_sa + w3 * R_diversity

其中：
- R_docking   = Gnina 結合自由能分數（取負值，使高親和力得高分）
- R_sa        = Synthetic Accessibility Score（RDKit 計算，1-10 映射取逆）
- R_diversity = 與已知 PSMA binder 的 Tanimoto 距離（避免 reward hacking / 模式崩潰）

建議初始權重：w1=0.6, w2=0.25, w3=0.15
```

> 相比 SMILES 方案，去掉了 `R_validity`（SELFIES 天生保證）使 reward signal 更穩定，梯度不再有稀疏問題。

**GRPO 實作細節：**
- Group size G：通常 4~8，依計算資源調整
- KL 懲罰係數 β：控制與 reference policy 偏離程度
- Clip ratio ε：0.2（標準 PPO clipping）
- 每輪 docking 計算為主要瓶頸，需 batch 平行化

---

## 技術疑慮與建議

### Peptide vs Small Molecule 生成的根本差異

| 維度 | Small Molecule (MOSES) | Peptide |
|---|---|---|
| 表示法 | SMILES（環、分支複雜） | AA 序列（線性/環） |
| 化學空間 | Lipinski-like, MW < 500 | MW 500–5000 |
| 合成方式 | 有機合成 | 固相合成 (SPPS) |
| 藥物動力學 | 口服生體可用率佳 | 通常需注射，易降解 |
| Docking | 標準配體對接 | 需 peptide-protein docking |

**重要：** PSMA 的已知藥物（PSMA-617, DCFPyL）是「peptidomimetic」而非純 peptide——核心為 Glu-urea 藥效基團接上小分子 linker。若要生成真正的 peptide，需確認模型的序列空間與 docking pipeline 相容。

### Docking Score Reward 的潛在問題

1. **計算成本高：** 每個生成序列都需要 3D 建模 + docking，建議用輕量化代理模型（surrogate model）加速
2. **Reward hacking：** 模型可能學會生成計算上得分高但無生物活性的結構，需加入 diversity reward
3. **Peptide docking 精度：** Vina 對 peptide 效果較差，Gnina（CNN-based）或 FlexPepDock（Rosetta）較適合

### 後處理與驗證建議

生成的候選 peptide 應通過：
1. 合法性過濾（標準 AA 組成、無衝突修飾）
2. PSMA docking 分數篩選（top 1%）
3. ADMET 預測（SwissADME 或 pkCSM）
4. 結構多樣性分析（避免 cluster 過於集中）
5. 分子動力學模擬驗證（top candidate）

---

## 工具鏈建議

| 工具 | 用途 |
|---|---|
| `transformers` + `trl` | Gemma 訓練、GRPO 實作 |
| `AutoDock Gnina` | Docking reward computation |
| `selfies` | SELFIES ↔ SMILES 轉換、tokenization |
| `OpenBabel` / `RDKit` | SMILES 處理、3D 生成（decode 後使用）|
| `ESMFold` / `PeptideBuilder` | Peptide 3D 構型預測 |
| `PyMOL` / `UCSF ChimeraX` | 視覺化 binding pose |
| `DeepChem` | ADMET 預測 |
| `wandb` | 訓練監控 |

---

## 里程碑

1. [ ] 資料管線建立（peptide 語料清洗、tokenizer 設計）
2. [ ] Phase 1 pretraining 完成，perplexity 收斂
3. [ ] Phase 2 PSMA fine-tuning，驗證靶向序列生成率
4. [ ] 8BOW docking pipeline 建立並驗證（PSMA-617 作為 benchmark，應重現文獻 Ki）
5. [ ] Phase 3 GRPO 訓練，reward 曲線穩定上升
6. [ ] Top-100 候選物後處理分析
7. [ ] 實驗合作夥伴驗證（可選）

---

## 參考文獻

- 8BOW 結構及 PSMA-617 SAR：*ACS Omega* 2024, DOI: 10.1021/acsomega.4c10142
- KuE 藥效基團設計：*J. Nuclear Medicine* 58(S2):17S
- PSMA inhibitor 綜述：*Metamorphosis of PSMA inhibitors*, PMC8921357, 2022
- 最新結構修飾：*Paving the way for future PSMA inhibitors*, PMC12474757, 2025
- Phage display peptide (PLOS ONE 2013)：10.1371/journal.pone.0068339
- SELFIES：Krenn et al., *iScience* 2020, DOI: 10.1016/j.isci.2020.101299
- PeptideMTR（SMILES-based peptide LM，可轉 SELFIES）：bioRxiv 2026.01.06
- HELM-GPT (macrocyclic peptide generation)：*Bioinformatics* 2024, btae364
- MolOrgGPT (GRPO for molecular generation)：*J. Chem. Inf. Model.* 2025
- MOSES benchmark：*Front. Pharmacol.* 2020, DOI: 10.3389/fphar.2020.565644
- Gnina docking：McNutt et al., *J. Cheminformatics* 2021
