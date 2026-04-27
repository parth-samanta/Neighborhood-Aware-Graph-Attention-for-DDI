# Neighborhood-Aware Graph Attention for Drug-Drug Interaction Prediction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)

A multimodal drug–drug interaction (DDI) prediction framework that fuses **FastText semantic embeddings** (trained on PubMed abstracts) with **TransE structural embeddings** (trained on the UMLS knowledge graph) via a learned per-dimension gated fusion mechanism, further contextualised by a **relation-aware multi-head neighbourhood attention** module over local KG context.

> **Test AUROC: 0.9803 · AUPRC: 0.9647** on 2,644,175 DrugBank-sourced drug pairs.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Inductive Generalisation](#inductive-generalisation)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Overview

Drug–drug interactions (DDIs) are a leading cause of preventable hospital admissions, especially in polypharmacy settings where patients take five or more medications simultaneously. Exhaustive experimental profiling of all drug pairs is infeasible, motivating scalable computational prediction.

This framework is grounded entirely in the **UMLS Metathesaurus (2023 release)**, providing a single consistent ontological foundation for:
- Knowledge graph construction (385,980 nodes, 921,126 directed edges, 16 relation types)
- Structural embedding training (TransE via PyKEEN)
- Semantic embedding training (FastText skip-gram on PubMed abstracts)

Interaction labels come from **DrugBank**, mapped into the shared **RxNorm** identifier space.

---

## Architecture

```
Drug A ──┐                              ┌── Drug B
         │                              │
    [FastText embedding (128d)]    [FastText embedding (128d)]
    [TransE embedding   (128d)]    [TransE embedding   (128d)]
         │                              │
    ┌────▼────────────────────┐    ┌────▼────────────────────┐
    │  Gated Fusion           │    │  Gated Fusion           │
    │  g = σ(Wg [w‖s])       │    │  g = σ(Wg [w‖s])       │
    │  f = g⊙w + (1-g)⊙s    │    │  f = g⊙w + (1-g)⊙s    │
    └────────────┬────────────┘    └────────────┬────────────┘
                 │                              │
    ┌────────────▼────────────┐    ┌────────────▼────────────┐
    │ Relation-Aware          │    │ Relation-Aware          │
    │ Multi-Head Attn (H=4)  │    │ Multi-Head Attn (H=4)  │
    │ 15 KG neighbours        │    │ 15 KG neighbours        │
    │ + TransE relation embs  │    │ + TransE relation embs  │
    └────────────┬────────────┘    └────────────┬────────────┘
                 │   hA (128d)    hB (128d)  │
                 └──────────────┬────────────┘
                                │
                   z = [hA‖hB‖hA⊙hB‖|hA-hB|]  (512d)
                                │
                   ┌────────────▼────────────┐
                   │  MLP: 512→256→64→1      │
                   │  ReLU + BN + Dropout    │
                   └────────────┬────────────┘
                                │
                         Interaction logit
```

### Key Components

**Gated Fusion** — learns per-dimension soft weights to dynamically balance semantic (PubMed co-occurrence) and structural (KG relational position) information.

**Relation-Aware Neighbourhood Attention** — contextualises each drug's fused vector using its 15 nearest UMLS neighbours. Attention keys are additively modulated by pre-trained TransE relation embeddings, conditioning scores on edge type rather than node identity alone.

**Four-Part Interaction Vector** — concatenates `[hA, hB, hA⊙hB, |hA−hB|]`, making both element-wise similarity and complementarity explicit to the final MLP.

---

## Repository Structure

```
.
├── data/                    # Processed datasets and embeddings
│   ├── kg/                  # UMLS KG node/edge tables
│   ├── embeddings/          # FastText and TransE embedding files
│   └── pairs/               # Train/val/test DDI pair splits
├── models/                  # Saved model checkpoints
├── results/                 # Evaluation outputs and plots
├── src/
│   ├── kg_construction.py   # UMLS KG builder (MRSTY, MRCONSO, MRREL)
│   ├── fasttext_train.py    # PubMed retrieval + FastText training
│   ├── transe_train.py      # TransE training via PyKEEN
│   ├── dataset.py           # DDI pair curation and negative sampling
│   ├── neighbourhood.py     # Pre-computation of 15-neighbour matrices
│   ├── model.py             # GatedFusion + RelationAttention + MLP
│   ├── train.py             # Training loop with OneCycleLR
│   ├── evaluate.py          # AUROC, AUPRC, F1, accuracy
│   └── inductive_eval.py    # Drug-level holdout evaluation
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/parth-samanta/Neighborhood-Aware-Graph-Attention-for-DDI.git
cd Neighborhood-Aware-Graph-Attention-for-DDI
pip install -r requirements.txt
```

> **GPU:** Training was conducted on an NVIDIA RTX 5060 Ti. Mixed-precision (float16 forward / float32 gradients) is enabled automatically on CUDA devices.

### PyTorch Geometric

Install the PyG wheels matching your CUDA version from the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

---

## Data Preparation

### 1. UMLS Access

A UMLS licence is required. Download the 2023 Metathesaurus release from [https://www.nlm.nih.gov/research/umls/](https://www.nlm.nih.gov/research/umls/) and place the following files in `data/umls/`:

- `MRSTY.RRF` — semantic type assignments
- `MRCONSO.RRF` — concept names and sources
- `MRREL.RRF` — directed typed relationships

### 2. Knowledge Graph Construction

```bash
python src/kg_construction.py \
    --umls_dir data/umls/ \
    --out_dir data/kg/
```

Produces a NetworkX `MultiDiGraph` with 385,980 nodes and 921,126 edges across 16 relation types and 6 node categories (drug, disease, protein, gene, molecular function, biological process).

### 3. Semantic Embeddings (FastText)

```bash
python src/fasttext_train.py \
    --kg_nodes data/kg/nodes.csv \
    --out_dir data/embeddings/ \
    --dim 128 \
    --window 7 \
    --epochs 5 \
    --workers 20
```

Retrieves PubMed abstracts via NCBI Entrez API (with exponential backoff), anchors drug entities using FlashText, and trains a 128-dimensional FastText skip-gram model.

### 4. Structural Embeddings (TransE)

```bash
python src/transe_train.py \
    --kg_dir data/kg/ \
    --out_dir data/embeddings/ \
    --dim 128 \
    --lr 5e-3 \
    --epochs 250 \
    --batch_size 8192
```

Trains TransE via PyKEEN on an 80/10/10 KG split. Outputs 128-dimensional entity and relation embedding files.

### 5. DDI Pair Curation

```bash
python src/dataset.py \
    --drugbank_interactions data/raw/drugbank_interactions.csv \
    --drugbank_rxnorm_crosswalk data/raw/drugbank2rxnorm.csv \
    --kg_nodes data/kg/nodes.csv \
    --fasttext_model data/embeddings/fasttext.bin \
    --out_dir data/pairs/ \
    --neg_ratio 2 \
    --seed 42
```

Yields 881,725 positive pairs and 1,763,450 negative pairs (1:2 ratio), split 80/10/10 with stratification.

---

## Usage

### Training

```bash
python src/train.py \
    --pairs_dir data/pairs/ \
    --embeddings_dir data/embeddings/ \
    --kg_dir data/kg/ \
    --out_dir models/ \
    --variant full \
    --batch_size 2048 \
    --epochs 30 \
    --lr 3e-4 \
    --seed 42
```

Available `--variant` options: `full`, `semantic_only`, `structural_only`, `no_neighborhood`, `no_relations`, `flat_attn`.

### Evaluation

```bash
python src/evaluate.py \
    --model_checkpoint models/best_full.pt \
    --pairs_dir data/pairs/ \
    --embeddings_dir data/embeddings/ \
    --kg_dir data/kg/
```

### Inductive Evaluation

```bash
python src/inductive_eval.py \
    --pairs_dir data/pairs/ \
    --embeddings_dir data/embeddings/ \
    --kg_dir data/kg/ \
    --holdout_fractions 0.1 0.2 0.3 0.4 0.5 \
    --variants full no_relations structural_only flat_attn semantic_only no_neighborhood \
    --out_dir results/inductive/
```

---

## Results

### Transductive Test Set Performance

| Model Variant      | AUROC      | AUPRC      | F1         | Accuracy   |
|--------------------|------------|------------|------------|------------|
| **Full (proposed)**| **0.9803** | **0.9647** | **0.8866** | **0.9198** |
| NoRelations        | 0.9786     | 0.9617     | 0.8816     | 0.9160     |
| StructuralOnly     | 0.9680     | 0.9420     | 0.8547     | 0.8956     |
| FlatAttn           | 0.9584     | 0.9238     | 0.8368     | 0.8819     |
| SemanticOnly       | 0.9430     | 0.8955     | 0.8080     | 0.8586     |
| NoNeighborhood     | 0.9474     | 0.9050     | 0.8141     | 0.8634     |

Trained on 2,644,175 drug pairs (881,725 positive, 1,763,450 negative) with a 1:2 class ratio.

---

## Ablation Study

Six architectural variants were evaluated under identical hyperparameters and data splits:

- **Neighbourhood context is the dominant driver** — removing it (NoNeighborhood) causes the largest single drop (−0.033 AUROC), exceeding the contribution of any other component.
- **Learned attention outperforms uniform pooling** — FlatAttn (uniform mean-pool) drops 0.022 AUROC vs. the full model, confirming that not all KG neighbours are equally informative.
- **Structural > Semantic alone** — TransE structural embeddings (StructuralOnly: 0.968) outperform FastText semantic embeddings (SemanticOnly: 0.943) in isolation, but neither matches the gated fusion of both.
- **Relation-type conditioning is helpful but marginal** — NoRelations differs from Full by only 0.0017 AUROC, suggesting TransE node positions already implicitly encode much of the relational context.

---

## Inductive Generalisation

The model is evaluated under a **drug-level holdout** protocol withholding 10%–50% of drugs entirely from training, testing two scenarios:

- **Inductive**: both drugs in the test pair are unseen (hardest case).
- **Semi-inductive**: one drug is unseen, one was seen during training (realistic new-drug screening scenario).

| Setting        | 10% holdout | 50% holdout |
|----------------|-------------|-------------|
| Inductive      | 0.722       | 0.697       |
| Semi-inductive | 0.855       | 0.839       |

Performance degrades gracefully across a fivefold increase in the unseen drug fraction, demonstrating that pre-trained embeddings and KG neighbourhood context — not memorised pair-level statistics — drive generalisation.

---

## Training Hyperparameters

| Hyperparameter       | Value                           |
|----------------------|---------------------------------|
| Optimiser            | Adam (β₁=0.9, β₂=0.999)        |
| Peak learning rate   | 3×10⁻⁴                         |
| LR schedule          | OneCycleLR (cosine, 10% warmup) |
| Batch size           | 2,048                           |
| Training epochs      | 30                              |
| Gradient clip norm   | 1.0                             |
| BCE positive weight  | 2.0 (accounts for 1:2 imbalance)|
| Attention heads      | 4                               |
| Attention dropout    | 0.10                            |
| MLP hidden dims      | [512, 256, 64]                  |
| MLP dropout          | 0.30                            |

---

## Limitations

- **Drug universe coverage**: Only 2,994 of 254,261 KG drug nodes are training-eligible (those with both an RxNorm identifier and DrugBank interaction labels, ~1.2%). Expanding to SIDER, OFFSIDES, or FAERS could broaden coverage significantly.
- **Frozen embeddings**: FastText and TransE embeddings are pre-trained and frozen. End-to-end fine-tuning or low-rank adaptation may further improve performance.
- **Neighbourhood size**: A fixed 15-neighbour window was chosen based on the median degree of drug nodes. Larger or relation-stratified neighbourhoods could amplify the benefit of typed-edge conditioning.

---

## Citation

If you use this code or the results in your work, please cite:

```bibtex
@misc{samanta2026neighbourhood,
  title        = {Neighbourhood-Aware Graph Attention for Drug-Drug Interaction Prediction},
  author       = {Samanta, Parthsarathi and Prabhu, Sidhanth and Modi, Shrey and Hegde, Sindhoor},
  year         = {2026},
  howpublished = {Course Project, AID 829 / Natural Language Processing, International Institute of Information Technology Bangalore (IIIT-B)},
  url          = {https://github.com/parth-samanta/Neighborhood-Aware-Graph-Attention-for-DDI}
}
```

---

## Authors

| Name | GitHub |
|------|--------|
| **Parthsarathi Samanta** | [@parth-samanta](https://github.com/parth-samanta) |
| **Sidhanth Prabhu** | [@SidhanthPrabhu](https://github.com/SidhanthPrabhu) |
| **Shrey Modi** | [@Shrey0604](https://github.com/Shrey0604) |
| **Sindhoor Hegde** | [@AgentOfFedora](https://github.com/AgentOfFedora) |
