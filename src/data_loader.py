"""
data_loader.py
--------------
Loads all CSV files into memory-efficient tensors and index mappings.

Files:
  - master_drug_embeddings.csv          : rxnorm -> CUI, Word2Vec (graph_*) [128-dim]
  - transe_neighbors_dict.csv           : CUI -> TransE node embedding (dim_*) [128-dim]
  - transe_relation_embeddings.csv      : relation_type -> TransE relation embedding
                                          (rel_dim_*) [128-dim], 16 unique relations
  - relation_aware_neighborhood_15.csv  : CUI -> 15 x (neighbor_CUI, relation_type) pairs
  - ddi_training_dataset_TRANSE_ready.csv : (drug_a_rxnorm, drug_b_rxnorm, label)
"""

import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

# Constants

EMB_DIM = 128
NUM_NEIGHBORS = 15
W2V_PREFIX   = "graph_"    # Word2Vec columns in master_drug_embeddings
TRANSE_PREFIX = "dim_"     # TransE node columns in transe_neighbors_dict
REL_PREFIX    = "rel_dim_" # TransE relation columns in transe_relation_embeddings

# Placeholder zero vector index (used when a neighbor CUI or relation is missing)
PAD_IDX = 0


# Low-level CSV helpers

def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float_row(row: dict, prefix: str, n_dims: int) -> np.ndarray:
    return np.array([float(row[f"{prefix}{i}"]) for i in range(1, n_dims + 1)],
                    dtype=np.float32)



# EmbeddingStore  –  central lookup structure


class EmbeddingStore:
    """
    Holds all embeddings as contiguous float32 tensors.

    drug_w2v_matrix      [N_drugs, 128]   – Word2Vec per drug
    drug_transe_matrix   [N_drugs, 128]   – TransE per drug (own node embedding)
    node_transe_matrix   [N_nodes+1, 128] – TransE for ALL KG nodes; row 0 = PAD zero
    rel_transe_matrix    [N_rels+1,  128] – TransE for each relation type; row 0 = PAD zero
    neighbor_matrix      [N_drugs,  15]   – node indices into node_transe_matrix
    relation_matrix      [N_drugs,  15]   – relation indices into rel_transe_matrix
    """

    def __init__(
        self,
        master_emb_path: str,
        transe_neighbors_path: str,
        neighborhood_path: str,
        transe_relation_path: str,
    ):
        # 1. Load TransE node embeddings (drugs + neighbor-only nodes)
        print("  Loading TransE node embeddings …")
        tr_rows = _read_csv(transe_neighbors_path)

        # row 0 = PAD zero vector
        self.cui_to_node_idx: dict[str, int] = {}
        node_emb_list: list[np.ndarray] = [np.zeros(EMB_DIM, dtype=np.float32)]

        for row in tr_rows:
            cui = row["CUI"]
            vec = _float_row(row, TRANSE_PREFIX, EMB_DIM)
            self.cui_to_node_idx[cui] = len(node_emb_list)
            node_emb_list.append(vec)

        self.node_transe_matrix = torch.tensor(
            np.stack(node_emb_list, axis=0), dtype=torch.float32
        )  # [N_nodes+1, 128]

        # 2. Load TransE relation embeddings
        print("  Loading TransE relation embeddings …")
        rel_rows = _read_csv(transe_relation_path)

        # row 0 = PAD zero vector
        self.relation_to_idx: dict[str, int] = {}
        rel_emb_list: list[np.ndarray] = [np.zeros(EMB_DIM, dtype=np.float32)]

        for row in rel_rows:
            rel_name = row["relation"]
            vec = _float_row(row, REL_PREFIX, EMB_DIM)
            self.relation_to_idx[rel_name] = len(rel_emb_list)
            rel_emb_list.append(vec)

        self.rel_transe_matrix = torch.tensor(
            np.stack(rel_emb_list, axis=0), dtype=torch.float32
        )  # [N_rels+1, 128]
        self.n_relations = len(rel_emb_list) - 1  # excluding PAD

        print(f"    {self.n_relations} relation types loaded.")

        # 3. Load drug-level embeddings
        print("  Loading drug-level embeddings (Word2Vec + TransE) …")
        drug_rows = _read_csv(master_emb_path)

        self.rxnorm_to_drug_idx: dict[str, int] = {}
        self.drug_idx_to_cui: dict[int, str] = {}
        w2v_list: list[np.ndarray] = []
        drug_transe_list: list[np.ndarray] = []

        for row in drug_rows:
            rxnorm = row["rxnorm"]
            cui    = row["CUI"]
            idx    = len(w2v_list)
            self.rxnorm_to_drug_idx[rxnorm] = idx
            self.drug_idx_to_cui[idx] = cui

            w2v_vec  = _float_row(row, W2V_PREFIX, EMB_DIM)
            node_idx = self.cui_to_node_idx.get(cui, PAD_IDX)
            tr_vec   = self.node_transe_matrix[node_idx].numpy()

            w2v_list.append(w2v_vec)
            drug_transe_list.append(tr_vec)

        self.n_drugs = len(w2v_list)
        self.drug_w2v_matrix = torch.tensor(
            np.stack(w2v_list, axis=0), dtype=torch.float32
        )  # [N_drugs, 128]
        self.drug_transe_matrix = torch.tensor(
            np.stack(drug_transe_list, axis=0), dtype=torch.float32
        )  # [N_drugs, 128]

        # 4. Build neighbor + relation index matrices
        print("  Building neighborhood + relation matrices …")
        nb_rows = _read_csv(neighborhood_path)

        cui_to_nb_node_indices: dict[str, list[int]] = {}
        cui_to_nb_rel_indices:  dict[str, list[int]] = {}
        missing_rels: set[str] = set()

        for row in nb_rows:
            cui        = row["CUI"]
            node_idxs  = []
            rel_idxs   = []
            for k in range(1, NUM_NEIGHBORS + 1):
                nb_cui  = row.get(f"neighbor_{k}", "")
                rel_str = row.get(f"relation_{k}", "")
                node_idxs.append(self.cui_to_node_idx.get(nb_cui, PAD_IDX))
                rel_idx = self.relation_to_idx.get(rel_str, PAD_IDX)
                if rel_str and rel_idx == PAD_IDX:
                    missing_rels.add(rel_str)
                rel_idxs.append(rel_idx)
            cui_to_nb_node_indices[cui] = node_idxs
            cui_to_nb_rel_indices[cui]  = rel_idxs

        if missing_rels:
            print(f"  WARNING: {len(missing_rels)} unseen relation(s) mapped to PAD: {missing_rels}")

        neighbor_idx_list: list[list[int]] = []
        relation_idx_list: list[list[int]] = []
        for drug_idx in range(self.n_drugs):
            cui = self.drug_idx_to_cui[drug_idx]
            neighbor_idx_list.append(
                cui_to_nb_node_indices.get(cui, [PAD_IDX] * NUM_NEIGHBORS)
            )
            relation_idx_list.append(
                cui_to_nb_rel_indices.get(cui, [PAD_IDX] * NUM_NEIGHBORS)
            )

        self.neighbor_matrix = torch.tensor(
            np.array(neighbor_idx_list, dtype=np.int64)
        )  # [N_drugs, 15]
        self.relation_matrix = torch.tensor(
            np.array(relation_idx_list, dtype=np.int64)
        )  # [N_drugs, 15]

        print(
            f"  EmbeddingStore ready: {self.n_drugs} drugs | "
            f"{self.node_transe_matrix.shape[0]-1} KG nodes | "
            f"{self.n_relations} relation types."
        )

    def get_drug_embeddings(self, drug_indices: torch.Tensor):

        # Return Word2Vec and TransE embeddings for a batch of drug indices.
        
        return (
            self.drug_w2v_matrix[drug_indices],
            self.drug_transe_matrix[drug_indices],
        )

    def get_neighbor_and_relation_embeddings(self, drug_indices: torch.Tensor):
        """
        Return TransE embeddings for both neighbor nodes and their edge
        relation types for a batch of drugs.
        """
        nb_node_idxs = self.neighbor_matrix[drug_indices]   # [B, 15]
        nb_rel_idxs  = self.relation_matrix[drug_indices]   # [B, 15]
        return (
            self.node_transe_matrix[nb_node_idxs],          # [B, 15, 128]
            self.rel_transe_matrix[nb_rel_idxs],            # [B, 15, 128]
        )

# DDIDataset

class DDIDataset(Dataset):
    # Each sample: (drug_a_idx, drug_b_idx, label)
    
    def __init__(
        self,
        pairs: list[tuple[int, int, int]],
    ):
        self.pairs = pairs   # list of (a_idx, b_idx, label)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        a, b, y = self.pairs[i]
        return (
            torch.tensor(a, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )

# Dataset factory 

def build_datasets(
    ddi_path: str,
    store: EmbeddingStore,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[DDIDataset, DDIDataset, DDIDataset]:

    print("  Reading DDI pairs …")
    rows = _read_csv(ddi_path)

    pairs: list[tuple[int, int, int]] = []
    skipped = 0
    for row in rows:
        rxn_a = row["drug_a_rxnorm"]
        rxn_b = row["drug_b_rxnorm"]
        label = int(row["interaction"])

        idx_a = store.rxnorm_to_drug_idx.get(rxn_a)
        idx_b = store.rxnorm_to_drug_idx.get(rxn_b)
        if idx_a is None or idx_b is None:
            skipped += 1
            continue
        pairs.append((idx_a, idx_b, label))

    if skipped:
        print(f"  WARNING: skipped {skipped} pairs (missing rxnorm in embeddings).")

    labels = [p[2] for p in pairs]
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"  Total pairs: {len(pairs):,}  |  pos={pos:,}  neg={neg:,}")

    # Stratified splits
    idx_all = list(range(len(pairs)))
    idx_train_val, idx_test = train_test_split(
        idx_all, test_size=test_ratio, stratify=labels, random_state=seed
    )
    labels_tv = [labels[i] for i in idx_train_val]
    val_frac = val_ratio / (1 - test_ratio)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=val_frac, stratify=labels_tv, random_state=seed
    )

    train_pairs = [pairs[i] for i in idx_train]
    val_pairs   = [pairs[i] for i in idx_val]
    test_pairs  = [pairs[i] for i in idx_test]

    print(
        f"  Split → train={len(train_pairs):,}  "
        f"val={len(val_pairs):,}  test={len(test_pairs):,}"
    )
    return DDIDataset(train_pairs), DDIDataset(val_pairs), DDIDataset(test_pairs)


# Inductive split factory

def build_inductive_splits(
    ddi_path: str,
    store: EmbeddingStore,
    unseen_drug_fractions: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    seed: int = 42,
) -> dict[float, dict]:
    """
    Drug-level holdout split for inductive generalisation evaluation.

    For each fraction f:
      - Hold out f% of drugs as 'unseen' (withheld from training pairs entirely)
      - train            : both drugs in the seen set
      - inductive        : both drugs in the unseen set  (hardest — zero pair exposure)
      - semi_inductive   : one seen drug, one unseen drug  (realistic new-drug scenario)

    Embeddings are frozen and pre-computed for all drugs, so this tests
    pair-level inductive generalisation only. Drugs in the unseen set still
    have their TransE / FastText vectors available at inference time.

    Args:
        ddi_path               : path to DDI CSV
        store                  : EmbeddingStore (already loaded)
        unseen_drug_fractions  : list of holdout fractions to evaluate
        seed                   : random seed for reproducibility

    Returns:
        dict keyed by fraction, each value a dict with keys:
            train           : DDIDataset
            inductive       : DDIDataset
            semi_inductive  : DDIDataset
            n_seen_drugs    : int
            n_unseen_drugs  : int
            stats           : dict with pair counts per split
    """
    print("  Reading DDI pairs for inductive split …")
    rows = _read_csv(ddi_path)

    pairs: list[tuple[int, int, int]] = []
    for row in rows:
        rxn_a = row["drug_a_rxnorm"]
        rxn_b = row["drug_b_rxnorm"]
        label = int(row["interaction"])
        idx_a = store.rxnorm_to_drug_idx.get(rxn_a)
        idx_b = store.rxnorm_to_drug_idx.get(rxn_b)
        if idx_a is None or idx_b is None:
            continue
        pairs.append((idx_a, idx_b, label))

    all_drug_indices = list(range(store.n_drugs))
    results = {}

    for frac in unseen_drug_fractions:
        seen_drugs, unseen_drugs = train_test_split(
            all_drug_indices,
            test_size=frac,
            random_state=seed,
        )
        seen_set   = set(seen_drugs)
        unseen_set = set(unseen_drugs)

        train_pairs:          list[tuple[int, int, int]] = []
        inductive_pairs:      list[tuple[int, int, int]] = []
        semi_inductive_pairs: list[tuple[int, int, int]] = []

        for (a, b, y) in pairs:
            a_seen = a in seen_set
            b_seen = b in seen_set
            if a_seen and b_seen:
                train_pairs.append((a, b, y))
            elif not a_seen and not b_seen:
                inductive_pairs.append((a, b, y))
            else:
                semi_inductive_pairs.append((a, b, y))

        results[frac] = {
            "train":          DDIDataset(train_pairs),
            "inductive":      DDIDataset(inductive_pairs),
            "semi_inductive": DDIDataset(semi_inductive_pairs),
            "n_seen_drugs":   len(seen_set),
            "n_unseen_drugs": len(unseen_set),
            "stats": {
                "train_pairs":          len(train_pairs),
                "inductive_pairs":      len(inductive_pairs),
                "semi_inductive_pairs": len(semi_inductive_pairs),
            },
        }

        print(
            f"  frac={frac:.0%} | unseen_drugs={len(unseen_set)} | "
            f"train={len(train_pairs):,} | "
            f"inductive={len(inductive_pairs):,} | "
            f"semi_inductive={len(semi_inductive_pairs):,}"
        )

    return results


def make_loader(
    dataset: DDIDataset,
    batch_size: int = 2048,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
