
#Fast smoke test that validates all components WITHOUT requiring the full data files.

import os
import csv
import tempfile
import torch
import numpy as np

EMB_DIM      = 128
N_DRUGS      = 20
N_NODES      = 50     # drugs + neighbor-only nodes
N_PAIRS      = 200
N_NEIGHBORS  = 15
RNG          = np.random.default_rng(0)

# Match real relation types from transe_relation_embeddings.csv
RELATION_TYPES = [
    "associated_with", "form_of", "gene_associated_with_disease",
    "has_ingredient", "has_mechanism_of_action", "has_target",
    "has_tradename", "isa", "may_treat", "may_prevent",
    "has_physiologic_effect", "has_part", "part_of",
    "may_inhibit_effect_of", "may_diagnose",
    "gene_product_plays_role_in_biological_process",
]


def _rand_vec(d=EMB_DIM):
    return RNG.standard_normal(d).astype(np.float32)


# CSV writers

def write_master_emb(path, drug_records):
    graph_cols = [f"graph_{i}" for i in range(1, EMB_DIM + 1)]
    text_cols  = [f"text_{i}"  for i in range(1, EMB_DIM + 1)]
    fieldnames = ["CUI", "rxnorm", "node_id", "preferred_name", "name"] + graph_cols + text_cols
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in drug_records:
            row = {"CUI": r["cui"], "rxnorm": r["rxnorm"],
                   "node_id": r["node_id"], "preferred_name": r["name"], "name": r["name"]}
            for i, v in enumerate(r["w2v"],  1): row[f"graph_{i}"] = v
            for i, v in enumerate(r["text"], 1): row[f"text_{i}"]  = v
            w.writerow(row)


def write_transe_nodes(path, node_records):
    dim_cols   = [f"dim_{i}" for i in range(1, EMB_DIM + 1)]
    fieldnames = ["CUI"] + dim_cols
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in node_records:
            row = {"CUI": r["cui"]}
            for i, v in enumerate(r["transe"], 1): row[f"dim_{i}"] = v
            w.writerow(row)


def write_transe_relations(path, rel_types):
    """transe_relation_embeddings.csv — one 128-dim vec per relation type."""
    dim_cols   = [f"rel_dim_{i}" for i in range(1, EMB_DIM + 1)]
    fieldnames = ["relation"] + dim_cols
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rel in rel_types:
            row = {"relation": rel}
            vec = _rand_vec()
            for i, v in enumerate(vec, 1): row[f"rel_dim_{i}"] = v
            w.writerow(row)


def write_relation_neighborhood(path, drug_cuis, all_cuis, rel_types):
    """relation_aware_neighborhood_15.csv — interleaved neighbor_k / relation_k."""
    cols = ["CUI"]
    for k in range(1, N_NEIGHBORS + 1):
        cols += [f"neighbor_{k}", f"relation_{k}"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for cui in drug_cuis:
            row = {"CUI": cui}
            sampled_nb  = RNG.choice(all_cuis, size=N_NEIGHBORS, replace=False)
            sampled_rel = RNG.choice(rel_types, size=N_NEIGHBORS)
            for k, (nb, rel) in enumerate(zip(sampled_nb, sampled_rel), 1):
                row[f"neighbor_{k}"] = nb
                row[f"relation_{k}"] = rel
            w.writerow(row)


def write_ddi(path, rxnorms):
    fieldnames = ["drug_a_name", "drug_a_rxnorm", "drug_b_name", "drug_b_rxnorm", "interaction"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _ in range(N_PAIRS):
            a, b = RNG.choice(rxnorms, size=2, replace=False)
            w.writerow({"drug_a_name": f"drug_{a}", "drug_a_rxnorm": a,
                         "drug_b_name": f"drug_{b}", "drug_b_rxnorm": b,
                         "interaction": int(RNG.random() > 0.5)})


# ests

def build_synthetic_store(tmpdir):
    """Creates synthetic CSVs and returns (store, ddi_path)."""
    drug_records = [
        {"cui": f"C{i:07d}", "rxnorm": str(1000 + i), "node_id": i,
         "name": f"drug_{i}", "w2v": _rand_vec(), "text": _rand_vec()}
        for i in range(N_DRUGS)
    ]
    extra_nodes = [{"cui": f"N{j:07d}", "transe": _rand_vec()}
                   for j in range(N_NODES - N_DRUGS)]
    all_node_records = (
        [{"cui": d["cui"], "transe": _rand_vec()} for d in drug_records]
        + extra_nodes
    )
    all_cuis = np.array([r["cui"] for r in all_node_records])

    emb_path = os.path.join(tmpdir, "master_drug_embeddings.csv")
    tr_path  = os.path.join(tmpdir, "transe_neighbors_dict.csv")
    rel_path = os.path.join(tmpdir, "transe_relation_embeddings.csv")
    nb_path  = os.path.join(tmpdir, "relation_aware_neighborhood_15.csv")
    ddi_path = os.path.join(tmpdir, "ddi.csv")

    write_master_emb(emb_path, drug_records)
    write_transe_nodes(tr_path, all_node_records)
    write_transe_relations(rel_path, RELATION_TYPES)
    write_relation_neighborhood(nb_path, [d["cui"] for d in drug_records],
                                all_cuis, RELATION_TYPES)
    write_ddi(ddi_path, [str(1000 + i) for i in range(N_DRUGS)])

    from data_loader import EmbeddingStore
    store = EmbeddingStore(emb_path, tr_path, nb_path, rel_path)
    return store, ddi_path


def test_embedding_store(tmpdir):
    print("[1] EmbeddingStore … ", end="")
    from data_loader import build_datasets

    store, ddi_path = build_synthetic_store(tmpdir)

    assert store.drug_w2v_matrix.shape    == (N_DRUGS, EMB_DIM)
    assert store.drug_transe_matrix.shape == (N_DRUGS, EMB_DIM)
    assert store.neighbor_matrix.shape    == (N_DRUGS, N_NEIGHBORS)
    assert store.relation_matrix.shape    == (N_DRUGS, N_NEIGHBORS)
    assert store.rel_transe_matrix.shape  == (len(RELATION_TYPES) + 1, EMB_DIM)  # +1 for PAD
    assert store.node_transe_matrix.shape[1] == EMB_DIM

    # Lookup API
    idx = torch.tensor([0, 1, 2])
    w2v, tr = store.get_drug_embeddings(idx)
    assert w2v.shape == tr.shape == (3, EMB_DIM)

    nb_vecs, rel_vecs = store.get_neighbor_and_relation_embeddings(idx)
    assert nb_vecs.shape  == (3, N_NEIGHBORS, EMB_DIM)
    assert rel_vecs.shape == (3, N_NEIGHBORS, EMB_DIM)

    # Dataset splits
    train_ds, val_ds, test_ds = build_datasets(ddi_path, store,
                                               val_ratio=0.1, test_ratio=0.1)
    assert len(train_ds) + len(val_ds) + len(test_ds) == N_PAIRS
    print("PASS")
    return store


def test_model_forward(store):
    print("[2] Model forward pass (all ablation configs) … ", end="")
    from model import DDIModel

    B = 8
    idx = torch.arange(B)

    configs = [
        {},
        {"use_semantic_only": True},
        {"use_naive_concat": True},
        {"use_flat_attention": True},
        {"use_naive_pair_concat": True},
        {"use_semantic_only": True, "use_flat_attention": True, "use_naive_pair_concat": True},
    ]
    for kw in configs:
        model = DDIModel(**kw)
        model.eval()
        with torch.no_grad():
            logit = model(idx, idx, store)
        assert logit.shape == (B,), f"Expected [{B}], got {logit.shape} (config={kw})"

    print(f"PASS  ({len(configs)} configs)")


def test_training_step(store):
    print("[3] Training step (loss + backward + clip + scheduler) … ", end="")
    import torch.nn as nn
    from torch.cuda.amp import autocast, GradScaler
    from model import DDIModel

    device = torch.device("cpu")
    model  = DDIModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler    = GradScaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=10, pct_start=0.3,
    )

    B      = 16
    drug_a = torch.randint(0, N_DRUGS, (B,))
    drug_b = torch.randint(0, N_DRUGS, (B,))
    labels = torch.randint(0, 2, (B,)).float()

    model.train()
    optimizer.zero_grad()
    with autocast(enabled=False):
        logits = model(drug_a, drug_b, store)
        loss   = criterion(logits, labels)
    scaler.scale(loss).backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    assert not torch.isnan(loss), "NaN loss!"
    print("PASS")


def test_relation_embedding_wiring(store):
    # Verify that different relation vecs produce different attention outputs.
    print("[4] Relation embedding wiring … ", end="")
    from model import RelationAwareMultiHeadAttention

    attn = RelationAwareMultiHeadAttention(dim=EMB_DIM, n_heads=4)
    attn.eval()

    B, N, D = 2, N_NEIGHBORS, EMB_DIM
    drug_vec = torch.randn(B, D)
    nb_vecs  = torch.randn(B, N, D)
    rel_A    = torch.randn(B, N, D)   # real relation vecs
    rel_B    = torch.zeros(B, N, D)   # zero relation (no relation info)

    with torch.no_grad():
        out_A = attn(drug_vec, nb_vecs, rel_A)
        out_B = attn(drug_vec, nb_vecs, rel_B)

    assert not torch.allclose(out_A, out_B), \
        "Outputs should differ when relation vecs differ"
    assert out_A.shape == (B, D)
    print("PASS")


def test_ablation_component_types(store):
    print("[5] Ablation component type checks … ", end="")
    from model import (DDIModel, GatedFusionLayer, NaiveConcatFusion,
                       RelationAwareMultiHeadAttention, FlatAttentionAggregation,
                       DDIInteractionMLP, NaivePairConcatMLP)

    # Full model
    m = DDIModel()
    assert isinstance(m.fusion,            GatedFusionLayer)
    assert isinstance(m.neighborhood_attn, RelationAwareMultiHeadAttention)
    assert isinstance(m.interaction,       DDIInteractionMLP)

    # Semantic-only has no fusion AND no neighbourhood attn
    m = DDIModel(use_semantic_only=True)
    assert m.fusion is None
    assert m.neighborhood_attn is None

    assert isinstance(DDIModel(use_naive_concat=True).fusion,                  NaiveConcatFusion)
    assert isinstance(DDIModel(use_flat_attention=True).neighborhood_attn,     FlatAttentionAggregation)
    assert isinstance(DDIModel(use_naive_pair_concat=True).interaction,        NaivePairConcatMLP)
    print("PASS")


# Runner 

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    with tempfile.TemporaryDirectory() as tmpdir:
        store = test_embedding_store(tmpdir)
        test_model_forward(store)
        test_training_step(store)
        test_relation_embedding_wiring(store)
        test_ablation_component_types(store)

    print("\n✓ All smoke tests passed.")
