# -*- coding: utf-8 -*-
"""
train.py
--------
Complete training loop implementing:
  - Binary Cross-Entropy with pos_weight for class imbalance
  - Gradient norm clipping
  - OneCycleLR scheduler (linear warmup + cosine decay)
  - Mixed precision (autocast + GradScaler)
  - AUROC / AUPRC primary metrics; F1 / Accuracy secondary
  - Checkpoint saving (best AUROC)
  - Ablation runner
  - Inductive generalisation sweep (drug-level holdout)

COMMANDS
--------
Standard training (full model):
    python train.py

All ablation variants (pair-level split):
    python train.py --ablation

Inductive sweep (full model, drug-level split):
    python train.py --inductive

Inductive sweep (all ablation variants, drug-level split):
    python train.py --inductive --ablation

Inductive sweep (single variant):
    python train.py --inductive --use_semantic_only

Custom fractions / epochs for inductive:
    python train.py --inductive --ablation --fractions 0.1 0.2 0.3 0.4 0.5 --epochs 30
"""

import os
import time
import math
import json
import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
import numpy as np

from data_loader import EmbeddingStore, build_datasets, build_inductive_splits, make_loader
from model import DDIModel

# Default config (override via CLI or ablation dict)

DEFAULT_CFG = dict(
    # Paths
    master_emb_path      = "data/master_drug_embeddings.csv",
    transe_nb_path       = "data/transe_neighbors_dict.csv",
    transe_rel_path      = "data/transe_relation_embeddings.csv",
    neighborhood_path    = "data/relation_aware_neighborhood_15.csv",
    ddi_path             = "data/ddi_training_dataset_TRANSE_ready.csv",
    output_dir           = "outputs",

    # Training
    epochs               = 30,
    batch_size           = 2048,
    lr                   = 3e-4,
    weight_decay         = 1e-4,
    grad_clip            = 1.0,
    num_workers          = 4,
    seed                 = 42,
    val_ratio            = 0.10,
    test_ratio           = 0.10,

    # Model
    dim                  = 128,
    n_heads              = 4,
    attn_dropout         = 0.10,
    mlp_dropout          = 0.30,
    mlp_hidden           = [512, 256, 64],

    # Scheduler
    pct_start            = 0.10,    # fraction of training for warmup

    # Ablation flags (all False = full model)
    use_semantic_only    = False,
    use_naive_concat     = False,
    use_flat_attention   = False,
    use_naive_pair_concat= False,
    use_structural_only  = False,
    use_no_neighborhood  = False,
    use_no_relations     = False,

    # Inductive sweep options
    inductive_fractions  = [0.1, 0.2, 0.3, 0.4, 0.5],
)

# Ablation configs (shared by both pair-level and inductive runners)

ABLATION_CONFIGS = [
    ("Full",             {}),
    ("StructuralOnly",   {"use_structural_only":   True}),
    ("SemanticOnly",     {"use_semantic_only":      True}),
    ("NoNeighborhood",   {"use_no_neighborhood":    True}),
    ("NoRelations",      {"use_no_relations":       True}),
    ("NaiveConcatFusion",{"use_naive_concat":       True}),
    ("FlatAttention",    {"use_flat_attention":     True}),
    ("NaivePairConcat",  {"use_naive_pair_concat":  True}),
    ("FullBaseline",     {"use_semantic_only": True,
                          "use_flat_attention": True,
                          "use_naive_pair_concat": True}),
]

# Helpers

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(dataset) -> float:
    # Returns neg/pos ratio for BCE pos_weight.
    labels = [p[2] for p in dataset.pairs]
    pos = max(sum(labels), 1)
    neg = len(labels) - pos
    return neg / pos


def _model_flags(cfg: dict) -> dict:
    # Extract just the DDIModel ablation kwargs from a cfg dict.
    return {
        "use_semantic_only":     cfg["use_semantic_only"],
        "use_structural_only":   cfg["use_structural_only"],
        "use_naive_concat":      cfg["use_naive_concat"],
        "use_flat_attention":    cfg["use_flat_attention"],
        "use_naive_pair_concat": cfg["use_naive_pair_concat"],
        "use_no_neighborhood":   cfg["use_no_neighborhood"],
        "use_no_relations":      cfg["use_no_relations"],
    }


def _move_store_to_device(store, device):
    store.drug_w2v_matrix    = store.drug_w2v_matrix.to(device)
    store.drug_transe_matrix = store.drug_transe_matrix.to(device)
    store.node_transe_matrix = store.node_transe_matrix.to(device)
    store.rel_transe_matrix  = store.rel_transe_matrix.to(device)
    store.neighbor_matrix    = store.neighbor_matrix.to(device)
    store.relation_matrix    = store.relation_matrix.to(device)


def _load_store(cfg: dict) -> EmbeddingStore:
    return EmbeddingStore(
        master_emb_path       = cfg["master_emb_path"],
        transe_neighbors_path = cfg["transe_nb_path"],
        transe_relation_path  = cfg["transe_rel_path"],
        neighborhood_path     = cfg["neighborhood_path"],
    )

# Evaluation

@torch.no_grad()
def evaluate(model, loader, store, device, criterion) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for drug_a, drug_b, labels in loader:
        drug_a = drug_a.to(device)
        drug_b = drug_b.to(device)
        labels = labels.to(device)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(drug_a, drug_b, store)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = torch.sigmoid(torch.tensor(all_logits)).numpy()
    all_preds  = (all_probs >= 0.5).astype(int)

    n = len(all_labels)
    metrics = {
        "loss"    : total_loss / n,
        "auroc"   : roc_auc_score(all_labels, all_probs),
        "auprc"   : average_precision_score(all_labels, all_probs),
        "f1"      : f1_score(all_labels, all_preds, zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }
    return metrics


@torch.no_grad()
def evaluate_inductive(model, dataset, store, device, batch_size=2048) -> dict | None:
    """
    Evaluates on a DDIDataset without a criterion (no loss computed).
    Returns None if the split is empty or has only one class.
    """
    if len(dataset) == 0:
        return None

    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []

    model.eval()
    for drug_a, drug_b, labels in loader:
        drug_a = drug_a.to(device)
        drug_b = drug_b.to(device)
        logits = model(drug_a, drug_b, store)
        probs  = torch.sigmoid(logits).float().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    if len(set(all_labels)) < 2:
        return None

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs >= 0.5).astype(int)

    return {
        "auroc"   : roc_auc_score(all_labels, all_probs),
        "auprc"   : average_precision_score(all_labels, all_probs),
        "f1"      : f1_score(all_labels, all_preds, zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }

# Training loop

def train_one_epoch(
    model, loader, store, optimizer, scheduler, scaler,
    criterion, device, grad_clip: float, epoch: int,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for step, (drug_a, drug_b, labels) in enumerate(loader):
        drug_a = drug_a.to(device, non_blocking=True)
        drug_b = drug_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(drug_a, drug_b, store)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return {"loss": total_loss / max(n_batches, 1), "lr": scheduler.get_last_lr()[0]}


def _build_optimizer_and_scheduler(model, cfg, n_steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )
    total_steps = cfg["epochs"] * n_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = cfg["lr"],
        total_steps      = max(total_steps, 1),
        pct_start        = cfg["pct_start"],
        anneal_strategy  = "cos",
        div_factor       = 25.0,
        final_div_factor = 1e4,
    )
    return optimizer, scheduler

# Main training entry point  (pair-level split)


def train(cfg: dict) -> dict:
    """
    Runs the full training pipeline for a given config dict.
    Returns best val metrics and test metrics (evaluated once on best model).
    """
    set_seed(cfg["seed"])
    device  = get_device()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    flags     = _model_flags(cfg)
    model_tag = DDIModel(**flags).config_str

    print(f"\n{'='*70}")
    print(f"  Model config : {model_tag}")
    print(f"  Device       : {device}")
    print(f"{'='*70}")

    # Load embeddings 
    print("\n[1/4] Loading embeddings")
    store = _load_store(cfg)

    # Build datasets
    print("\n[2/4] Building datasets")
    train_ds, val_ds, test_ds = build_datasets(
        ddi_path   = cfg["ddi_path"],
        store      = store,
        val_ratio  = cfg["val_ratio"],
        test_ratio = cfg["test_ratio"],
        seed       = cfg["seed"],
    )

    train_loader = make_loader(train_ds, cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"])
    val_loader   = make_loader(val_ds,   cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader  = make_loader(test_ds,  cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    # Model
    print("\n[3/4] Building model")
    model = DDIModel(
        dim          = cfg["dim"],
        n_heads      = cfg["n_heads"],
        mlp_hidden   = cfg["mlp_hidden"],
        attn_dropout = cfg["attn_dropout"],
        mlp_dropout  = cfg["mlp_dropout"],
        **flags,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    print(f"  Moving embedding store to {device}")
    _move_store_to_device(store, device)

    # Loss 
    pw = compute_pos_weight(train_ds)
    print(f"  pos_weight (neg/pos): {pw:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    # Optimizer + scheduler
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg, len(train_loader))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    print(f"\n[4/4] Training for {cfg['epochs']} epochs\n")
    best_auroc       = -1.0
    best_state       = None
    best_val_metrics = {}
    history          = []
    ckpt_path        = out_dir / f"best_{model_tag}.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, store, optimizer, scheduler, scaler,
            criterion, device, cfg["grad_clip"], epoch,
        )
        val_metrics = evaluate(model, val_loader, store, device, criterion)
        elapsed     = time.time() - t0

        print(
            f"Ep {epoch:>3}/{cfg['epochs']}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"AUROC={val_metrics['auroc']:.4f}  "
            f"AUPRC={val_metrics['auprc']:.4f}  "
            f"F1={val_metrics['f1']:.4f}  "
            f"lr={train_metrics['lr']:.2e}  "
            f"[{elapsed:.1f}s]"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["auroc"] > best_auroc:
            best_auroc       = val_metrics["auroc"]
            best_state       = deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            torch.save(
                {"epoch": epoch, "state_dict": best_state,
                 "val_metrics": val_metrics, "cfg": cfg},
                ckpt_path,
            )
            print(f"  [OK] New best AUROC={best_auroc:.4f} - checkpoint saved.")

    # Test set evaluation
    print("\n-- Test set evaluation (best checkpoint) --")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, store, device, criterion)
    print(
        f"  AUROC={test_metrics['auroc']:.4f}  "
        f"AUPRC={test_metrics['auprc']:.4f}  "
        f"F1={test_metrics['f1']:.4f}  "
        f"Acc={test_metrics['accuracy']:.4f}"
    )

    results = {
        "model_config": model_tag,
        "best_val"    : best_val_metrics,
        "test"        : test_metrics,
        "history"     : history,
    }

    result_path = out_dir / f"results_{model_tag}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved -> {result_path}")

    return results

# Pair-level ablation runner  (python train.py --ablation)

def run_ablation(base_cfg: dict):
    """Runs all ablation variants with pair-level splits and prints a summary table."""
    summary = []
    for name, overrides in ABLATION_CONFIGS:
        print(f"\n\n{'#'*70}")
        print(f"  ABLATION: {name}")
        print(f"{'#'*70}")
        cfg     = {**base_cfg, **overrides}
        results = train(cfg)
        summary.append({
            "variant"   : name,
            "val_auroc" : results["best_val"].get("auroc", 0),
            "val_auprc" : results["best_val"].get("auprc", 0),
            "test_auroc": results["test"].get("auroc", 0),
            "test_auprc": results["test"].get("auprc", 0),
            "test_f1"   : results["test"].get("f1", 0),
        })

    print(f"\n\n{'='*70}")
    print("  ABLATION SUMMARY")
    print(f"{'='*70}")
    hdr = (f"{'Variant':<25}  {'ValAUROC':>10}  {'ValAUPRC':>10}  "
           f"{'TestAUROC':>10}  {'TestAUPRC':>10}  {'TestF1':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(
            f"{s['variant']:<25}  "
            f"{s['val_auroc']:>10.4f}  {s['val_auprc']:>10.4f}  "
            f"{s['test_auroc']:>10.4f}  {s['test_auprc']:>10.4f}  "
            f"{s['test_f1']:>8.4f}"
        )

    out_path = Path(base_cfg["output_dir"]) / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary saved -> {out_path}")


# Inductive training  (one model, one fraction)

def train_inductive_one_fraction(
    model_flags: dict,
    split: dict,
    cfg: dict,
    store,
    device,
    model_tag: str,
) -> dict:
    """
    Trains a fresh model on seen-drug pairs for one unseen fraction,
    then evaluates on inductive and semi-inductive test sets.
    Returns a result dict for that fraction.
    """
    frac  = split["unseen_frac"]
    stats = split["stats"]

    print(f"\n{'='*60}")
    print(f"  [{model_tag}]  Unseen fraction: {frac:.0%}  |  "
          f"seen={split['n_seen_drugs']} drugs  unseen={split['n_unseen_drugs']} drugs")
    print(f"  train={stats['train_pairs']:,}  |  "
          f"inductive={stats['inductive_pairs']:,}  |  "
          f"semi_inductive={stats['semi_inductive_pairs']:,}")
    print(f"{'='*60}")

    if stats["train_pairs"] == 0:
        print("  [SKIP] No training pairs at this fraction.")
        return None

    model = DDIModel(
        dim          = cfg["dim"],
        n_heads      = cfg["n_heads"],
        mlp_hidden   = cfg["mlp_hidden"],
        attn_dropout = cfg["attn_dropout"],
        mlp_dropout  = cfg["mlp_dropout"],
        **model_flags,
    ).to(device)

    pw = compute_pos_weight(split["train"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    train_loader = make_loader(
        split["train"],
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = cfg["num_workers"],
    )

    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg, len(train_loader))
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Train
    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, store, optimizer, scheduler,
            scaler, criterion, device, cfg["grad_clip"], epoch,
        )
        if epoch % max(1, cfg["epochs"] // 5) == 0 or epoch == cfg["epochs"]:
            print(f"  Ep {epoch:>3}/{cfg['epochs']}  "
                  f"loss={train_metrics['loss']:.4f}  "
                  f"lr={train_metrics['lr']:.2e}")

    # Evaluate
    inductive_metrics     = evaluate_inductive(model, split["inductive"],     store, device, cfg["batch_size"])
    semi_inductive_metrics= evaluate_inductive(model, split["semi_inductive"],store, device, cfg["batch_size"])

    def _fmt(m):
        return (f"AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}") if m else "N/A (empty or single-class)"

    print(f"\n  inductive      : {_fmt(inductive_metrics)}")
    print(f"  semi_inductive : {_fmt(semi_inductive_metrics)}")

    return {
        "model_variant":        model_tag,
        "unseen_frac":          frac,
        "n_seen_drugs":         split["n_seen_drugs"],
        "n_unseen_drugs":       split["n_unseen_drugs"],
        "train_pairs":          stats["train_pairs"],
        "inductive_pairs":      stats["inductive_pairs"],
        "semi_inductive_pairs": stats["semi_inductive_pairs"],
        "inductive":            inductive_metrics,
        "semi_inductive":       semi_inductive_metrics,
    }

# Inductive runner  (python train.py --inductive)

def run_inductive(cfg: dict):
    """
    Runs the inductive generalisation sweep for a single model variant.
    - Drug-level holdout: unseen_drug_fractions of drugs are held out entirely
    - Trains on seen-drug pairs only
    - Evaluates on inductive (both unseen) and semi-inductive (one unseen) sets
    - Save results to outputs/inductive_results_<variant>.json
    """
    set_seed(cfg["seed"])
    device  = get_device()
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    flags     = _model_flags(cfg)
    model_tag = DDIModel(**flags).config_str
    fractions = cfg["inductive_fractions"]

    print(f"\n{'='*70}")
    print(f"  INDUCTIVE SWEEP — {model_tag}")
    print(f"  Fractions : {fractions}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  Device    : {device}")
    print(f"{'='*70}")

    print("\n[1/2] Loading embeddings")
    store = _load_store(cfg)
    _move_store_to_device(store, device)

    print("\n[2/2] Building inductive splits")
    splits_by_frac = build_inductive_splits(
        ddi_path              = cfg["ddi_path"],
        store                 = store,
        unseen_drug_fractions = fractions,
        seed                  = cfg["seed"],
    )

    summary = []
    for frac in fractions:
        split_data = {**splits_by_frac[frac], "unseen_frac": frac}
        result = train_inductive_one_fraction(
            flags, split_data, cfg, store, device, model_tag
        )
        if result is not None:
            summary.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  INDUCTIVE SUMMARY — {model_tag}")
    print(f"{'='*70}")
    hdr = (f"{'Frac':>6}  {'Unseen':>8}  "
           f"{'Ind.AUROC':>10}  {'Ind.AUPRC':>10}  "
           f"{'Semi.AUROC':>11}  {'Semi.AUPRC':>11}")
    print(hdr)
    print("-" * len(hdr))
    for row in summary:
        def _v(m, k):
            return f"{m[k]:.4f}" if m else "  N/A  "
        print(
            f"{row['unseen_frac']:>6.0%}  "
            f"{row['n_unseen_drugs']:>8}  "
            f"{_v(row['inductive'], 'auroc'):>10}  "
            f"{_v(row['inductive'], 'auprc'):>10}  "
            f"{_v(row['semi_inductive'], 'auroc'):>11}  "
            f"{_v(row['semi_inductive'], 'auprc'):>11}"
        )

    out_path = out_dir / f"inductive_results_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved -> {out_path}")

    return summary


# Inductive ablation runner  (python train.py --inductive --ablation)


def run_inductive_ablation(base_cfg: dict):
    """Runs the inductive sweep for every ablation variant sequentially."""
    all_results = {}

    for name, overrides in ABLATION_CONFIGS:
        print(f"\n\n{'#'*70}")
        print(f"  INDUCTIVE ABLATION: {name}")
        print(f"{'#'*70}")
        cfg = {**base_cfg, **overrides}
        all_results[name] = run_inductive(cfg)

    # Cross-variant summary at 0.2 fraction (middle point) for a quick comparison
    print(f"\n\n{'='*70}")
    print("  INDUCTIVE ABLATION SUMMARY  (@ 20% unseen drugs)")
    print(f"{'='*70}")
    hdr = (f"{'Variant':<25}  "
           f"{'Ind.AUROC':>10}  {'Ind.AUPRC':>10}  "
           f"{'Semi.AUROC':>11}  {'Semi.AUPRC':>11}")
    print(hdr)
    print("-" * len(hdr))

    for name, rows in all_results.items():
        # Find the row closest to 0.2
        row = min(rows, key=lambda r: abs(r["unseen_frac"] - 0.2)) if rows else None
        if row is None:
            print(f"{name:<25}  {'N/A':>10}  {'N/A':>10}  {'N/A':>11}  {'N/A':>11}")
            continue
        def _v(m, k):
            return f"{m[k]:.4f}" if m else "  N/A  "
        print(
            f"{name:<25}  "
            f"{_v(row['inductive'], 'auroc'):>10}  "
            f"{_v(row['inductive'], 'auprc'):>10}  "
            f"{_v(row['semi_inductive'], 'auroc'):>11}  "
            f"{_v(row['semi_inductive'], 'auprc'):>11}"
        )

    out_path = Path(base_cfg["output_dir"]) / "inductive_ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved -> {out_path}")


# CLI


def parse_args():
    p = argparse.ArgumentParser(
        description="DDI Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    p.add_argument("--master_emb_path",   default=DEFAULT_CFG["master_emb_path"])
    p.add_argument("--transe_nb_path",    default=DEFAULT_CFG["transe_nb_path"])
    p.add_argument("--transe_rel_path",   default=DEFAULT_CFG["transe_rel_path"])
    p.add_argument("--neighborhood_path", default=DEFAULT_CFG["neighborhood_path"])
    p.add_argument("--ddi_path",          default=DEFAULT_CFG["ddi_path"])
    p.add_argument("--output_dir",        default=DEFAULT_CFG["output_dir"])

    # Training
    p.add_argument("--epochs",       type=int,   default=DEFAULT_CFG["epochs"])
    p.add_argument("--batch_size",   type=int,   default=DEFAULT_CFG["batch_size"])
    p.add_argument("--lr",           type=float, default=DEFAULT_CFG["lr"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_CFG["weight_decay"])
    p.add_argument("--grad_clip",    type=float, default=DEFAULT_CFG["grad_clip"])
    p.add_argument("--num_workers",  type=int,   default=DEFAULT_CFG["num_workers"])
    p.add_argument("--seed",         type=int,   default=DEFAULT_CFG["seed"])
    p.add_argument("--pct_start",    type=float, default=DEFAULT_CFG["pct_start"])

    # Model
    p.add_argument("--n_heads",      type=int,   default=DEFAULT_CFG["n_heads"])
    p.add_argument("--attn_dropout", type=float, default=DEFAULT_CFG["attn_dropout"])
    p.add_argument("--mlp_dropout",  type=float, default=DEFAULT_CFG["mlp_dropout"])

    # Ablation flags
    p.add_argument("--use_semantic_only",     action="store_true")
    p.add_argument("--use_naive_concat",      action="store_true")
    p.add_argument("--use_flat_attention",    action="store_true")
    p.add_argument("--use_naive_pair_concat", action="store_true")
    p.add_argument("--use_structural_only",   action="store_true")
    p.add_argument("--use_no_neighborhood",   action="store_true")
    p.add_argument("--use_no_relations",      action="store_true")

    # Mode flags
    p.add_argument("--ablation",   action="store_true",
                   help="Run all ablation variants sequentially.")
    p.add_argument("--inductive",  action="store_true",
                   help="Run drug-level inductive generalisation sweep instead of pair-level training.")
    p.add_argument("--fractions",  nargs="+", type=float,
                   default=DEFAULT_CFG["inductive_fractions"],
                   help="Unseen-drug fractions for inductive sweep (default: 0.1 0.2 0.3 0.4 0.5).")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = {**DEFAULT_CFG, **vars(args)}
    # Rename fractions -> inductive_fractions; remove mode flags from cfg
    cfg["inductive_fractions"] = cfg.pop("fractions", DEFAULT_CFG["inductive_fractions"])
    cfg.pop("ablation",  None)
    cfg.pop("inductive", None)

    if args.inductive:
        if args.ablation:
            # All variants × all fractions
            run_inductive_ablation(cfg)
        else:
            # Single variant (full model unless ablation flag passed)
            run_inductive(cfg)
    else:
        if args.ablation:
            # All variants, pair-level split
            run_ablation(cfg)
        else:
            # Single variant, pair-level split
            train(cfg)
