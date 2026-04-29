import pandas as pd
import networkx as nx
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from configs.config import PROCESSED, PROCESSED_FILES, NODE_TYPE_PRIORITY
from src.ingestion.parse_umls import parse_umls_all


# ── Relation type registry ────────────────────────────────────────────────────
from configs.config import PROCESSED, PROCESSED_FILES, NODE_TYPE_PRIORITY, RELATION_MAP

# Derive relation registry directly from RELATION_MAP's canonical names.
# Using a set → sorted list so the mapping is stable across runs.
RELATION_TYPES = sorted(set(RELATION_MAP.values()))
RELATION2IDX   = {r: i for i, r in enumerate(RELATION_TYPES)}
IDX2RELATION   = {i: r for r, i in RELATION2IDX.items()}


def build_knowledge_graph(include_definitions: bool = False):
 

    # ── Step 1: Parse UMLS ───────────────────────────────────────────────────
    sty_df, name_df, rel_df, def_df = parse_umls_all()

    # ── Step 2: Build node_df ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Building node table")
    print("=" * 60)

    sty_df = sty_df.copy()
    sty_df["type_rank"] = sty_df["node_type"].map(NODE_TYPE_PRIORITY).fillna(99)
    primary_sty = (
        sty_df
        .sort_values(["cui", "type_rank"])
        .drop_duplicates(subset=["cui"], keep="first")
        [["cui", "node_type"]]
    )

    # Merge with preferred names
    node_df = primary_sty.merge(name_df, on="cui", how="left")
    node_df = node_df.rename(columns={
        "cui" : "CUI",
        "name": "preferred_name",
        "sab" : "SAB",
    })

    # Flag CUIs with no preferred name (they'll still be in the graph)
    missing_names = node_df["preferred_name"].isna().sum()
    if missing_names:
        print(f"[WARN] {missing_names:,} CUIs have no preferred name — "
              f"falling back to CUI as name")
        node_df["preferred_name"] = node_df["preferred_name"].fillna(node_df["CUI"])

    print(f"Node table: {len(node_df):,} nodes")
    print(node_df["node_type"].value_counts().to_string())

    # ── Step 3: Build edge_df ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Building edge table")
    print("=" * 60)

    edge_df = rel_df.rename(columns={
        "cui1"          : "CUI1",
        "cui2"          : "CUI2",
        "rela_original" : "RELA",
        "sab"           : "SAB",
    })

    print(f"Edge table: {len(edge_df):,} edges")

    # ── Step 4: Assign integer node IDs ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Assigning integer node IDs")
    print("=" * 60)
    
    #Find all CUIs that actually appear in the edge table
    connected_cuis = set(edge_df["CUI1"]).union(set(edge_df["CUI2"]))
    
    #Filter node_df to only include those connected CUIs
    before_nodes = len(node_df)
    node_df = node_df[node_df["CUI"].isin(connected_cuis)].reset_index(drop=True)
    print(f"Dropped {before_nodes - len(node_df):,} isolated nodes")
    print(f"Final nodes: {len(node_df):,}")

    unique_cuis = node_df["CUI"].unique()
    cui_to_id   = {cui: idx for idx, cui in enumerate(unique_cuis)}

    node_df = node_df.copy()
    node_df["node_id"] = node_df["CUI"].map(cui_to_id)

    edge_df = edge_df.copy()
    edge_df["src_id"] = edge_df["CUI1"].map(cui_to_id)
    edge_df["tgt_id"] = edge_df["CUI2"].map(cui_to_id)

    # Drop edges where either endpoint is not in the node table
    before = len(edge_df)
    edge_df = edge_df.dropna(subset=["src_id", "tgt_id"])
    edge_df["src_id"] = edge_df["src_id"].astype(int)
    edge_df["tgt_id"] = edge_df["tgt_id"].astype(int)
    print(f"Dropped {before - len(edge_df):,} edges with unknown CUIs")
    print(f"Final edges: {len(edge_df):,}")

    # Map RELA → integer relation ID (fallback to RO for anything unmapped)
    fallback_id = RELATION2IDX.get("RO", len(RELATION_TYPES) - 1)
    edge_df["relation_id"] = (
        edge_df["RELA"]
        .map(RELATION2IDX)
        .fillna(fallback_id)
        .astype(int)
    )

    # ── Step 5: Add node types to edges (needed for hetero PyG graph) ─────────
    cui_to_type = dict(zip(node_df["CUI"], node_df["node_type"]))
    edge_df["src_type"] = edge_df["CUI1"].map(cui_to_type).fillna("unknown")
    edge_df["tgt_type"] = edge_df["CUI2"].map(cui_to_type).fillna("unknown")

    # ── Step 6: Build NetworkX graph ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Assembling NetworkX graph")
    print("=" * 60)

    G = nx.MultiDiGraph()

    for _, row in tqdm(node_df.iterrows(),
                       total=len(node_df), desc="Adding nodes"):
        G.add_node(
            int(row["node_id"]),
            cui       = row["CUI"],
            name      = row["preferred_name"],
            node_type = row["node_type"],
            sab       = row.get("SAB", ""),
        )

    for _, row in tqdm(edge_df.iterrows(),
                       total=len(edge_df), desc="Adding edges"):
        G.add_edge(
            int(row["src_id"]),
            int(row["tgt_id"]),
            relation    = row["RELA"],
            relation_id = int(row["relation_id"]),
            sab         = row["SAB"],
        )

    print(f"\n{'='*60}")
    print(f"GRAPH COMPLETE")
    print(f"  Nodes : {G.number_of_nodes():,}")
    print(f"  Edges : {G.number_of_edges():,}")
    print(f"\n  Node types:")
    for t, n in node_df["node_type"].value_counts().items():
        print(f"    {t:25s}: {n:,}")
    print(f"\n  Edge types (top 15):")
    for t, n in edge_df["RELA"].value_counts().head(15).items():
        print(f"    {t:40s}: {n:,}")

    return G, node_df, edge_df


def export_to_pyg(node_df, edge_df, out_path=None):
    """
    Converts to PyTorch Geometric HeteroData.
    Node features = learnable integer indices (nn.Embedding).
    """
    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        print("[ERROR] pip install torch-geometric")
        return None

    out_path = out_path or PROCESSED_FILES["graph_pyg"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    data = HeteroData()

    # ── Build per-type local ID maps ──────────────────────────────────────────
    node_type_local: dict[tuple, int] = {}

    for ntype in node_df["node_type"].unique():
        type_nodes = (
            node_df[node_df["node_type"] == ntype]
            .reset_index(drop=True)
        )
        n = len(type_nodes)
        for local_id, (_, row) in enumerate(type_nodes.iterrows()):
            node_type_local[(ntype, int(row["node_id"]))] = local_id

        data[ntype].x         = torch.arange(n, dtype=torch.long)
        data[ntype].num_nodes = n
        data[ntype].cuis      = type_nodes["CUI"].tolist()
        data[ntype].names     = type_nodes["preferred_name"].tolist()
        print(f"  [{ntype}] {n:,} nodes")

    # ── Build per-relation edge indices ───────────────────────────────────────
    for rela in edge_df["RELA"].unique():
        rel_rows = edge_df[edge_df["RELA"] == rela]
        src_type = rel_rows["src_type"].iloc[0]
        tgt_type = rel_rows["tgt_type"].iloc[0]

        if src_type == "unknown" or tgt_type == "unknown":
            continue

        srcs, tgts = [], []
        for _, row in rel_rows.iterrows():
            s = node_type_local.get((src_type, int(row["src_id"])), -1)
            t = node_type_local.get((tgt_type, int(row["tgt_id"])), -1)
            if s >= 0 and t >= 0:
                srcs.append(s)
                tgts.append(t)

        if not srcs:
            continue

        ei = torch.tensor([srcs, tgts], dtype=torch.long)
        data[src_type, rela, tgt_type].edge_index = ei
        print(f"  ({src_type}, {rela}, {tgt_type}): {len(srcs):,}")

    torch.save(data, out_path)
    print(f"\n[PyG] Saved → {out_path}")
    return data


def save_graph_artifacts(G, node_df, edge_df):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    node_df.to_csv(PROCESSED_FILES["nodes"], sep="\t", index=False)
    edge_df.to_csv(PROCESSED_FILES["edges"], sep="\t", index=False)
    with open(PROCESSED_FILES["graph_pkl"], "wb") as f:
        pickle.dump(G, f)
    print(f"[Saved] nodes → {PROCESSED_FILES['nodes']}")
    print(f"[Saved] edges → {PROCESSED_FILES['edges']}")
    print(f"[Saved] graph → {PROCESSED_FILES['graph_pkl']}")


if __name__ == "__main__":
    G, node_df, edge_df = build_knowledge_graph()
    save_graph_artifacts(G, node_df, edge_df)
    export_to_pyg(node_df, edge_df)
