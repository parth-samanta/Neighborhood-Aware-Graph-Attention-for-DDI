
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from configs.config import (
    RAW_FILES, KEEP_SEMANTIC_TYPES, KEEP_SOURCES_CONCEPTS,
    KEEP_SOURCES_RELATIONS, RELATION_MAP, SEMANTIC_TYPE_TO_NODE_TYPE,
    NODE_TYPE_PRIORITY,
)



# Step 1: Parse MRSTY — get CUI -> semantic type mapping

def parse_mrsty(path: Path = None) -> pd.DataFrame:
    """
    Reads MRSTY.RRF and returns a DataFrame of CUIs with their
    semantic types, filtered to only the types we care about.

    Returns DataFrame with columns:
        cui, tui, sty (semantic type name), node_type
    """
    path = path or RAW_FILES["mrsty"]
    print(f"[MRSTY] Loading semantic types from {path.name}...")

    df = pd.read_csv(
        path, sep="|", header=None, index_col=False,
        names=["cui","tui","stn","sty","atui","cvf","empty"],
        low_memory=False,   
        dtype=str,
    )
    df = df[["cui","tui","sty"]].copy()
    print(f"[MRSTY] Total rows: {len(df):,}")

    # Filter to semantic types we want
    df = df[df["tui"].isin(KEEP_SEMANTIC_TYPES)].copy()
    print(f"[MRSTY] After semantic type filter: {len(df):,}")

    # Assign node type from TUI
    df["node_type"] = df["tui"].map(SEMANTIC_TYPE_TO_NODE_TYPE)

    # One CUI can have multiple semantic types — keep all, resolve later
    df = df.drop_duplicates(subset=["cui","tui"])

    print(f"[MRSTY] Unique CUIs with relevant semantic types: "
          f"{df['cui'].nunique():,}")
    print(f"[MRSTY] Node type distribution:")
    print(df["node_type"].value_counts().to_string())

    return df



# Step 2: Parse MRCONSO — get CUI -> preferred name mapping


def parse_mrconso(
    path         : Path = None,
    relevant_cuis: set  = None,
) -> pd.DataFrame:
    path = path or RAW_FILES["mrconso"]
    print(f"\n[MRCONSO] Loading concept names from {path.name}...")
    print(f"[MRCONSO] Reading in chunks (file is large)...")

    col_names = [
        "cui","lat","ts","lui","stt","sui","ispref","aui",
        "saui","scui","sdui","sab","tty","code","str",
        "srl","suppress","cvf","empty"
    ]

    records = []
    chunk_size = 500_000

    chunks = pd.read_csv(
        path, sep="|", header=None, index_col=False,
        names=col_names, chunksize=chunk_size,
        low_memory=False, 
        dtype=str,
    )

    for chunk in tqdm(chunks, desc="  Parsing MRCONSO"):
        # Filter 1: English only
        chunk = chunk[chunk["lat"] == "ENG"]

        # Filter 2: not suppressed
        chunk = chunk[chunk["suppress"] == "N"]

        # Filter 3: trusted source vocabularies
        chunk = chunk[chunk["sab"].isin(KEEP_SOURCES_CONCEPTS)]

        # Filter 4: only relevant CUIs (if provided)
        if relevant_cuis is not None:
            chunk = chunk[chunk["cui"].isin(relevant_cuis)]

        if len(chunk) == 0:
            continue

        records.append(chunk[["cui","str","sab","tty","code"]].copy())

    if not records:
        print("[MRCONSO] WARNING: No records found after filtering")
        return pd.DataFrame(columns=["cui","name","sab","tty","code"])

    df = pd.concat(records, ignore_index=True)
    df = df.rename(columns={"str": "name"})
    print(f"[MRCONSO] Total filtered rows: {len(df):,}")

    # ── Pick one preferred name per CUI ──────────────────────────────────────
    # Priority: RXNORM > SNOMEDCT_US > MSH > NCI > others
    # Within source: prefer TTY = IN (ingredient) for drugs,
    #                PT (preferred term) for others
    source_priority = {
        "RXNORM"      : 1,
        "SNOMEDCT_US" : 2,
        "MSH"         : 3,
        "NCI"         : 4,
        "GO"          : 5,
        "OMIM"        : 6,
        "DRUGBANK"    : 7,
    }
    tty_priority = {
        "IN"  : 1,   # ingredient (RxNorm drug names)
        "PT"  : 2,   # preferred term
        "PEP" : 3,   # preferred entry point
        "ET"  : 4,   # entry term
        "SY"  : 5,   # synonym
    }

    df["src_rank"] = df["sab"].map(source_priority).fillna(99).astype(int)
    df["tty_rank"] = df["tty"].map(tty_priority).fillna(99).astype(int)

    # Sort by priority and keep first (best) name per CUI
    df = df.sort_values(["cui","src_rank","tty_rank"])
    preferred = df.drop_duplicates(subset=["cui"], keep="first")
    preferred = preferred[["cui","name","sab","tty","code"]].copy()

    print(f"[MRCONSO] Unique CUIs with preferred name: {len(preferred):,}")
    print(f"[MRCONSO] Name source distribution:")
    print(preferred["sab"].value_counts().head(10).to_string())

    return preferred


# Step 3: Parse MRREL — get edges between concepts

def parse_mrrel(
    path         : Path = None,
    relevant_cuis: set  = None,
) -> pd.DataFrame:
    
    #Reads MRREL.RRF and returns edges between relevant CUIs.

   
    path = path or RAW_FILES["mrrel"]
    print(f"\n[MRREL] Loading relations from {path.name}...")
    print(f"[MRREL] Reading in chunks (file is large)...")

    col_names = [
        "cui1","aui1","stype1","rel","cui2","aui2","stype2",
        "rela","rui","srui","sab","sl","rg","dir",
        "suppress","cvf","empty"
    ]

    records = []
    chunk_size = 500_000

    chunks = pd.read_csv(
        path, sep="|", header=None, index_col=False,
        names=col_names, chunksize=chunk_size,
        low_memory=False,   
        dtype=str,
    )

    for chunk in tqdm(chunks, desc="  Parsing MRREL"):
        # Filter 1: not suppressed
        chunk = chunk[chunk["suppress"].fillna("N") == "N"]

        # Filter 2: DISABLED. 
        # Allow AUI/CODE level relations to pass through to avoid dropping 
        # valid edges from RxNorm, NCI, etc. Deduplication happens later.
        # chunk = chunk[
        #     (chunk["stype1"] == "CUI") & (chunk["stype2"] == "CUI")
        # ]

        # Filter 3: trusted source vocabularies
        chunk = chunk[chunk["sab"].isin(KEEP_SOURCES_RELATIONS)]

        # Filter 4: only named RELA relations we care about
        chunk = chunk[chunk["rela"].isin(RELATION_MAP.keys())]

        # Filter 5: both CUIs must be relevant
        if relevant_cuis is not None:
            chunk = chunk[
                chunk["cui1"].isin(relevant_cuis) &
                chunk["cui2"].isin(relevant_cuis)
            ]

        if len(chunk) == 0:
            continue

        records.append(
            chunk[["cui1","cui2","rela","sab"]].copy()
        )

    if not records:
        print("[MRREL] WARNING: No relations found after filtering")
        return pd.DataFrame(columns=["cui1","cui2","relation","rela_original","sab"])

    df = pd.concat(records, ignore_index=True)

    # Map RELA to our canonical relation names
    df["relation"]      = df["rela"].map(RELATION_MAP)
    df["rela_original"] = df["rela"]

    df = df[["cui1","cui2","relation","rela_original","sab"]].copy()
    df = df.drop_duplicates(subset=["cui1","cui2","relation"])

    print(f"[MRREL] Total unique relations: {len(df):,}")
    print(f"[MRREL] Relation distribution:")
    print(df["relation"].value_counts().head(15).to_string())

    return df



# Step 4: Parse MRDEF — get definitions (optional, for text features)


def parse_mrdef(
    path         : Path = None,
    relevant_cuis: set  = None,
) -> pd.DataFrame:
   
    #Reads MRDEF.RRF and returns one definition per CUI.
   
    path = path or RAW_FILES["mrdef"]
    print(f"\n[MRDEF] Loading definitions from {path.name}...")

    col_names = [
        "cui","aui","atui","satui","sab","def","suppress","cvf","empty"
    ]

    df = pd.read_csv(
        path, sep="|", header=None, index_col=False,
        names=col_names,
        low_memory=False,  
        dtype=str,
    )

    # Filter suppressed and to relevant CUIs
    df = df[df["suppress"] == "N"]
    if relevant_cuis is not None:
        df = df[df["cui"].isin(relevant_cuis)]

    # Keep only trusted sources
    df = df[df["sab"].isin(KEEP_SOURCES_CONCEPTS)]

    # Priority: MSH > NCI > SNOMEDCT_US
    src_priority = {"MSH": 1, "NCI": 2, "SNOMEDCT_US": 3}
    df["src_rank"] = df["sab"].map(src_priority).fillna(99).astype(int)
    df = df.sort_values(["cui","src_rank"])
    df = df.drop_duplicates(subset=["cui"], keep="first")

    df = df[["cui","def","sab"]].rename(columns={"def": "definition"})
    print(f"[MRDEF] Definitions for {len(df):,} CUIs")

    return df



# Master parse function — runs all 4 in the right order

def parse_umls_all() -> tuple:
    """
    Runs all UMLS parsers in the correct order:
        1. MRSTY   → get relevant CUIs (semantic type filter)
        2. MRCONSO → get preferred names for those CUIs
        3. MRREL   → get edges between those CUIs
        4. MRDEF   → get definitions for those CUIs (optional)

    Returns:
        sty_df  : CUI -> semantic type (and node_type)
        name_df : CUI -> preferred name
        rel_df  : edges (cui1, cui2, relation, rela_original, sab)
        def_df  : CUI -> definition
    """
    print("=" * 60)
    print("Parsing UMLS Metathesaurus")
    print("=" * 60)

    # Step 1: semantic types — defines which CUIs enter the graph
    sty_df = parse_mrsty()
    relevant_cuis = set(sty_df["cui"].tolist())
    print(f"\n[UMLS] Working set: {len(relevant_cuis):,} relevant CUIs")

    # Step 2: preferred names for those CUIs
    name_df = parse_mrconso(relevant_cuis=relevant_cuis)

    # Step 3: edges between those CUIs
    rel_df = parse_mrrel(relevant_cuis=relevant_cuis)

    # Step 4: definitions (optional — used for text features later)
    def_df = parse_mrdef(relevant_cuis=relevant_cuis)


    # Resolve primary node type per CUI (drug > protein > gene > disease > ...)
    sty_df["type_rank"] = sty_df["node_type"].map(NODE_TYPE_PRIORITY).fillna(99)
    sty_df = sty_df.sort_values(["cui","type_rank"])
    primary_type = sty_df.drop_duplicates(subset=["cui"], keep="first")

    print(f"\n{'='*60}")
    print(f"UMLS PARSE SUMMARY")
    print(f"{'='*60}")
    print(f"  Relevant CUIs (nodes) : {len(relevant_cuis):,}")
    print(f"  CUIs with names       : {len(name_df):,}")
    print(f"  Relations (edges)     : {len(rel_df):,}")
    print(f"  CUIs with definitions : {len(def_df):,}")
    print(f"\n  Node types (primary):")
    print(primary_type["node_type"].value_counts().to_string())

    return sty_df, name_df, rel_df, def_df


if __name__ == "__main__":
    sty_df, name_df, rel_df, def_df = parse_umls_all()
