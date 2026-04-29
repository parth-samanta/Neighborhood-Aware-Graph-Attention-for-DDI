
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA      = ROOT / "data"
RAW       = DATA / "raw"
PROCESSED = DATA / "processed"
EMBEDDINGS= DATA / "embeddings"

# ── Raw file paths ────────────────────────────────────────────────────────────
RAW_FILES = {
    "mrconso" : RAW / "umls" / "MRCONSO.RRF",
    "mrrel"   : RAW / "umls" / "MRREL.RRF",
    "mrsty"   : RAW / "umls" / "MRSTY.RRF",
    "mrdef"   : RAW / "umls" / "MRDEF.RRF",
}

# ── Processed output paths ────────────────────────────────────────────────────
PROCESSED_FILES = {
    "nodes"    : PROCESSED / "nodes.tsv",
    "edges"    : PROCESSED / "edges.tsv",
    "graph_pkl": PROCESSED / "graph.pkl",
    "graph_pyg": PROCESSED / "graph.pt",
    "cui_name" : PROCESSED / "cui_to_name.tsv",
}

# ── Semantic types to keep ────────────────────────────────────────────────────
KEEP_SEMANTIC_TYPES = {
    "T121",  # Pharmacologic Substance
    "T200",  # Clinical Drug
    "T195",  # Antibiotic
    "T109",  # Organic Chemical
    "T116",  # Amino Acid, Peptide, or Protein
    "T126",  # Enzyme
    "T192",  # Receptor
    "T028",  # Gene or Genome
    "T044",  # Molecular Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T191",  # Neoplastic Process
    "T038",  # Biologic Function
    "T043",  # Cell Function
}

# ── Source vocabularies for concept names ─────────────────────────────────────
KEEP_SOURCES_CONCEPTS = {
    "RXNORM",
    "SNOMEDCT_US",
    "MSH",
    "NCI",
    "GO",
    "OMIM",
    "DRUGBANK",
}

# ── Source vocabularies for relations ─────────────────────────────────────────
KEEP_SOURCES_RELATIONS = {
    "RXNORM",
    "SNOMEDCT_US",
    "MSH",
    "NCI",
    "GO",
    "DRUGBANK",
    "NDFRT",
    "MED-RT"
}

# ── MRREL RELA -> canonical relation name ─────────────────────────────────────
RELATION_MAP = {
    "has_mechanism_of_action"                      : "has_mechanism_of_action",
    "has_physiologic_effect"                       : "has_physiologic_effect",
    "may_inhibit_effect_of"                        : "inhibits",
    "inhibits"                                     : "inhibits",
    "has_ingredient"                               : "has_ingredient",
    "metabolized_by"                               : "metabolized_by",
    "substrate_of"                                 : "substrate_of",
    "has_target"                                   : "has_target",
    "may_treat"                                    : "may_treat",
    "may_prevent"                                  : "may_prevent",
    "may_diagnose"                                 : "may_diagnose",
    "has_contraindication"                         : "contraindicated_with",
    "has_tradename"                                : "has_tradename",
    "form_of"                                      : "form_of",
    "gene_associated_with_disease"                 : "associated_with_disease",
    "associated_with"                              : "associated_with",
    "interacts_with"                               : "interacts_with",
    "gene_product_plays_role_in_biological_process": "plays_role_in",
    "isa"                                          : "isa",
    "part_of"                                      : "part_of",
    "has_part"                                     : "has_part",
    "has_drug_interaction_with"                    : "interacts_with",   
    "may_interact_with"                            : "interacts_with",   
    "CI_with"                                      : "contraindicated_with",  
}

# ── TUI -> node type ──────────────────────────────────────────────────────────
SEMANTIC_TYPE_TO_NODE_TYPE = {
    "T121": "drug",
    "T200": "drug",
    "T195": "drug",
    "T109": "drug",
    "T116": "protein",
    "T126": "protein",
    "T192": "protein",
    "T028": "gene",
    "T044": "molecular_function",
    "T047": "disease",
    "T048": "disease",
    "T191": "disease",
    "T038": "biological_process",
    "T043": "biological_process",
}

# ── Node type priority (used when a CUI has multiple semantic types) ───────────
NODE_TYPE_PRIORITY = {
    "drug"              : 1,
    "protein"           : 2,
    "gene"              : 3,
    "disease"           : 4,
    "molecular_function": 5,
    "biological_process": 6,
}

