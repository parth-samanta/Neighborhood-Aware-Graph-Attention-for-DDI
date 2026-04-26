# -*- coding: utf-8 -*-

#Optimized Corpus builder + FastText trainer for DDI pipeline

import json
import re
import os
import glob
import logging
import multiprocessing as mp
from functools import partial
from flashtext import KeywordProcessor
from tqdm import tqdm
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
import nltk

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


# CONSTANTS

# Regex sentence splitter
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_CLEAN_PATTERN = re.compile(r'[^a-z0-9_ ]')
_SPACE_COLLAPSE = re.compile(r'\s+')

# NLTK SETUP

def ensure_nltk_punkt():
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[1], quiet=True)

# ALIGNMENT PRE-FLIGHT CHECK

def check_node_id_alignment(golden_drugs: list[str]) -> None:
    """
    Ensure every KG node ID contains only [a-z0-9_] so Node2Vec and
    FastText tokens are byte-for-byte identical
    """
    bad = [d for d in golden_drugs if re.search(r'[^a-z0-9_]', d)]
    if bad:
        log.error(f"ALIGNMENT VIOLATION: {len(bad)} node IDs will be misaligned:")
        for d in bad[:30]:
            log.error(f"  '{d}'")
        raise ValueError("Fix node ID characters before proceeding. See log above.")
    log.info(f"Alignment pre-flight: all {len(golden_drugs):,} node IDs are clean.")


# SENTENCE SPLITTING

def split_sentences_fast(text: str) -> list[str]:
    # Regex-based sentence splitter
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

# CLEANING

def _clean_sentence(text: str) -> str:
    """
    Called after FlashText anchoring so drug underscore tokens are safe.
    Hyphens in non-drug text become spaces; everything outside [a-z0-9_ ] is stripped.
    """
    text = text.lower()
    text = text.replace('-', ' ')        # hyphen -> space (drugs already underscored)
    text = _CLEAN_PATTERN.sub('', text)
    text = _SPACE_COLLAPSE.sub(' ', text).strip()
    return text



# WORKER INITIALIZER  (runs once per subprocess)

_worker_kp: KeywordProcessor = None

def _init_worker(golden_drugs: list[str]) -> None:
    global _worker_kp
    _worker_kp = _build_kp_local(golden_drugs)


def _build_kp_local(golden_drugs: list[str]) -> KeywordProcessor:
    kp = KeywordProcessor(case_sensitive=False)
    for drug in golden_drugs:
        canonical     = drug
        surface_space = drug.replace('_', ' ')
        surface_dash  = drug.replace('_', '-')
        kp.add_keyword(canonical,     canonical)
        kp.add_keyword(surface_space, canonical)
        kp.add_keyword(surface_dash,  canonical)
    return kp


# WORKER FUNCTION  (called per chunk of lines)

def _process_chunk(chunk: list[str], text_key: str) -> list[str]:
    """
    Processes a list of raw JSONL lines. Returns cleaned sentence strings.
    Runs in a subprocess - uses the module-level _worker_kp.
    """
    out = []
    for line in chunk:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw_text = data.get(text_key, '').strip()
        if not raw_text:
            continue
        for sent in split_sentences_fast(raw_text):
            anchored = _worker_kp.replace_keywords(sent)
            clean    = _clean_sentence(anchored)
            if clean:
                out.append(clean)
    return out

# PHASE 1: BUILD FLASHTEXT

def build_keyword_processor(golden_drugs: list[str]) -> KeywordProcessor:
    log.info("Building FlashText Keyword Processor (main process)...")
    kp = _build_kp_local(golden_drugs)
    log.info(f"Processor built for {len(golden_drugs):,} entities "
             f"({len(golden_drugs)*3:,} surface forms).")
    return kp

# PHASE 2: PARALLEL CORPUS BUILDER

def process_jsonl_parallel(
    input_pattern: str,
    output_txt: str,
    golden_drugs: list[str],
    text_key: str = 'abstract',
    n_workers: int = 20,
    chunk_size: int = 2000,
) -> int:

    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        raise FileNotFoundError(f"No files matched: {input_pattern}")
    log.info(f"Found {len(input_files)} file(s). Launching {n_workers} workers...")

    # Read all lines upfront
    all_lines = []
    for fp in input_files:
        with open(fp, 'r', encoding='utf-8') as f:
            all_lines.extend(f.readlines())
    log.info(f"Loaded {len(all_lines):,} raw lines into memory.")

    # Split into chunks
    chunks = [all_lines[i:i+chunk_size] for i in range(0, len(all_lines), chunk_size)]
    log.info(f"Split into {len(chunks):,} chunks of {chunk_size} lines each.")

    total_sentences = 0
    worker_fn = partial(_process_chunk, text_key=text_key)

    with open(output_txt, 'w', encoding='utf-8') as outfile:
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(golden_drugs,)
        ) as pool:
            for result_batch in tqdm(
                pool.imap(worker_fn, chunks, chunksize=4),
                total=len(chunks),
                desc="Building corpus"
            ):
                for sentence in result_batch:
                    outfile.write(sentence + '\n')
                    total_sentences += 1

    log.info(f"Corpus written: {output_txt} | {total_sentences:,} sentences total.")
    return total_sentences

# PHASE 2b: SENTINEL INJECTION

def inject_sentinel_sentence(corpus_file: str, golden_drugs: list[str]) -> None:
    """
    Appends one line containing all Golden Drug canonical tokens.
    Guarantees every drug is in FastText's explicit vocabulary even if it
    never appeared in any abstract.
    """
    sentinel = ' '.join(golden_drugs)
    with open(corpus_file, 'a', encoding='utf-8') as f:
        f.write(sentinel + '\n')
    log.info(f"Sentinel injected: {len(golden_drugs):,} drug tokens.")

# PHASE 3: TRAIN FASTTEXT

def train_fasttext(
    corpus_file: str,
    model_save_path: str,
    vector_size: int = 128,
    window: int = 7,
    epochs: int = 5,
    workers: int = 20,
    min_count: int = 1,
) -> FastText:
    
    # Trains FastText using the C-level LineSentence corpus reader.

    log.info("Starting FastText training via C-level LineSentence reader...")
    log.info(f"  vector_size={vector_size}, window={window}, "
             f"epochs={epochs}, workers={workers}")

    sentences = LineSentence(corpus_file)

    model = FastText(
        corpus_file=corpus_file,  
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,    # sip-gram                
        epochs=epochs,
    )

    model.save(model_save_path)
    log.info(f"Model saved: {model_save_path}")
    return model



# PHASE 4: VOCABULARY AUDIT

def audit_vocabulary(trained_model: FastText, golden_drugs: list[str]) -> dict:
    """
    Three-tier audit:
    - EXPLICIT    : in key_to_index (full trained embedding)
    - SUBWORD_ONLY: FastText can infer via n-gram composition (should be 0 after sentinel)
    - TRULY_MISSING: no vector at all (means cleaning stripped chars from node ID)
    """
    log.info("--- Vocabulary Audit ---")
    explicit_vocab = set(trained_model.wv.key_to_index.keys())
    golden_set     = set(golden_drugs)

    explicit     = golden_set & explicit_vocab
    not_explicit = golden_set - explicit_vocab

    subword_only   = set()
    truly_missing  = set()
    for drug in not_explicit:
        try:
            _ = trained_model.wv[drug]
            subword_only.add(drug)
        except KeyError:
            truly_missing.add(drug)

    log.info(f"Total explicit vocab size : {len(explicit_vocab):,}")
    log.info(f"Golden - EXPLICIT         : {len(explicit):,} / {len(golden_set):,}")
    log.info(f"Golden - SUBWORD only     : {len(subword_only):,}")
    log.info(f"Golden - TRULY MISSING    : {len(truly_missing):,}")

    if subword_only:
        log.warning("Subword-only (check sentinel survived cleaning):")
        for d in list(subword_only)[:20]: log.warning(f"  {d}")
    if truly_missing:
        log.error("Truly missing (cleaning regex stripped node ID chars):")
        for d in list(truly_missing)[:20]: log.error(f"  {d}")
    if not subword_only and not truly_missing:
        log.info("PERFECT: 100% of Golden Drugs are in the EXPLICIT vocabulary.")

    return {"explicit": explicit, "subword_only": subword_only, "truly_missing": truly_missing}

# MAIN

if __name__ == "__main__":

    # ---- CONFIGURATION ----
    INPUT_JSONL_PATTERN = "hybrid_abstracts.jsonl"
    OUTPUT_CORPUS       = "fasttext_training_corpus.txt"
    MODEL_SAVE_PATH     = "custom_ddi_fasttext.model"
    TEXT_KEY            = "abstract"
    VECTOR_SIZE         = 128
    WINDOW              = 7
    EPOCHS              = 5       
    N_WORKERS           = 20      
    FT_WORKERS          = 20      

    # 1. On-the-Fly Normalization for Raw Drug Names
    import re
    
    def normalize_to_kg_node(raw_name: str) -> str:
        """Applies Phase 1 rules: lowercase, space/dash to underscore, strip punctuation"""
        name = raw_name.lower()
        name = name.replace(" ", "_").replace("-", "_")
        name = re.sub(r'[^a-z0-9_]', '', name) # Strip periods, commas, parentheses
        name = re.sub(r'_+', '_', name)        # Collapse duplicate underscores
        return name.strip('_')                 # Remove leading/trailing underscores

    # 2. Load and normalize 3,072 golden drugs
    with open("unique_nodes.txt", "r", encoding="utf-8") as f:
        golden_drugs = [normalize_to_kg_node(line.strip()) for line in f if line.strip()]

    # ---- EXECUTE ----
    ensure_nltk_punkt()

    # Step 0: Alignment guard
    check_node_id_alignment(golden_drugs)

    # Step 1: Parallel corpus build (replaces single-threaded stream)
    process_jsonl_parallel(
        input_pattern=INPUT_JSONL_PATTERN,
        output_txt=OUTPUT_CORPUS,
        golden_drugs=golden_drugs,
        text_key=TEXT_KEY,
        n_workers=N_WORKERS,
        chunk_size=2000,
    )

    # Step 2: Sentinel injection
    inject_sentinel_sentence(OUTPUT_CORPUS, golden_drugs)

    # Step 3: Train (C-level reader)
    model = train_fasttext(
        corpus_file=OUTPUT_CORPUS,
        model_save_path=MODEL_SAVE_PATH,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        epochs=EPOCHS,
        workers=FT_WORKERS,
    )

    # Step 4: Audit
    audit_results = audit_vocabulary(model, golden_drugs)