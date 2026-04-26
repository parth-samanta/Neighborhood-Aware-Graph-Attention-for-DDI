# -*- coding: utf-8 -*-
import os
import time
import json
import urllib.error
import urllib.request
import threading
import http.client
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio import Entrez, Medline


# CONFIGURATION

Entrez.email = os.getenv("ENTREZ_EMAIL")
Entrez.api_key = os.getenv("ENTREZ_API_KEY")

INPUT_NODES_FILE = "unique_nodes.txt"
OUTPUT_ABSTRACTS_FILE = "hybrid_abstracts.jsonl"

COMMON_CACHE_FILE = "cache_common_drugs.txt"
RARE_CACHE_FILE = "cache_rare_drugs.txt"
PROGRESS_FILE = "cache_completed_drugs.txt"  

FREQUENCY_THRESHOLD = 5000


MAX_WORKERS = 3        
BATCH_SIZE = 1000      
BASE_SLEEP = 0.35   

# Lock to prevent race conditions when multiple threads write to the same file
file_lock = threading.Lock()


# HELPER: EXPONENTIAL BACKOFF

def fetch_with_retry(func, max_retries=6, base_delay=3, *args, **kwargs):
    """Wraps API calls with exponential backoff. Handles HTTP 400, IncompleteRead, etc."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except urllib.error.HTTPError as e:
            # HTTP 400 from NCBI = server overloaded / history key expired ? back off hard
            delay = base_delay * (2 ** attempt)
            print(f"Network error: {e}. Retrying in {delay} seconds (Attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
        except (urllib.error.URLError, http.client.IncompleteRead, ConnectionResetError, Exception) as e:
            delay = base_delay * (2 ** attempt)
            print(f"Network error: {e}. Retrying in {delay} seconds (Attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)
    raise Exception(f"Max retries exceeded for function {func.__name__}")


# PHASE 1: PROFILER & ROUTING

def categorize_drugs(input_file):
    if os.path.exists(COMMON_CACHE_FILE) and os.path.exists(RARE_CACHE_FILE):
        print("Found cached drug categories. Skipping profiling phase...")
        with open(COMMON_CACHE_FILE, 'r') as f:
            common = [line.strip() for line in f if line.strip()]
        with open(RARE_CACHE_FILE, 'r') as f:
            rare = [line.strip() for line in f if line.strip()]
        return common, rare

    print(f"Phase 1: Profiling drug frequencies on PubMed (Threshold: {FREQUENCY_THRESHOLD})...")
    with open(input_file, 'r', encoding='utf-8') as f:
        all_drugs = [line.strip() for line in f if line.strip()]

    common, rare = [], []

    for i, drug in enumerate(all_drugs):
        try:
            handle = fetch_with_retry(Entrez.esearch, db="pubmed", term=f'"{drug}"[tiab]', retmax=0)
            record = Entrez.read(handle)
            handle.close()

            count = int(record["Count"])
            if count >= FREQUENCY_THRESHOLD:
                common.append(drug)
            else:
                rare.append(drug)

            if i % 100 == 0 and i > 0:
                print(f" -> Profiled {i}/{len(all_drugs)} drugs...")

            time.sleep(BASE_SLEEP)

        except Exception as e:
            print(f"Critical error profiling {drug}: {e}. Defaulting to Rare.")
            rare.append(drug)

    with open(COMMON_CACHE_FILE, 'w') as f:
        f.write('\n'.join(common))
    with open(RARE_CACHE_FILE, 'w') as f:
        f.write('\n'.join(rare))

    return common, rare


# PHASE 2: HYBRID EXTRACTION

def load_completed_drugs():
    """Load set of already-completed drugs for resume support."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def mark_drug_complete(drug):
    """Atomically append a drug name to the progress file."""
    with file_lock:
        with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
            f.write(drug + "\n")

def process_single_drug(drug, query_type, output_file):
    
    # Worker function: handles the full search + sequential download for ONE drug.

    drug_total = 0

    if query_type == "signal":
        full_query = (
            f'("{drug}"[tiab]) AND '
            f'(interaction[tiab] OR interactions[tiab] OR '
            f'pharmacology[tiab] OR mechanism[tiab])'
        )
        MAX_RECORDS = 10_000   # Hard cap for hub drugs
    else:
        full_query = f'"{drug}"[tiab]'
        MAX_RECORDS = None     # No cap for tail drugs

    try:
        #  Step 1: Search (get WebEnv + count)
        handle = fetch_with_retry(
            Entrez.esearch,
            db="pubmed",
            term=full_query,
            usehistory="y"
        )
        search_record = Entrez.read(handle)
        handle.close()

        count = int(search_record["Count"])
        if count == 0:
            return 0

        if MAX_RECORDS:
            count = min(count, MAX_RECORDS)

        webenv = search_record["WebEnv"]
        query_key = search_record["QueryKey"]

        inter_batch_sleep = BASE_SLEEP if count < 5_000 else 0.5

        print(f" -> Thread started: '{drug}' ({count:,} records)")

        time.sleep(1.0)

        # Step 2: Sequential download in batches 
        for start in range(0, count, BATCH_SIZE):
            fetch_handle = fetch_with_retry(
                Entrez.efetch,
                db="pubmed",
                retstart=start,
                retmax=BATCH_SIZE,
                webenv=webenv,
                query_key=query_key,
                rettype="medline",
                retmode="text"
            )

            # Parse while handle is open, then close immediately
            batch_records = list(Medline.parse(fetch_handle))
            fetch_handle.close()

            # Step 3: Write with thread-safe lock
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for rec in batch_records:
                        abstract = rec.get("AB", "").strip()
                        if abstract:
                            record_data = {
                                "pmid": rec.get("PMID", "Unknown"),
                                "query_drug": drug,
                                "abstract": abstract
                            }
                            f.write(json.dumps(record_data) + "\n")
                            drug_total += 1

            time.sleep(inter_batch_sleep)

        # Mark as done so can resume if the script is interrupted
        mark_drug_complete(drug)

    except Exception as e:
        print(f"\n[!] Failed to process drug '{drug}': {e}")

    return drug_total


def fetch_abstracts(drug_list, query_type, output_file, completed_drugs):
    total_fetched = 0
    mode_name = "STRICT SIGNAL" if query_type == "signal" else "WIDE NET"

    # Skip already-completed drugs (resume support)
    remaining = [d for d in drug_list if d not in completed_drugs]
    skipped = len(drug_list) - len(remaining)
    if skipped:
        print(f" -> Resuming: skipping {skipped} already-completed drugs.")

    print(f"\nStarting {mode_name} extraction for {len(remaining)} drugs...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_drug, drug, query_type, output_file): drug
            for drug in remaining
        }

        for future in as_completed(futures):
            drug_name = futures[future]
            try:
                result = future.result()
                total_fetched += result
            except Exception as e:
                print(f"Thread error on '{drug_name}': {e}")

    print(f"\n -> Phase Complete! Total extracted this run: {total_fetched:,}")
    return total_fetched

# MAIN EXECUTION

if __name__ == "__main__":
    if not os.path.exists(INPUT_NODES_FILE):
        print(f"STOP: Could not find {INPUT_NODES_FILE}. Please ensure the file is in this directory.")
        exit()

    common_drugs, rare_drugs = categorize_drugs(INPUT_NODES_FILE)

    print("\n" + "="*45)
    print(" PROFILING REPORT ")
    print("="*45)
    print(f"Current Threshold: > {FREQUENCY_THRESHOLD:,} papers = Hub")
    print(f"Hubs (Common Drugs): {len(common_drugs):,} drugs -> Strict Signal filter")
    print(f"Tails (Rare Drugs):  {len(rare_drugs):,} drugs -> Extract all instances")
    print("="*45)

    # Load progress
    completed_drugs = load_completed_drugs()
    if completed_drugs:
        print(f"\n[Resume mode] {len(completed_drugs)} drugs already completed.")

    user_input = input("\nDo you want to proceed with extraction? (y/n): ")
    if user_input.lower() != 'y':
        print("Extraction aborted.")
        exit()

    # Only clear output file if starting completely fresh (no progress)
    if not completed_drugs and os.path.exists(OUTPUT_ABSTRACTS_FILE):
        os.remove(OUTPUT_ABSTRACTS_FILE)

    print("\n--- PHASE 2A: HUB DRUGS ---")
    common_count = fetch_abstracts(
        common_drugs, query_type="signal",
        output_file=OUTPUT_ABSTRACTS_FILE,
        completed_drugs=completed_drugs
    )

    print("\n--- PHASE 2B: TAIL DRUGS ---")
    rare_count = fetch_abstracts(
        rare_drugs, query_type="all",
        output_file=OUTPUT_ABSTRACTS_FILE,
        completed_drugs=completed_drugs
    )

    print("\n" + "="*50)
    print(" HYBRID EXTRACTION COMPLETE ")
    print("="*50)
    print(f"Total Abstracts from Common Drugs: {common_count:,}")
    print(f"Total Abstracts from Rare Drugs:   {rare_count:,}")
    print(f"Total Dataset Size:                {common_count + rare_count:,} abstracts")
    print(f"Saved to:                          {OUTPUT_ABSTRACTS_FILE}")
    print("="*50)