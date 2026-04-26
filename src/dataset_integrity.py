import pandas as pd

def verify_full_multimodal_dataset():
    print("Loading datasets...")

    # 1. Load original training data
    try:
        df_train = pd.read_csv('ddi_training_dataset.csv')
        print(f" -> Loaded Original Training Pairs: {len(df_train):,}")
    except FileNotFoundError:
        print("Error: Could not find ddi_training_dataset.csv")
        return

    # 2. Load Valid Graph IDs (from drug_embeddings_fixed.csv)
    try:
        # Need the 'rxnorm' column to check existence
        df_graph = pd.read_csv('drug_embeddings_fixed.csv', usecols=['rxnorm'])
        
        # Convert to string and drop '.0' to prevent float/string mismatch errors
        valid_graph_ids = set(df_graph['rxnorm'].dropna().astype(str).str.replace('.0', '', regex=False))
        print(f" -> Loaded {len(valid_graph_ids):,} valid Graph IDs.")
    except FileNotFoundError:
        print("Error: Could not find drug_embeddings_fixed.csv")
        return

    # 3. Load Valid NLP Names (from golden_drugs_nlp_embeddings.csv)
    try:
        # nneed the 'node_id' column to check existence
        df_nlp = pd.read_csv('golden_drugs_nlp_embeddings.csv', usecols=['node_id'])
        valid_nlp_names = set(df_nlp['node_id'].dropna().astype(str))
        print(f" -> Loaded {len(valid_nlp_names):,} valid NLP text keys.")
    except FileNotFoundError:
        print("Error: Could not find golden_drugs_nlp_embeddings.csv")
        return

    print("\nVerifying training pairs across BOTH modalities...")
    
    # Format the columns in the training dataset for comparison
    train_rx_a = df_train['drug_a_rxnorm'].astype(str).str.replace('.0', '', regex=False)
    train_rx_b = df_train['drug_b_rxnorm'].astype(str).str.replace('.0', '', regex=False)
    train_name_a = df_train['drug_a_name'].astype(str)
    train_name_b = df_train['drug_b_name'].astype(str)

    # Check Graph validity (Both RxNorm IDs MUST be in the fixed graph embeddings)
    graph_valid_a = train_rx_a.isin(valid_graph_ids)
    graph_valid_b = train_rx_b.isin(valid_graph_ids)
    
    # Check NLP validity (Both string names MUST be in the NLP embeddings)
    nlp_valid_a = train_name_a.isin(valid_nlp_names)
    nlp_valid_b = train_name_b.isin(valid_nlp_names)
    
    # COMBINED MASK: A pair is strictly valid ONLY if all 4 conditions are True
    final_mask = graph_valid_a & graph_valid_b & nlp_valid_a & nlp_valid_b
    
    df_valid = df_train[final_mask].copy()
    
    # Calculate stats
    total_original = len(df_train)
    total_valid = len(df_valid)
    total_lost = total_original - total_valid
    
    print("\n" + "="*50)
    print("🔍 MULTIMODAL DATASET INTEGRITY REPORT 🔍")
    print("="*50)
    print(f"Original Dataset Size: {total_original:,} pairs")
    print(f"Fully Valid Pairs:     {total_valid:,} pairs")
    print(f"Dropped Pairs:         {total_lost:,} pairs")
    print("-" * 50)
    
    if total_valid > 0:
        print(f"Retention Rate:        {total_valid/total_original:.2%}")
        
        # Save fully verified dataset
        output_file = 'ddi_training_dataset_fully_verified.csv'
        df_valid.to_csv(output_file, index=False)
        print(f"\n Saved fully verified dataset to: {output_file}")
        print("   (Use this exact file to concatenate your arrays and train!)")
    else:
        print(" Critical Error: 0 valid pairs remain. Check your data alignments.")
    print("="*50 + "\n")

if __name__ == "__main__":
    verify_full_multimodal_dataset()