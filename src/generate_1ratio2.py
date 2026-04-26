import pandas as pd
import random

def balance_dataset_with_negatives():
    print("Loading datasets...")
    # 1. Load existing training data
    try:
        df_train = pd.read_csv('ddi_training_dataset_TRANSE_ready.csv')
    except FileNotFoundError:
        print("Error: Could not find ddi_training_dataset_TRANSE_ready.csv")
        return
        
    # 2. Load valid drugs from master embeddings
    try:
        df_drugs = pd.read_csv('master_drug_embeddings.csv', usecols=['rxnorm', 'name'])
        df_drugs['rxnorm'] = df_drugs['rxnorm'].astype(str).str.replace('.0', '', regex=False)
        # Create a list of valid drug dictionaries for rapid random sampling
        valid_drugs = df_drugs[['rxnorm', 'name']].drop_duplicates().to_dict('records')
        print(f" -> Loaded {len(valid_drugs):,} valid drugs for sampling.")
    except FileNotFoundError:
        print("Error: Could not find master_drug_embeddings.csv")
        return
    
    # 3. Calculate how many negatives we need
    num_positives = len(df_train[df_train['interaction'] == 1])
    num_current_negatives = len(df_train[df_train['interaction'] == 0])
    
    target_negatives = num_positives * 2
    negatives_to_add = target_negatives - num_current_negatives
    
    print("\n" + "-"*40)
    print(f"Current Positives: {num_positives:,}")
    print(f"Current Negatives: {num_current_negatives:,}")
    print(f"Target Negatives:  {target_negatives:,} (to reach 1:2 ratio)")
    print("-" * 40)
    
    if negatives_to_add <= 0:
        print("\nDataset already has a 1:2 ratio (or higher). No new samples needed.")
        return
        
    print(f"\nGenerating {negatives_to_add:,} new completely unique negative pairs...")
    
    # 4. Track existing pairs (undirected) to prevent creating a false negative
    # 'frozenset' used so that (Drug A, Drug B) is mathematically identical to (Drug B, Drug A)
    existing_pairs = set()
    for _, row in df_train.iterrows():
        rx_a = str(row['drug_a_rxnorm']).replace('.0', '')
        rx_b = str(row['drug_b_rxnorm']).replace('.0', '')
        existing_pairs.add(frozenset([rx_a, rx_b]))
        
    new_negatives = []
    generated_pairs = set() # To ensure we don't duplicate within the new batch
    
    random.seed(42) # Set seed for reproducibility
    
    # 5. Generate loop
    while len(new_negatives) < negatives_to_add:
        # Randomly sample 2 different drugs from our master pool
        drug_a, drug_b = random.sample(valid_drugs, 2)
        
        pair_set = frozenset([drug_a['rxnorm'], drug_b['rxnorm']])
        
        # Validate: Not in original data, not already generated, and not the exact same drug
        if pair_set not in existing_pairs and pair_set not in generated_pairs and len(pair_set) == 2:
            generated_pairs.add(pair_set)
            new_negatives.append({
                'drug_a_name': drug_a['name'],
                'drug_a_rxnorm': drug_a['rxnorm'],
                'drug_b_name': drug_b['name'],
                'drug_b_rxnorm': drug_b['rxnorm'],
                'interaction': 0
            })
            
            # Print a progress update occasionally
            if len(new_negatives) % 25000 == 0:
                print(f" -> Created {len(new_negatives):,} / {negatives_to_add:,}")
                
    # 6. Combine and Shuffle
    print("\nMerging and shuffling final dataset...")
    df_new_negatives = pd.DataFrame(new_negatives)
    
    df_final = pd.concat([df_train, df_new_negatives], ignore_index=True)
    
    # Shuffle the dataset so the new negatives aren't all stuck at the very bottom so that neural network gets diverse batches
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 7. Save
    output_file = 'ddi_training_dataset_balanced_1_2.csv'
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print(" BALANCED TRAINING DATASET GENERATED ")
    print("="*50)
    print(f"Total Positives: {len(df_final[df_final['interaction'] == 1]):,}")
    print(f"Total Negatives: {len(df_final[df_final['interaction'] == 0]):,}")
    print(f"Total Pairs:     {len(df_final):,}")
    print(f"Saved to:        {output_file}")
    print("="*50 + "\n")

if __name__ == '__main__':
    balance_dataset_with_negatives()