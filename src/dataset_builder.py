import pandas as pd
import random
import re

def build_multimodal_dataset():
    # 1. Load dataset
    print("Loading exact FastText keys from golden_drugs_nlp_embeddings.csv...")
    try:
        # Node_id column to get the exact string names
        df_embed = pd.read_csv('golden_drugs_nlp_embeddings.csv', usecols=['node_id'])
        valid_embedding_keys = set(df_embed['node_id'].astype(str))
        print(f" -> Loaded {len(valid_embedding_keys):,} exact NLP embedding keys.")
    except FileNotFoundError:
        print("Error: Could not find golden_drugs_nlp_embeddings.csv")
        return

    # 2. Load Knowledge Graph mapping
    print("Loading Knowledge Graph mapping (nodes.tsv)...")
    try:
        df_kg = pd.read_csv('nodes.tsv', sep='\t')
        kg_rxnorm = df_kg[df_kg['SAB'] == 'RXNORM'].dropna(subset=['code', 'preferred_name'])
        
        # Build dictionary: RxNorm Code -> EXACT FastText Key
        code_to_exact_name = {}
        for _, row in kg_rxnorm.iterrows():
            code = str(row['code'])
            raw_name = str(row['preferred_name'])
            
            # Create few variations to bridge the gap between KG and FastText keys
            variations = [
                raw_name,                                  # Exact match
                raw_name.lower(),                          # Lowercase
                re.sub(r'\s+', '_', raw_name.lower()),     # Spaces to underscores
                re.sub(r'[\s\-]+', '_', raw_name.lower())  # Spaces and hyphens to underscores
            ]
            
            # Check which variation perfectly matches embeddings file
            for var in variations:
                if var in valid_embedding_keys:
                    code_to_exact_name[code] = var
                    break
                
        valid_codes_list = list(code_to_exact_name.keys())
        print(f" -> Successfully anchored {len(valid_codes_list):,} RxNorm nodes to exact embedding keys.")
    except FileNotFoundError:
        print("Error: Could not find nodes.tsv")
        return

    # 3. Process Positive Edges (Label: 1)
    print("Extracting positive interactions (kg_positive_edges_rxnorm.csv)...")
    try:
        df_pos = pd.read_csv('kg_positive_edges_rxnorm.csv')
        positive_pairs = set()
        positive_records = []
        
        for _, row in df_pos.iterrows():
            c1, c2 = str(row['rxnorm_a']), str(row['rxnorm_b'])
            
            # Skip if either drug didn't successfully map to an exact embedding key, or if it's a self-loop
            if c1 not in code_to_exact_name or c2 not in code_to_exact_name or c1 == c2: 
                continue 
                
            pair = tuple(sorted([c1, c2]))
            
            if pair not in positive_pairs:
                positive_pairs.add(pair)
                positive_records.append({
                    'drug_a_name': code_to_exact_name[pair[0]], # The EXACT NLP Token from your CSV
                    'drug_a_rxnorm': pair[0],                   # The Graph ID
                    'drug_b_name': code_to_exact_name[pair[1]], # The EXACT NLP Token from your CSV
                    'drug_b_rxnorm': pair[1],                   # The Graph ID
                    'interaction': 1                            # Target Label
                })
        print(f" -> Found {len(positive_pairs):,} fully playable positive pairs.")
    except FileNotFoundError:
        print("Error: Could not find kg_positive_edges_rxnorm.csv")
        return

    # 4. Generate Balanced Negative Samples (Label: 0)
    print("Generating perfectly balanced negative samples (Label: 0)...")
    target_negatives = len(positive_pairs) 
    negative_pairs = set()
    negative_records = []
    
    while len(negative_pairs) < target_negatives:
        # Randomly sample two valid RxNorm codes from approved list
        d1, d2 = random.sample(valid_codes_list, 2)
        pair = tuple(sorted([d1, d2]))
        
        # Ensure it is NOT a known positive interaction, and NOT already generated
        if pair not in positive_pairs and pair not in negative_pairs:
            negative_pairs.add(pair)
            negative_records.append({
                'drug_a_name': code_to_exact_name[pair[0]],
                'drug_a_rxnorm': pair[0],
                'drug_b_name': code_to_exact_name[pair[1]],
                'drug_b_rxnorm': pair[1],
                'interaction': 0
            })
            
        if len(negative_pairs) % 100000 == 0:
            print(f"    ... Generated {len(negative_pairs):,} / {target_negatives:,} negatives")

    # 5. Assemble and Shuffle the Final Training Data
    print("Assembling final ML CSV...")
    df_final = pd.DataFrame(positive_records + negative_records)
    
    # Shuffle the dataset thoroughly
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_file = 'ddi_training_dataset.csv'
    df_final.to_csv(output_file, index=False)
    
    print("\n" + "="*55)
    print("✅ MULTIMODAL DDI TRAINING DATASET COMPLETE ✅")
    print("="*55)
    print(f"Total Rows:       {len(df_final):,}")
    print(f"Positive Samples: {len(positive_pairs):,} (Interaction = 1)")
    print(f"Negative Samples: {len(negative_pairs):,} (Interaction = 0)")
    print(f"Saved to:         {output_file}")
    print("="*55 + "\n")
    print("Sample Output:")
    print(df_final.head(5))

if __name__ == "__main__":
    build_multimodal_dataset()