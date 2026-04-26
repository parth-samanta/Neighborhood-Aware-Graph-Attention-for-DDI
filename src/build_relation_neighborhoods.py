import pandas as pd
import numpy as np
import random

def build_relation_neighborhood_table(sample_size=15):
    print("Loading your Golden Drugs...")
    try:
        df_drugs = pd.read_csv('drug_embeddings_fixed.csv')
        df_drugs = df_drugs.dropna(subset=['CUI'])
        valid_cuis = set(df_drugs['CUI'].unique())
        print(f" -> Loaded {len(valid_cuis):,} unique valid drug CUIs.")
    except FileNotFoundError:
        print("Error: Could not find drug_embeddings_fixed.csv")
        return

    print("Loading Knowledge Graph edges (edges.tsv)...")
    try:
        df_edges = pd.read_csv('edges.tsv', sep='\t')
        # Remove self-loops
        df_edges = df_edges[df_edges['CUI1'] != df_edges['CUI2']].copy()
        print(f" -> Loaded {len(df_edges):,} valid edges.")
    except FileNotFoundError:
        print("Error: Could not find edges.tsv")
        return
        
    print(f"Sampling up to {sample_size} neighbors AND relations per drug...\n")
    
    # Extract Source pairs (Drug -> Relation -> Neighbor)
    src_pairs = df_edges[df_edges['CUI1'].isin(valid_cuis)][['CUI1', 'CUI2', 'relation']]
    src_pairs = src_pairs.rename(columns={'CUI1': 'Drug_CUI', 'CUI2': 'Neighbor_CUI', 'relation': 'Relation'})
    
    # Extract Target pairs (Neighbor -> Relation -> Drug)
    tgt_pairs = df_edges[df_edges['CUI2'].isin(valid_cuis)][['CUI2', 'CUI1', 'relation']]
    tgt_pairs = tgt_pairs.rename(columns={'CUI2': 'Drug_CUI', 'CUI1': 'Neighbor_CUI', 'relation': 'Relation'})
    
    # Combine all valid neighbor relationships
    all_pairs = pd.concat([src_pairs, tgt_pairs], ignore_index=True).drop_duplicates()
    
    # Create a tuple of (Neighbor, Relation) for easy grouping
    all_pairs['neighbor_tuple'] = list(zip(all_pairs['Neighbor_CUI'], all_pairs['Relation']))
    
    # Group by Drug_CUI to get a list of all unique (Neighbor, Relation) pairs
    neighbor_dict = all_pairs.groupby('Drug_CUI')['neighbor_tuple'].apply(list).to_dict()
    
    neighborhood_records = []
    random.seed(42) # Keep random sampling reproducible
    
    for cui in valid_cuis:
        neighbors = neighbor_dict.get(cui, [])
        
        # Sample or Pad
        if len(neighbors) >= sample_size:
            sampled = random.sample(neighbors, sample_size)
        else:
            # Pad the rest of the 15 slots with empty tuples ('', '')
            sampled = neighbors + [('', '')] * (sample_size - len(neighbors))
            
        # Build the row dictionary
        row = {'CUI': cui}
        for i in range(sample_size):
            row[f'neighbor_{i+1}'] = sampled[i][0]
            row[f'relation_{i+1}'] = sampled[i][1] # <- Storing the edge type
            
        neighborhood_records.append(row)
        
    df_neighborhood = pd.DataFrame(neighborhood_records)
    
    # Order columns so it reads: neighbor_1, relation_1, neighbor_2, relation_2...
    cols = ['CUI']
    for i in range(1, sample_size + 1):
        cols.extend([f'neighbor_{i}', f'relation_{i}'])
    df_neighborhood = df_neighborhood[cols]
    
    output_file = 'relation_aware_neighborhood_15.csv'
    df_neighborhood.to_csv(output_file, index=False)
    
    print("="*60)
    print("RELATION-AWARE NEIGHBORHOOD TABLE GENERATED")
    print("="*60)
    print(f"Saved to: {output_file}")
    print("="*60 + "\n")
    print("Sample Output (Showing first 2 neighbors & relations):")
    print(df_neighborhood.iloc[:, :5].head(5))

if __name__ == '__main__':
    build_relation_neighborhood_table(sample_size=15)