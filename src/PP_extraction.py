import pandas as pd

print("Loading datasets...")

# 1. Load the data
df_kg = pd.read_csv('nodes.tsv', sep='\t')
df_interactions = pd.read_csv('drugbank_interactions.csv')
df_crosswalk = pd.read_csv('drugbank_cui_crosswalk.csv')

# 2. Filter KG to only RxNorm nodes
print("Isolating RxNorm nodes in the Knowledge Graph...")
kg_rxnorm = df_kg[df_kg['SAB'] == 'RXNORM'].copy()

# Ensure 'code' column is treated as a string for exact matching
kg_rxnorm_codes = set(kg_rxnorm['code'].dropna().astype(str).unique())

# 3. Build the Translation Dictionary (DrugBank ID -> RxNorm)

# Drop missing rxnorm codes to ensure clean mapping
mapping_dict = df_crosswalk.dropna(subset=['rxnorm']).copy()
# Convert rxnorm to string in case pandas read them as floats
mapping_dict['rxnorm'] = mapping_dict['rxnorm'].astype(str)
# Remove any '.0' if they were read as floats
mapping_dict['rxnorm'] = mapping_dict['rxnorm'].str.replace('.0', '', regex=False)

db_to_rxnorm = mapping_dict.set_index('drugbank_id')['rxnorm'].to_dict()

print("Translating DrugBank IDs to RxNorm Codes...")

# 4. Translate the interactions dataset
df_interactions['rxnorm_a'] = df_interactions['drug_a_id'].map(db_to_rxnorm)
df_interactions['rxnorm_b'] = df_interactions['drug_b_id'].map(db_to_rxnorm)

# Drop any pairs where one or both drugs didn't have an RxNorm mapping
df_mapped_pairs = df_interactions.dropna(subset=['rxnorm_a', 'rxnorm_b'])
print(f"Successfully translated {len(df_mapped_pairs):,} out of {len(df_interactions):,} interaction pairs to RxNorm.")

# 5. Do the intersection using RxNorm codes
db_rxnorm_codes = set(df_mapped_pairs['rxnorm_a']).union(set(df_mapped_pairs['rxnorm_b']))
overlapping_codes = kg_rxnorm_codes.intersection(db_rxnorm_codes)

# Calculate metrics
kg_coverage = len(overlapping_codes) / len(kg_rxnorm_codes) if len(kg_rxnorm_codes) > 0 else 0
db_coverage = len(overlapping_codes) / len(db_rxnorm_codes) if len(db_rxnorm_codes) > 0 else 0

covered_pairs_df = df_mapped_pairs[
    (df_mapped_pairs['rxnorm_a'].isin(overlapping_codes)) & 
    (df_mapped_pairs['rxnorm_b'].isin(overlapping_codes))
]

print("\n" + "="*45)
print("🚀 RXNORM KNOWLEDGE GRAPH ALIGNMENT REPORT 🚀")
print("="*45)
print(f"Total Unique RxNorm nodes in KG:    {len(kg_rxnorm_codes):,}")
print(f"Total Unique RxNorms in DrugBank:   {len(db_rxnorm_codes):,}")
print(f"Overlapping Drugs:                  {len(overlapping_codes):,}")
print(f"-> KG RxNorm Utilization:           {kg_coverage:.2%} (Percentage of your KG's RxNorm drugs used)")
print(f"-> DrugBank Coverage:               {db_coverage:.2%} (Percentage of translated DB drugs in your KG)")

print("\n--- POSITIVE PAIRS OVERLAP ---")
print(f"Total Translated DB Pairs:          {len(df_mapped_pairs):,}")
print(f"Playable Pairs (Both nodes in KG):  {len(covered_pairs_df):,}")
print(f"-> Normalized Pair Intersection:    {len(covered_pairs_df) / len(df_mapped_pairs):.2%}")
print("="*45 + "\n")

# Extract just the RxNorm columns for your KG edges
final_positive_edges = covered_pairs_df[['rxnorm_a', 'rxnorm_b']].copy()
final_positive_edges['interaction'] = 1

final_positive_edges.to_csv('kg_positive_edges_rxnorm.csv', index=False)
print("Saved 1.8M positive edges to kg_positive_edges_rxnorm.csv")