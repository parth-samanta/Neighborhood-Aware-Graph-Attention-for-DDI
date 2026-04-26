import pandas as pd

def extract_relation_embeddings():
    print("Scanning TransE Relation Embeddings (medical_kg_relations.vec)...")
    rel_vectors = []
    
    try:
       
        with open('medical_kg_relations.vec', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                
                # Skip header line
                if len(parts) < 10: 
                    continue
                    
                rel_name = parts[0]
                vector = [float(x) for x in parts[1:]]
                
                row = {'relation': rel_name}
                for i, val in enumerate(vector):
                    # Name rel_dim_ to easily distinguish from node embeddings
                    row[f'rel_dim_{i+1}'] = val
                rel_vectors.append(row)
                
    except FileNotFoundError:
        print("Error: Could not find medical_kg_relations.vec")
        return

    df_relations = pd.DataFrame(rel_vectors)
    output_file = 'transe_relation_embeddings.csv'
    df_relations.to_csv(output_file, index=False)
    
    print("="*60)
    print(" RELATION EMBEDDINGS EXTRACTED ")
    print("="*60)
    print(f"Extracted {len(df_relations)} unique relation types.")
    print(f"Vectors have {len(df_relations.columns) - 1} dimensions.")
    print(f"Saved cleanly to: {output_file}")
    print("="*60 + "\n")

if __name__ == '__main__':
    extract_relation_embeddings()