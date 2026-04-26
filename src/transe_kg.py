import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.utils import set_random_seed

set_random_seed(42)

if not os.path.exists("outputs"):
    os.makedirs("outputs")

# 1. Load graph and node metadata
with open("knowledge_graph/graph.pkl", "rb") as f:
    G = pickle.load(f)

# Clean out isolated nodes and self-loops (TransE can handle self-loops, but usually better without)
G.remove_edges_from(nx.selfloop_edges(G))
G.remove_nodes_from(list(nx.isolates(G)))

print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

nodes_df = pd.read_csv("knowledge_graph/nodes.tsv", sep="\t")
node_type = dict(zip(nodes_df["node_id"], nodes_df["node_type"]))
node_name = dict(zip(nodes_df["node_id"], nodes_df["preferred_name"]))

#  2. Extract Triples for TransE
triples = []
for head, tail, data in G.edges(data=True):
    # Try to extract a relation type from edge attributes. 
    # If graph doesn't have named relations, default to "related_to".
    rel = data.get("relation", data.get("type", data.get("label", "related_to")))
    triples.append([str(head), str(rel), str(tail)])

triples = np.array(triples)
tf = TriplesFactory.from_labeled_triples(triples)

# 3. Train TransE embeddings 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training TransE on {device.upper()}...")

#  3. Train TransE on 5060 Ti with Checkpointing 
tf_train, tf_test, _ = tf.split([0.8, 0.1, 0.1], random_state=42)
result = pipeline(
    training=tf_train,
    testing=tf_test, 
    model='TransE',
    model_kwargs=dict(embedding_dim=128),
    optimizer='Adam',
    optimizer_kwargs=dict(lr=0.005),
    training_kwargs=dict(
        num_epochs=250,        
        batch_size=8192,
        use_tqdm_batch=True,
        checkpoint_name='transe_checkpoint.pt',
        checkpoint_directory='outputs/checkpoints',
        checkpoint_frequency=5,
    ),
    evaluation_kwargs=None,
    evaluator_kwargs=None,
    random_seed=42,
    device=device,
)

model = result.model
result.save_to_directory("outputs/medical_kg_transe_model")

#  4. Save embeddings
# Get the mapping of string node IDs to PyKEEN's internal integer indices
entity_to_id = tf.entity_to_id
# Extract the actual tensor from the GPU and convert to numpy array
embeddings_tensor = model.entity_representations[0](indices=None).detach().cpu().numpy()

# Save in word2vec format
print(" Saving embeddings...")
with open("outputs/medical_kg_embeddings.vec", "w") as f:
    f.write(f"{len(entity_to_id)} {embeddings_tensor.shape[1]}\n")
    for ent, idx in entity_to_id.items():
        vec_str = " ".join(map(str, embeddings_tensor[idx]))
        f.write(f"{ent} {vec_str}\n")

# 5. Save Relation Embeddings
# Get the mapping for relations
relation_to_id = tf.relation_to_id
# Extract relation vectors (same dimension as entities in TransE)
rel_embeddings_tensor = model.relation_representations[0](indices=None).detach().cpu().numpy()

print(" Saving relation embeddings...")
with open("outputs/medical_kg_relations.vec", "w") as f:
    # Header: [Number of relations] [Vector dimension]
    f.write(f"{len(relation_to_id)} {rel_embeddings_tensor.shape[1]}\n")
    for rel, idx in relation_to_id.items():
        vec_str = " ".join(map(str, rel_embeddings_tensor[idx]))
        f.write(f"{rel} {vec_str}\n")

print(" Complete! Check outputs/ folder for your model and .vec file.")


