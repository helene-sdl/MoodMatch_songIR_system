import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util

CORPUS_PATH = "data/processed/corpus.json"
EMBEDDINGS_PATH = "data/processed/embeddings.npy"

with open("evaluation/queries.json") as f:
    queries = json.load(f)
print("Loading corpus from disk...")
with open(CORPUS_PATH) as f:
    corpus = json.load(f)
print(f"Loaded {len(corpus)} documents")

corpus = corpus[:250]  #test locally with a smaller subset to speed up embedding generation and search

model = SentenceTransformer("all-MiniLM-L6-v2")

if os.path.exists(EMBEDDINGS_PATH):
    print("Loading embeddings from disk...")
    doc_embeddings = np.load(EMBEDDINGS_PATH)
    print(f"Loaded embeddings with shape: {doc_embeddings.shape}")
else:
    print("Generating embeddings (this will take a while)...")
    doc_texts = [doc["lyrics"] for doc in corpus]
    doc_embeddings = model.encode(doc_texts, show_progress_bar=True)
    os.makedirs("data/processed", exist_ok=True)
    np.save(EMBEDDINGS_PATH, doc_embeddings)
    print(f"Embeddings saved to {EMBEDDINGS_PATH}")

#Search function
def search(query, top_k=5):
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_indices = scores.topk(top_k).indices

    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25} {'Genre'}")
    print("-" * 90)
    for rank, idx in enumerate(top_indices, 1):
        doc = corpus[idx]
        print(f"{rank:<6} {scores[idx].item():.4f}  {doc['title'][:34]:<35} {doc['artist'][:24]:<25} {doc['genre']}")


print("SentenceTransformer Search Results:")

for q in queries:
    search(q)