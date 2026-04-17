import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from retrieval_modes.preprocessing import preprocess

EMBEDDINGS_PATH  = "data/processed/st_embeddings.npy"
ST_CORPUS_PICKLE = "data/processed/st_corpus.pkl"
FAISS_INDEX_PATH = "data/processed/faiss_index.bin"

print("Loading corpus from pickle...")
with open(ST_CORPUS_PICKLE, "rb") as f:
    corpus = pickle.load(f)
print(f"Loaded {len(corpus)} documents")


print("Loading embeddings...")
doc_embeddings = np.load(EMBEDDINGS_PATH).astype("float32")  
print(f"Embeddings shape: {doc_embeddings.shape}")

if os.path.exists(FAISS_INDEX_PATH):
    print("Loading FAISS index from disk...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"Loaded index with {index.ntotal} vectors")
else:
    print("Building FAISS index...")
    dimension = doc_embeddings.shape[1]       
    index = faiss.IndexFlatIP(dimension)       
    faiss.normalize_L2(doc_embeddings)         
    index.add(doc_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index with {index.ntotal} vectors")

model = SentenceTransformer("all-MiniLM-L6-v2")

def search(query: str, top_k: int = 5):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25} {'Year'}")
    print("-" * 90)
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        doc = corpus[idx]
        print(f"{rank:<6} {score:.4f}  {doc['title'][:34]:<35} {doc['artist'][:24]:<25} {doc['year']}")

if __name__ == "__main__":
    queries = [
        "contemplating life and existence",
        "hopeful for future",
        'songs that include the word "dreams"',
        "songs about cats, bc i love mine so much",
        "heartbreak crying moving on",
        "summer bangers",
        "nostalgic songs",
        'songs similar to "Blank Space" by Taylor Swift',
        "angry breakup",
        "songs about vienna",
    ]

    print("\nFAISS Search Results:")
    for q in queries:
        search(q)



