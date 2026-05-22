import os
import pickle
import random
 
import ijson
import numpy as np
from sentence_transformers import SentenceTransformer, util
 
from retrieval_modes.preprocessing import preprocess
 
SAMPLE_SIZE      = None  #Set to None on GPU server for full corpus
CORPUS_PATH      = "processed/corpus.json"
EMBEDDINGS_PATH  = "processed/st_embeddings.npy"
ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
MODEL_NAME       = "all-MiniLM-L6-v2"
 
QUERIES = [
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
 
 
def load_or_build_corpus() -> list:
    if os.path.exists(ST_CORPUS_PICKLE):
        print("Loading ST corpus from pickle...")
        with open(ST_CORPUS_PICKLE, "rb") as f:
            corpus = pickle.load(f)
        print(f"Loaded {len(corpus)} documents")
        return corpus
 
    print(f"Streaming corpus, sampling {SAMPLE_SIZE} documents...")
    random.seed(42)
    reservoir = []
 
    with open(CORPUS_PATH, "rb") as f:
        for i, doc in enumerate(ijson.items(f, "item")):
            if SAMPLE_SIZE is None:
                reservoir.append(doc)
            elif i < SAMPLE_SIZE:
                reservoir.append(doc)
            else:
                j = random.randint(0, i)
                if j < SAMPLE_SIZE:
                    reservoir[j] = doc
            if i % 100_000 == 0 and i > 0:
                print(f"  ...scanned {i:,} docs")
    
    os.makedirs("data/processed", exist_ok=True)
    with open(ST_CORPUS_PICKLE, "wb") as f:
        pickle.dump(reservoir, f)
    print(f"Sampled and saved {len(reservoir)} documents")
    return reservoir
 
 
def load_or_build_embeddings(corpus: list, model: SentenceTransformer) -> np.ndarray:
    if os.path.exists(EMBEDDINGS_PATH):
        print("Loading embeddings from disk...")
        doc_embeddings = np.load(EMBEDDINGS_PATH)
        print(f"Loaded embeddings shape: {doc_embeddings.shape}")
        return doc_embeddings
 
    print("Generating embeddings (this will take a while)...")
    doc_texts = [doc["lyrics"] for doc in corpus]
    doc_embeddings = model.encode(
        doc_texts,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    np.save(EMBEDDINGS_PATH, doc_embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    return doc_embeddings
 
 
def search(query: str, corpus: list, model: SentenceTransformer, doc_embeddings: np.ndarray, top_k: int = 5) -> None:
    query_embedding = model.encode(query, convert_to_numpy=True)
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    top_indices = scores.topk(top_k).indices
 
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25} {'Year'}")
    print("-" * 90)
    for rank, idx in enumerate(top_indices, 1):
        doc = corpus[idx.item()]
        print(f"{rank:<6} {scores[idx].item():.4f}  {doc['title'][:34]:<35} {doc['artist'][:24]:<25} {doc['year']}")
 
 
def main() -> None:
    corpus = load_or_build_corpus()
    model = SentenceTransformer(MODEL_NAME, device="cuda:1")
    doc_embeddings = load_or_build_embeddings(corpus, model)
 
    print("\nSentenceTransformer Search Results:")
    for q in QUERIES:
        search(q, corpus, model, doc_embeddings)
 
 
if __name__ == "__main__":
    main()
