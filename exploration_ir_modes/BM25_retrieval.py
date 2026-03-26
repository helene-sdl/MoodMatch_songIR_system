import os
import pickle
import random
import ijson
from rank_bm25 import BM25Okapi
from exploration_ir_modes.preprocessing import preprocess


SAMPLE_SIZE = 50_000  #set to None on GPU server for full corpus
CORPUS_PATH = "data/processed/corpus.json"
BM25_PICKLE  = "data/processed/bm25_index.pkl"

if os.path.exists(BM25_PICKLE):
    print("Loading BM25 index from pickle...")
    with open(BM25_PICKLE, "rb") as f:
        corpus, bm25 = pickle.load(f)
    print(f"Loaded {len(corpus)} documents")

else:
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_PATH}. "
            "Run the dataset download script first."
        )

    print(f"Streaming corpus, sampling {SAMPLE_SIZE} documents...")
    random.seed(42)
    reservoir = []
    with open(CORPUS_PATH, "rb") as f:
        for i, doc in enumerate(ijson.items(f, "item")):
            if i < SAMPLE_SIZE:
                reservoir.append(doc)
            else:
                j = random.randint(0, i)
                if j < SAMPLE_SIZE:
                    reservoir[j] = doc
            if i % 100_000 == 0 and i > 0:
                print(f"  ...scanned {i:,} docs")

    corpus = reservoir
    print(f"Sampled {len(corpus)} documents, building BM25 index...")
    bm25 = BM25Okapi([doc["tokens"] for doc in corpus])

    print("Saving index to pickle...")
    os.makedirs(os.path.dirname(BM25_PICKLE), exist_ok=True)
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump((corpus, bm25), f)
    print("Saved!")

def search(query: str, top_k: int = 5):
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25} {'Year'}")
    print("-" * 90)
    for rank, idx in enumerate(top_indices, 1):
        doc = corpus[idx]
        print(f"{rank:<6} {round(scores[idx],4):<8} {doc['title'][:34]:<35} {doc['artist'][:24]:<25} {doc['year']}")


if __name__ == "__main__":
    queries = [
        "contemplating life and existence",
        "hopeful for future",
        'songs that include the word "dreams"',
        "songs about cats or dogs",
        "heartbreak crying moving on",
        "summer bangers",
        "nostalgic songs",
        'songs similar to "Blank Space" by Taylor Swift',
        "angry breakup",
        "everything is changing",
    ]

    print("\nBM25 Search Results:")
    for q in queries:
        search(q)