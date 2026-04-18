import os
import pickle
import random
 
import ijson
from rank_bm25 import BM25Okapi
 
from retrieval_modes.preprocessing import preprocess
 
SAMPLE_SIZE = 50_000  #Set to None on GPU server for full corpus
CORPUS_PATH = "data/processed/corpus.json"
BM25_PICKLE = "data/processed/bm25_index.pkl"
 
QUERIES = [
    "contemplating life and existence",
    "hopeful for future",
    'songs that include the word "dreams"',
    "songs about cats bc i love mine so much",
    "heartbreak crying moving on",
    "summer bangers",
    "nostalgic songs",
    'songs similar to "Blank Space" by Taylor Swift',
    "angry breakup",
    "everything is changing",
]
 
 
def load_or_build_index() -> tuple:
    #Load BM25 index from pickle if available, otherwise build from corpus
    if os.path.exists(BM25_PICKLE):
        print("Loading BM25 index from pickle...")
        with open(BM25_PICKLE, "rb") as f:
            corpus, bm25 = pickle.load(f)
        print(f"Loaded {len(corpus)} documents")
        return corpus, bm25
 
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"Corpus not found at {CORPUS_PATH}. "
        )
 
    corpus = _stream_and_sample(CORPUS_PATH, SAMPLE_SIZE)
 
    print(f"Building BM25 index on {len(corpus)} documents...")
    bm25 = BM25Okapi([doc["tokens"] for doc in corpus])
 
    print("Saving index to pickle...")
    os.makedirs(os.path.dirname(BM25_PICKLE), exist_ok=True)
    with open(BM25_PICKLE, "wb") as f:
        pickle.dump((corpus, bm25), f)
    print("Saved!")
 
    return corpus, bm25
 
 
def _stream_and_sample(corpus_path: str, sample_size: int) -> list:
    print(f"Streaming corpus, sampling {sample_size} documents...")
    random.seed(42)
    reservoir = []
 
    with open(corpus_path, "rb") as f:
        for i, doc in enumerate(ijson.items(f, "item")):
            if i < sample_size:
                reservoir.append(doc)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = doc
            if i % 100_000 == 0 and i > 0:
                print(f"  ...scanned {i:,} docs")
 
    print(f"Sampled {len(reservoir)} documents")
    return reservoir
 
 
def search(query: str, corpus: list, bm25: BM25Okapi, top_k: int = 5) -> None:
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
 
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25} {'Year'}")
    print("-" * 90)
    for rank, idx in enumerate(top_indices, 1):
        doc = corpus[idx]
        print(f"{rank:<6} {round(scores[idx], 4):<8} {doc['title'][:34]:<35} {doc['artist'][:24]:<25} {doc['year']}")
 
 
def main() -> None:
    corpus, bm25 = load_or_build_index()
    print("\nBM25 Search Results:")
    for q in QUERIES:
        search(q, corpus, bm25)
 
 
if __name__ == "__main__":
    main()