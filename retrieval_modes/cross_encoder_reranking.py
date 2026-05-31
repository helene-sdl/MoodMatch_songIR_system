import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from retrieval_modes.rrf_retrieval import search_rrf

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_PICKLE         = "processed/bm25_index.pkl"
ST_CORPUS_PICKLE    = "processed/st_corpus.pkl"
FAISS_INDEX_PATH    = "processed/faiss_index.bin"
TOP_K               = 10
CANDIDATE_POOL      = 20  # rerank top-20 from RRF


@staticmethod
def _load_cross_encoder():
    return CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")


def rerank(
    query: str,
    candidates: list[dict],
    cross_encoder: CrossEncoder,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Rerank a list of candidate results using a cross-encoder.

    Args:
        query:         User query string
        candidates:    List of result dicts (must have 'lyrics' key)
        cross_encoder: Loaded CrossEncoder model
        top_k:         Number of results to return

    Returns:
        Reranked list of result dicts with updated 'score'
    """
    if not candidates:
        return []

    pairs = [(query, doc["lyrics"]) for doc in candidates]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(candidates, scores):
        doc["score"] = round(float(score), 4)

    reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def search_with_reranking(
    query: str,
    bm25_corpus: list,
    bm25,
    st_corpus: list,
    index: faiss.Index,
    model: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k: int = TOP_K,
    candidate_pool: int = CANDIDATE_POOL,
) -> list[dict]:
    """
    Full pipeline: RRF retrieval → cross-encoder reranking.

    Args:
        query:          User query
        bm25_corpus:    BM25 corpus
        bm25:           BM25 index
        st_corpus:      ST corpus
        index:          FAISS index
        model:          SentenceTransformer model
        cross_encoder:  CrossEncoder model
        top_k:          Final number of results
        candidate_pool: Candidates to fetch before reranking

    Returns:
        Reranked top-k results
    """
    candidates = search_rrf(
        query, bm25_corpus, bm25, st_corpus, index, model,
        top_k=candidate_pool, candidate_pool=200
    )
    return rerank(query, candidates, cross_encoder, top_k)


# --- Standalone test ---
QUERIES = [
    "contemplating life and existence",
    "heartbreak crying moving on",
    "nostalgic songs",
    "angry breakup",
    "summer bangers",
]

def main():
    print("Loading BM25...")
    with open(BM25_PICKLE, "rb") as f:
        bm25_corpus, bm25 = pickle.load(f)

    print("Loading ST + FAISS...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        st_corpus = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print("Loading cross-encoder...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")

    for q in QUERIES:
        results = search_with_reranking(q, bm25_corpus, bm25, st_corpus, index, model, cross_encoder)
        print(f"\nQuery: '{q}'")
        print(f"{'Rank':<6} {'Score':<10} {'Title':<35} {'Artist':<25} {'Year'}")
        print("-" * 90)
        for rank, r in enumerate(results, 1):
            print(f"{rank:<6} {r['score']:<10} {r['title'][:34]:<35} {r['artist'][:24]:<25} {r['year']}")


if __name__ == "__main__":
    main()