import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from retrieval_modes.preprocessing import preprocess
from retrieval_modes.query_expansion import expand_query, expand_and_preprocess

BM25_PICKLE      = "processed/bm25_index.pkl"
ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
FAISS_INDEX_PATH = "processed/faiss_index.bin"
MODEL_NAME       = "all-MiniLM-L6-v2"
TOP_K            = 10
RRF_K            = 60  # standard RRF constant


def reciprocal_rank_fusion(
    bm25_indices: list[int],
    st_indices: list[int],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.
    Returns a list of (doc_idx, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}

    for rank, idx in enumerate(bm25_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, idx in enumerate(st_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_rrf(
    query: str,
    bm25_corpus: list,
    bm25: BM25Okapi,
    st_corpus: list,
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int = TOP_K,
    candidate_pool: int = 100,
) -> list[dict]:
    """
    Hybrid search: BM25 + SentenceTransformer/FAISS fused with RRF.

    Returns:
        List of result dicts with title, artist, year, lyrics, score, idx
    """
    # --- BM25 candidates ---
    tokens =  expand_and_preprocess(query)
    bm25_scores = bm25.get_scores(tokens)
    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:candidate_pool]

    # --- FAISS candidates ---
    expanded = expand_query(query)
    query_embedding = model.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    _, faiss_indices = index.search(query_embedding, candidate_pool)
    st_indices = faiss_indices[0].tolist()

    # --- RRF fusion ---
    fused = reciprocal_rank_fusion(bm25_indices, st_indices)[:top_k]

    # --- Build result dicts ---
    results = []
    for idx, rrf_score in fused:
        doc = st_corpus[idx]
        results.append({
            "idx":    idx,
            "title":  doc["title"],
            "artist": doc["artist"],
            "year":   doc.get("year", ""),
            "lyrics": doc.get("lyrics", "")[:200],
            "score":  round(rrf_score, 6),
        })
    return results


# --- Standalone test ---
QUERIES = [
    "contemplating life and existence",
    "hopeful for future",
    "heartbreak crying moving on",
    "nostalgic songs",
    "angry breakup",
]

def main():
    print("Loading BM25 index...")
    with open(BM25_PICKLE, "rb") as f:
        bm25_corpus, bm25 = pickle.load(f)

    print("Loading ST corpus + FAISS index...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        st_corpus = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    for q in QUERIES:
        results = search_rrf(q, bm25_corpus, bm25, st_corpus, index, model)
        print(f"\nQuery: '{q}'")
        print(f"{'Rank':<6} {'Score':<10} {'Title':<35} {'Artist':<25} {'Year'}")
        print("-" * 90)
        for rank, r in enumerate(results, 1):
            print(f"{rank:<6} {r['score']:<10} {r['title'][:34]:<35} {r['artist'][:24]:<25} {r['year']}")

if __name__ == "__main__":
    main()