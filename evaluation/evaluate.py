"""
Evaluation script for MoodMatch IR system.
Computes Recall@k and nDCG@k at multiple k values for all retrieval methods.

Usage (on server):
    uv run python -m evaluation.evaluate
"""

import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from retrieval_modes.query_expansion import expand_query, expand_and_preprocess
from retrieval_modes.rrf_retrieval import search_rrf

GRADED_PATH      = "evaluation/queries_personal_graded.json"
BM25_PICKLE      = "processed/bm25_index.pkl"
ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
FAISS_INDEX_PATH = "processed/faiss_index.bin"
MODEL_NAME       = "all-MiniLM-L6-v2"

K_VALUES         = [10, 50, 100]
CANDIDATE_POOL   = 200  # fetch more candidates for higher k evaluation


def load_resources():
    print("Loading BM25...")
    with open(BM25_PICKLE, "rb") as f:
        bm25_corpus, bm25 = pickle.load(f)

    print("Loading ST corpus + FAISS...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        st_corpus = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    return bm25_corpus, bm25, st_corpus, index, model


def get_bm25_ids(query, bm25_corpus, bm25, top_k):
    tokens = expand_and_preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [str(i) for i in top_indices]


def get_st_ids(query, st_corpus, index, model, top_k):
    expanded = expand_query(query)
    query_embedding = model.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, top_k)
    return [str(i) for i in indices[0]]


def get_rrf_ids(query, bm25_corpus, bm25, st_corpus, index, model, top_k):
    results = search_rrf(query, bm25_corpus, bm25, st_corpus, index, model,
                         top_k=top_k, candidate_pool=CANDIDATE_POOL)
    return [str(r["idx"]) for r in results]


def get_reranking_ids(query, bm25_corpus, bm25, st_corpus, index, model, top_k):
    from retrieval_modes.cross_encoder_reranking import search_with_reranking
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    results = search_with_reranking(query, bm25_corpus, bm25, st_corpus, index, model,
                                    cross_encoder, top_k=top_k, candidate_pool=CANDIDATE_POOL)
    return [str(r["idx"]) for r in results]


def get_hyde_ids(query, st_corpus, index, model, top_k):
    from retrieval_modes.hyde import search_hyde
    results = search_hyde(query, st_corpus, index, model, top_k=top_k)
    return [str(r["idx"]) for r in results]


def recall_at_k(retrieved_ids, graded_ids, k):
    """Fraction of relevant docs found in top-k."""
    relevant = set(doc_id for doc_id, grade in graded_ids.items() if grade > 0)
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return found / len(relevant)


def dcg_at_k(relevances, k):
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids, graded_ids, k):
    relevances = [int(graded_ids.get(doc_id, 0)) for doc_id in retrieved_ids[:k]]
    ideal = sorted([int(v) for v in graded_ids.values()], reverse=True)[:k]
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate():
    with open(GRADED_PATH) as f:
        golden = json.load(f)

    bm25_corpus, bm25, st_corpus, index, model = load_resources()

    max_k = max(K_VALUES)
    methods = ["BM25", "ST", "RRF", "RRF+Rerank", "HyDE"]

    # store scores: method -> k -> list of scores per query
    recall_scores  = {m: {k: [] for k in K_VALUES} for m in methods}
    ndcg_scores    = {m: {k: [] for k in K_VALUES} for m in methods}

    print(f"\nEvaluating {len(golden)} queries...\n")

    for item in golden:
        query      = item["query"]
        graded_ids = item["graded_ids"]
        print(f"Query: {query[:60]}")

        # fetch top-max_k results for each method once
        bm25_ids    = get_bm25_ids(query, bm25_corpus, bm25, max_k)
        st_ids      = get_st_ids(query, st_corpus, index, model, max_k)
        rrf_ids     = get_rrf_ids(query, bm25_corpus, bm25, st_corpus, index, model, max_k)
        rerank_ids  = get_reranking_ids(query, bm25_corpus, bm25, st_corpus, index, model, max_k)
        hyde_ids    = get_hyde_ids(query, st_corpus, index, model, max_k)

        method_ids = {
            "BM25":       bm25_ids,
            "ST":         st_ids,
            "RRF":        rrf_ids,
            "RRF+Rerank": rerank_ids,
            "HyDE":       hyde_ids,
        }

        for method, ids in method_ids.items():
            for k in K_VALUES:
                recall_scores[method][k].append(recall_at_k(ids, graded_ids, k))
                ndcg_scores[method][k].append(ndcg_at_k(ids, graded_ids, k))

    # --- Print results ---
    print("\n" + "=" * 70)
    print("RECALL@K")
    print("=" * 70)
    header = f"{'Method':<14}" + "".join(f"  k={k:<6}" for k in K_VALUES)
    print(header)
    print("-" * 70)
    for method in methods:
        row = f"{method:<14}"
        for k in K_VALUES:
            mean = np.mean(recall_scores[method][k])
            row += f"  {mean:.3f}   "
        print(row)

    print("\n" + "=" * 70)
    print("nDCG@K")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for method in methods:
        row = f"{method:<14}"
        for k in K_VALUES:
            mean = np.mean(ndcg_scores[method][k])
            row += f"  {mean:.3f}   "
        print(row)

    print("\n" + "=" * 70)
    print("PER-QUERY RECALL@100")
    print("=" * 70)
    print(f"{'Query':<45}" + "".join(f"  {m:<10}" for m in methods))
    print("-" * 95)
    for i, item in enumerate(golden):
        query = item["query"]
        row = f"{query[:44]:<45}"
        for method in methods:
            row += f"  {recall_scores[method][100][i]:.2f}      "
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    evaluate()