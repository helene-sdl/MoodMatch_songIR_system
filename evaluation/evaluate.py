"""
Evaluation script for MoodMatch IR system.
Computes nDCG@10 and MAP@10 for BM25, ST, and RRF on the golden dataset.

Usage (on server):
    unset VIRTUAL_ENV
    uv run python -m evaluation.evaluate
"""

import json
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from retrieval_modes.preprocessing import preprocess
from retrieval_modes.rrf_retrieval import search_rrf

GRADED_PATH      = "evaluation/queries_personal_graded.json"
BM25_PICKLE      = "processed/bm25_index.pkl"
ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
FAISS_INDEX_PATH = "processed/faiss_index.bin"
MODEL_NAME       = "all-MiniLM-L6-v2"
TOP_K            = 50


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



def search_bm25(query, corpus, bm25, top_k=TOP_K):
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [str(i) for i in top_indices]


def search_st(query, corpus, index, model, top_k=TOP_K):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, top_k)
    return [str(i) for i in indices[0]]


def search_rrf_ids(query, bm25_corpus, bm25, st_corpus, index, model, top_k=TOP_K):
    results = search_rrf(query, bm25_corpus, bm25, st_corpus, index, model, top_k)
    return [str(r["idx"]) for r in results]


def dcg_at_k(relevances, k):
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids, graded_ids, k=TOP_K):
    relevances = [graded_ids.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    ideal = sorted(graded_ids.values(), reverse=True)[:k]
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(retrieved_ids, graded_ids, k=TOP_K):
    relevant = {doc_id for doc_id, grade in graded_ids.items() if grade > 0}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


def evaluate():
    with open(GRADED_PATH) as f:
        golden = json.load(f)

    bm25_corpus, bm25, st_corpus, index, model = load_resources()

    methods = {"BM25": [], "ST": [], "RRF": []}

    print(f"\n{'Query':<45} {'BM25':>8} {'ST':>8} {'RRF':>8}")
    print("-" * 75)

    for item in golden:
        query = item["query"]
        graded_ids = item["graded_ids"]

        bm25_ids  = search_bm25(query, bm25_corpus, bm25)
        st_ids    = search_st(query, st_corpus, index, model)
        rrf_ids   = search_rrf_ids(query, bm25_corpus, bm25, st_corpus, index, model)

        scores = {}
        for name, ids in [("BM25", bm25_ids), ("ST", st_ids), ("RRF", rrf_ids)]:
            ndcg = ndcg_at_k(ids, graded_ids)
            ap   = average_precision(ids, graded_ids)
            scores[name] = (ndcg, ap)
            methods[name].append((ndcg, ap))

        print(f"{query[:44]:<45} {scores['BM25'][0]:>8.3f} {scores['ST'][0]:>8.3f} {scores['RRF'][0]:>8.3f}")

    print("-" * 75)
    print(f"\n{'Mean nDCG@10':<45} ", end="")
    for name in ["BM25", "ST", "RRF"]:
        mean_ndcg = np.mean([s[0] for s in methods[name]])
        print(f"{mean_ndcg:>8.3f} ", end="")
    print()

    print(f"{'MAP@10':<45} ", end="")
    for name in ["BM25", "ST", "RRF"]:
        mean_ap = np.mean([s[1] for s in methods[name]])
        print(f"{mean_ap:>8.3f} ", end="")
    print()

    print("\nDone.")


if __name__ == "__main__":
    evaluate()