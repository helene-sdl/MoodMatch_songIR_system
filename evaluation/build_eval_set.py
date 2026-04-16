import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from retrieval_modes.preprocessing import preprocess

BM25_PICKLE      = "data/processed/bm25_index.pkl"
ST_CORPUS_PICKLE = "data/processed/st_corpus.pkl"
EMBEDDINGS_PATH  = "data/processed/st_embeddings.npy"
QUERIES_PATH     = "evaluation/queries.json"
OUTPUT_PATH      = "evaluation/queries_with_ids.json"

TOP_K = 10  

# --- Load BM25 ---
print("Loading BM25 index...")
with open(BM25_PICKLE, "rb") as f:
    bm25_corpus, bm25 = pickle.load(f)

# --- Load ST ---
print("Loading ST corpus and embeddings...")
with open(ST_CORPUS_PICKLE, "rb") as f:
    st_corpus = pickle.load(f)
doc_embeddings = np.load(EMBEDDINGS_PATH)
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load queries ---
with open(QUERIES_PATH) as f:
    raw = json.load(f)

queries = raw if isinstance(raw[0], str) else [q["query"] for q in raw]

results = []

for query in queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    # --- BM25 results ---
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)
    bm25_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:TOP_K]

    # --- ST results ---
    query_emb = model.encode(query, convert_to_numpy=True)
    st_scores = util.cos_sim(query_emb, doc_embeddings)[0].numpy()
    st_indices = sorted(
        range(len(st_scores)),
        key=lambda i: st_scores[i],
        reverse=True
    )[:TOP_K]

    # --- Merge unique candidates using stable IDs ---
    seen = {}

    # BM25 pass
    for idx in bm25_indices:
        doc = bm25_corpus[idx]
        title, artist = doc["title"], doc["artist"]
        key = (title.lower(), artist.lower())

        # stable ID (prefer real ID, fallback: title|artist)
        doc_id = doc.get("id", f"{title}|{artist}")

        if key not in seen:
            seen[key] = {
                "doc": doc,
                "id": doc_id,
                "bm25_score": round(scores[idx], 4),
                "st_score": None,
            }

    # ST pass
    for idx in st_indices:
        doc = st_corpus[idx]
        title, artist = doc["title"], doc["artist"]
        key = (title.lower(), artist.lower())
        doc_id = doc.get("id", f"{title}|{artist}")

        st_score = round(float(st_scores[idx]), 4)

        if key not in seen:
            seen[key] = {
                "doc": doc,
                "id": doc_id,
                "bm25_score": None,
                "st_score": st_score,
            }
        else:
            seen[key]["st_score"] = st_score

    # --- Sort candidates by best available score ---
    def best_score(c):
        b = c["bm25_score"] if c["bm25_score"] is not None else -1
        s = c["st_score"] if c["st_score"] is not None else -1
        return max(b, s)

    candidates = sorted(list(seen.values()), key=best_score, reverse=True)

    print(f"\nTotal candidates to judge: {len(candidates)}")
    print("y = relevant, n = not relevant, s = skip\n")

    # --- Judging ---
    relevant_ids = []

    for i, c in enumerate(candidates):
        doc = c["doc"]
        bm25_str = f"BM25: {c['bm25_score']}" if c["bm25_score"] else "BM25: —"
        st_str   = f"ST: {c['st_score']}"   if c["st_score"]   else "ST: —"

        title  = doc['title'][:40]
        artist = doc['artist'][:25]
        year   = doc.get("year", "")

        print(f"[{i+1:2d}] {title:<42} {artist:<27} {year}  |  {bm25_str}  {st_str}")

        while True:
            ans = input("     Relevant? (y/n/s): ").strip().lower()
            if ans in ("y", "n", "s"):
                break

        if ans == "y":
            relevant_ids.append(c["id"])
        elif ans == "s":
            break

    # --- Save per-query result ---
    results.append({
        "query": query,
        "candidates": [c["id"] for c in candidates],
        "relevant_ids": relevant_ids
    })

    print(f"\n → Marked {len(relevant_ids)} relevant documents")

# --- Save all queries ---
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved evaluation set to {OUTPUT_PATH}")