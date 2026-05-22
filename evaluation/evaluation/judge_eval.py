"""
Golden dataset judging script.
Loads queries_personal.json, shows each song, lets you grade 0/1/2.
Saves results to evaluation/queries_personal_graded.json.

Usage:
    uv run python -m evaluation.judge_eval
"""

import json
import os
import pickle

QUERIES_PATH   = "evaluation/queries_personal.json"
OUTPUT_PATH    = "evaluation/queries_personal_graded.json"
CORPUS_PICKLE  = "processed/st_corpus.pkl"

GRADES = {"0": 0, "1": 1, "2": 2}
GRADE_LABELS   = {0: "not relevant", 1: "somewhat relevant", 2: "highly relevant"}


def load_corpus(path: str) -> list:
    print("Loading corpus for metadata lookup...")
    with open(path, "rb") as f:
        corpus = pickle.load(f)
    print(f"Loaded {len(corpus):,} documents")
    return corpus


def get_song_info(corpus: list, doc_id: int) -> dict:
    if doc_id >= len(corpus):
        return {"title": "UNKNOWN", "artist": "UNKNOWN", "year": ""}
    doc = corpus[doc_id]
    return {
        "title":  doc.get("title", "UNKNOWN"),
        "artist": doc.get("artist", "UNKNOWN"),
        "year":   doc.get("year", ""),
        "lyrics": doc.get("lyrics", "")[:150],
    }


def judge(corpus: list) -> list:
    with open(QUERIES_PATH, "r") as f:
        queries = json.load(f)

    # load existing progress if any
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            graded = json.load(f)
        done_queries = {q["query"] for q in graded}
        print(f"Resuming — {len(done_queries)} queries already judged\n")
    else:
        graded = []
        done_queries = set()

    for q in queries:
        query = q["query"]
        if query in done_queries:
            continue

        print("\n" + "=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        print("Grade each song: 2 = highly relevant, 1 = somewhat relevant, 0 = not relevant")
        print("Press Enter to skip (keeps as 1), q to quit and save\n")

        graded_ids = {}
        for doc_id in q["relevant_ids"]:
            # deduplicate
            if doc_id in graded_ids:
                continue
            song = get_song_info(corpus, doc_id)
            print(f"  [{doc_id}] {song['title']} — {song['artist']} ({song['year']})")
            print(f"  \"{song['lyrics']}...\"")

            while True:
                grade = input("  Grade (0/1/2) or q to quit: ").strip().lower()
                if grade == "q":
                    # save progress and exit
                    _save(graded)
                    print("Progress saved. Run again to continue.")
                    return graded
                elif grade == "" :
                    graded_ids[doc_id] = 1  # default to somewhat relevant
                    break
                elif grade in GRADES:
                    graded_ids[doc_id] = GRADES[grade]
                    print(f"  → {GRADE_LABELS[GRADES[grade]]}")
                    break
                else:
                    print("  Invalid input, enter 0, 1, 2 or q")

        graded.append({
            "query": query,
            "graded_ids": graded_ids
        })
        _save(graded)
        print(f"\n✓ Query saved ({len(graded)}/{len(queries)} done)")

    print("\n✅ All queries judged!")
    return graded


def _save(graded: list):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(graded, f, indent=2)


def main():
    corpus = load_corpus(CORPUS_PICKLE)
    judge(corpus)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()