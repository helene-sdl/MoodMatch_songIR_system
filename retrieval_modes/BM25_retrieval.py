import re
import string
import json
import os

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

CORPUS_PATH = "data/processed/corpus.json"

if os.path.exists(CORPUS_PATH):
    print("Loading corpus from disk...")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} documents")

else:
    os.makedirs("data/processed", exist_ok=True)
    from datasets import load_dataset

    print("Downloading dataset...")
    ds = load_dataset("theelderemo/genius-lyrics-cleaned", split="train")
    print(f"Total songs: {len(ds)}")

    filtered = ds.filter(
        lambda row: str(row["tag"]).lower() == "pop",
        desc="Filtering to pop"
    )
    print(f"After filter: {len(filtered)} pop songs")

    print("Preprocessing (this will take a while)...")
    corpus = []
    for row in filtered:
        corpus.append({
            "title":  row.get("title", "Unknown"),
            "artist": row.get("artist", "Unknown"),
            "genre":  row.get("tag", ""),
            "year":   row.get("year", ""),
            "lyrics": row.get("lyrics", ""),
            "tokens": preprocess(row.get("lyrics", "")),
        })

    corpus = [doc for doc in corpus if doc["tokens"]]
    print(f"Documents ready: {len(corpus)}")

    print("Saving to disk...")
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f)
    print("Saved!")

print("\nBuilding BM25 index...")
bm25 = BM25Okapi([doc["tokens"] for doc in corpus])
print("Done!")

def search(query, top_k=5):
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25} {'Genre'}")
    print("-" * 90)
    for rank, idx in enumerate(top_indices, 1):
        doc = corpus[idx]
        print(f"{rank:<6} {round(scores[idx],4):<8} {doc['title'][:34]:<35} {doc['artist'][:24]:<25} {doc['genre']}")

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