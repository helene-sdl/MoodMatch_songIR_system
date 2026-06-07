"""
GLiNER theme extraction — adds theme nodes to existing knowledge graph.
Loads existing graph, runs GLiNER on lyrics in batches, saves back.

DO NOT rerun knowledge_graph.py — this only adds theme nodes.

Usage (on server):
    tmux new -s gliner
    uv run python -m retrieval_modes.gliner_themes
    Ctrl+B D to detach
"""

import os
import pickle
import torch
from gliner import GLiNER

GRAPH_PATH       = "processed/knowledge_graph.pkl"
ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
MODEL_NAME       = "urchade/gliner_small-v2.1"
BATCH_SIZE       = 32
CONFIDENCE       = 0.4
SAVE_EVERY       = 10_000  # save progress every N songs

THEMES = [
    "nostalgia", "longing", "heartbreak", "breakup", "loss", "grief",
    "hope", "joy", "freedom", "anger", "frustration", "love", "desire",
    "loneliness", "redemption", "memory", "youth", "summer", "night",
    "rain", "road trip", "party", "dancing", "fame", "money", "anxiety",
    "existentialism", "nature", "spirituality", "empowerment",
    "relationship", "moving on", "self-love",
]

SKIP_THEMES = {"neutral"}


def load_resources():
    print("Loading knowledge graph...")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph with {G.number_of_nodes()} nodes")

    print("Loading corpus...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)
    print(f"Loaded {len(corpus):,} songs")

    return G, corpus


def already_has_themes(G, song_id: str) -> bool:
    """Check if a song already has theme nodes attached."""
    if song_id not in G:
        return False
    for neighbor in G.successors(song_id):
        if G.nodes[neighbor].get("type") == "theme":
            return True
    return False


def extract_themes(model: GLiNER, texts: list[str]) -> list[list[tuple[str, float]]]:
    """
    Run GLiNER on a batch of texts.
    Returns list of (theme, score) pairs per text.
    """
    results = []
    entities_batch = model.batch_predict_entities(texts, THEMES, threshold=CONFIDENCE)
    for entities in entities_batch:
        seen = {}
        for e in entities:
            label = e["label"].lower()
            score = e["score"]
            if label in SKIP_THEMES:
                continue
            if label not in seen or score > seen[label]:
                seen[label] = score
        results.append(list(seen.items()))
    return results


def add_theme_nodes(G, song_idx: int, themes: list[tuple[str, float]]):
    """Add theme nodes and edges to the graph for a song."""
    song_id = f"song_{song_idx}"
    if song_id not in G:
        return
    for theme, score in themes:
        theme_id = f"theme_{theme.replace(' ', '_')}"
        if not G.has_node(theme_id):
            G.add_node(theme_id, type="theme", theme=theme)
        if not G.has_edge(song_id, theme_id):
            G.add_edge(song_id, theme_id, weight=round(score, 3))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    G, corpus = load_resources()

    print(f"Loading GLiNER model: {MODEL_NAME}...")
    model = GLiNER.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    total = len(corpus)
    processed = 0
    skipped = 0

    print(f"\nProcessing {total:,} songs in batches of {BATCH_SIZE}...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = corpus[batch_start:batch_start + BATCH_SIZE]
        indices = list(range(batch_start, batch_start + len(batch)))

        # skip songs that already have themes
        to_process = []
        to_process_idx = []
        for i, (idx, doc) in enumerate(zip(indices, batch)):
            song_id = f"song_{idx}"
            if already_has_themes(G, song_id):
                skipped += 1
            else:
                lyrics = doc.get("lyrics", "")[:500]  # truncate for speed
                if lyrics.strip():
                    to_process.append(lyrics)
                    to_process_idx.append(idx)

        if to_process:
            themes_batch = extract_themes(model, to_process)
            for idx, themes in zip(to_process_idx, themes_batch):
                add_theme_nodes(G, idx, themes)
                processed += 1

        # progress
        done = batch_start + len(batch)
        if done % 10_000 == 0 or done == total:
            print(f"  {done:,}/{total:,} — processed: {processed:,}, skipped (already done): {skipped:,}")

        # periodic save
        if processed > 0 and processed % SAVE_EVERY == 0:
            print(f"  Saving checkpoint at {processed:,} processed...")
            with open(GRAPH_PATH, "wb") as f:
                pickle.dump(G, f)
            print("  Saved.")

    print(f"\nDone! Processed {processed:,} songs, skipped {skipped:,}")
    print("Saving final graph...")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved to {GRAPH_PATH}")
    print(f"Graph now has {G.number_of_nodes():,} nodes")


if __name__ == "__main__":
    main()