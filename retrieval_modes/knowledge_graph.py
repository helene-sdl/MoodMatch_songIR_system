import os
import pickle

import networkx as nx
from tqdm import tqdm
from transformers import pipeline

# --- Config ---
ST_CORPUS_PICKLE   = "processed/st_corpus.pkl"
GRAPH_PATH         = "processed/knowledge_graph.pkl"
SAMPLE_SIZE        = None    # Set to None on GPU server for full corpus
MODEL_NAME         = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE         = 64      # process lyrics in batches for speed
CONFIDENCE_THRESHOLD = 0.3   # minimum score to assign a mood
TOP_K_MOODS        = 3       # store top 3 moods per song

# 28 emotions including: admiration, amusement, anger, annoyance, caring,
# confusion, curiosity, desire, disappointment, disgust, embarrassment,
# excitement, fear, gratitude, grief, joy, love, nervousness, nostalgia,
# optimism, pride, realization, relief, remorse, sadness, surprise, neutral


def load_corpus(sample_size: int | None) -> list:
    """Load corpus from pickle, optionally taking a sample for local testing."""
    print("Loading corpus...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)
    if sample_size:
        corpus = corpus[:sample_size]
        print(f"Using sample of {sample_size} songs for testing")
    else:
        print(f"Using full corpus of {len(corpus)} songs")
    return corpus


def pick_moods(results: list, top_k: int, threshold: float) -> list[str]:
    """
    From classifier output, pick top_k non-neutral moods above threshold.
    Falls back to top result if nothing passes the threshold.
    Returns a list of mood label strings.
    """
    valid = [
        r["label"].lower()
        for r in results
        if r["label"].lower() != "neutral" and r["score"] >= threshold
    ][:top_k]

    if not valid:
        # fallback: take top result regardless of neutral/threshold
        valid = [results[0]["label"].lower()]

    return valid


def build_graph(corpus: list, classifier) -> nx.DiGraph:
    """
    Build a directed knowledge graph from the corpus.

    Nodes: Song, Artist, Year, Mood
    Edges:
        Artist  -[made]---------> Song
        Song    -[released_in]--> Year
        Song    -[has_mood]-----> Mood  (up to TOP_K_MOODS per song)

    Mood labels predicted by emotion classifier in batches.
    Neutral skipped where possible; confidence threshold applied.
    """
    G = nx.DiGraph()

    # Process in batches for speed
    for batch_start in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Building graph"):
        batch = corpus[batch_start: batch_start + BATCH_SIZE]
        texts = [doc["lyrics"][:512] for doc in batch]

        try:
            batch_results = classifier(texts)
        except Exception:
            batch_results = [None] * len(batch)

        for i, (doc, results) in enumerate(zip(batch, batch_results)):
            idx       = batch_start + i
            song_id   = f"song_{idx}"
            artist_id = f"artist_{doc['artist'].lower().strip()}"
            year      = str(doc.get("year", "unknown"))
            year_id   = f"year_{year}"

            # Add nodes
            G.add_node(song_id,   type="song",   title=doc["title"], artist=doc["artist"], year=year)
            G.add_node(artist_id, type="artist", name=doc["artist"])
            G.add_node(year_id,   type="year",   year=year)

            # Add structural edges
            G.add_edge(artist_id, song_id, relation="made")
            G.add_edge(song_id,   year_id, relation="released_in")

            # Add mood edges
            if results:
                moods = pick_moods(results, TOP_K_MOODS, CONFIDENCE_THRESHOLD)
            else:
                moods = ["neutral"]

            for mood in moods:
                mood_id = f"mood_{mood}"
                G.add_node(mood_id, type="mood", mood=mood)
                G.add_edge(song_id, mood_id, relation="has_mood")

    return G


def save_graph(G: nx.DiGraph) -> None:
    """Serialize graph to disk as a pickle file."""
    os.makedirs("processed", exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {GRAPH_PATH}")


def print_stats(G: nx.DiGraph) -> None:
    """Print a summary of the graph including mood distribution."""
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_types = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        node_types[t] = node_types.get(t, 0) + 1
    print("\nNode types:")
    for t, count in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {t:<12} {count:,}")

    mood_counts = {}
    for node, data in G.nodes(data=True):
        if data.get("type") == "mood":
            mood_counts[data["mood"]] = len(list(G.predecessors(node)))

    print("\nMood distribution:")
    for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
        print(f"  {mood:<20} {count:,} songs")


def main() -> None:
    corpus = load_corpus(SAMPLE_SIZE)

    print(f"Loading emotion classifier ({MODEL_NAME})...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=TOP_K_MOODS + 1,  # +1 so we have room to skip neutral
        device="cuda:1",
        truncation=True,
        max_length=512,
    )

    G = build_graph(corpus, classifier)
    print_stats(G)
    save_graph(G)


if __name__ == "__main__":
    main()