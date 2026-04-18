#knowledge graph construction (song has a title, an artist, lyrics, a year, and most importantly, a mood/theme))
#either with classifier & labels (from huggingface) or with clustering and manually labeling(k-means)? 
#has to be changed when on GPU server, but for now we can do a small sample on CPU to test the concept and save the graph for later use in retrieval...
#also: "reclassify" emotions maybe 
import os
import pickle
 
import networkx as nx
from tqdm import tqdm
from transformers import pipeline
 
# --- Config ---
ST_CORPUS_PICKLE = "data/processed/st_corpus.pkl"
GRAPH_PATH       = "data/processed/knowledge_graph.pkl"
SAMPLE_SIZE      = 500   # Set to None on GPU server for full corpus
MODEL_NAME       = "SamLowe/roberta-base-go_emotions"
 
# 28 emotions including: admiration, amusement, anger, annoyance, caring,
# confusion, curiosity, desire, disappointment, disgust, embarrassment,
# excitement, fear, gratitude, grief, joy, love, nervousness, nostalgia,
# optimism, pride, realization, relief, remorse, sadness, surprise, neutral
 
 
def load_corpus(sample_size: int | None) -> list:
    #Load corpus from pickle, optionally taking a sample for local testing
    print("Loading corpus...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)
    if sample_size:
        corpus = corpus[:sample_size]
        print(f"Using sample of {sample_size} songs for testing")
    else:
        print(f"Using full corpus of {len(corpus)} songs")
    return corpus
 
 
def build_graph(corpus: list, classifier) -> nx.DiGraph:
    #Build a directed knowledge graph from the corpus.
    #Nodes: Song, Artist, Mood
    #Edges: Artist -[made]-> Song, Song -[has_mood]-> Mood
    #Mood labels are predicted by the emotion classifier.
    G = nx.DiGraph()
 
    for i, doc in enumerate(tqdm(corpus, desc="Building graph")):
        song_id   = f"song_{i}"
        artist_id = f"artist_{doc['artist'].lower().strip()}"
 
        #Predict mood from lyrics (truncated to 512 tokens for the model)
        try:
            result = classifier(doc["lyrics"][:512])[0][0]
            mood = result["label"].lower()
        except Exception:
            mood = "neutral"
 
        mood_id = f"mood_{mood}"
 
        #nodes
        G.add_node(song_id,   type="song",   title=doc["title"], artist=doc["artist"], year=doc.get("year", ""))
        G.add_node(artist_id, type="artist", name=doc["artist"])
        G.add_node(mood_id,   type="mood",   mood=mood)
 
        #edges
        G.add_edge(artist_id, song_id,  relation="made")
        G.add_edge(song_id,   mood_id,  relation="has_mood")
 
    return G
 
 
def save_graph(G: nx.DiGraph) -> None:
    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {GRAPH_PATH}")
 
 
def print_stats(G: nx.DiGraph) -> None:
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
 
    mood_counts = {}
    for node, data in G.nodes(data=True):
        if data.get("type") == "mood":
            mood_counts[data["mood"]] = len(list(G.predecessors(node)))
 
    print("\nMood distribution:")
    for mood, count in sorted(mood_counts.items(), key=lambda x: -x[1]):
        print(f"  {mood:<20} {count} songs")
 
 
def main() -> None:
    corpus = load_corpus(SAMPLE_SIZE)
 
    print(f"Loading emotion classifier ({MODEL_NAME})...")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=1,
        device=-1,       # CPU; change to 0 for GPU server
        truncation=True,
        max_length=512,
    )
 
    G = build_graph(corpus, classifier)
    print_stats(G)
    save_graph(G)
 
 
if __name__ == "__main__":
    main()