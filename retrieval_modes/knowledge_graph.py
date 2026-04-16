#knowledge graph construction (song has a title, an artist, lyrics, a year, and most importantly, a mood/theme))
#either with classifier & labels (from huggingface) or with clustering and manually labeling(k-means)? 
import pickle
import networkx as nx
from transformers import pipeline
from tqdm import tqdm

ST_CORPUS_PICKLE = "data/processed/st_corpus.pkl"
GRAPH_PATH       = "data/processed/knowledge_graph.pkl"
SAMPLE_SIZE      = 500  # set to None for full corpus on GPU server

print("Loading corpus...")
with open(ST_CORPUS_PICKLE, "rb") as f:
    corpus = pickle.load(f)

if SAMPLE_SIZE:
    corpus = corpus[:SAMPLE_SIZE]

print("Loading emotion classifier...")
classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=1,
    device=-1,
    truncation=True,
    max_length=512
)

G = nx.DiGraph()

for i, doc in enumerate(tqdm(corpus)):
    song_id   = f"song_{i}"
    artist_id = f"artist_{doc['artist'].lower().strip()}"
    mood      = classifier(doc["lyrics"][:512])[0][0]["label"].lower()

    G.add_node(song_id,   type="song",   title=doc["title"], artist=doc["artist"], year=doc.get("year", ""))
    G.add_node(artist_id, type="artist", name=doc["artist"])
    G.add_node(f"mood_{mood}", type="mood", mood=mood)

    G.add_edge(artist_id,       song_id,        relation="made")
    G.add_edge(song_id,         f"mood_{mood}", relation="has_mood")

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)
print("Saved!")
