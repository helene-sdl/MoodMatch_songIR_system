import pickle
import numpy as np
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from retrieval_modes.preprocessing import preprocess
 
BM25_PICKLE      = "processed/processed/bm25_index.pkl"
ST_CORPUS_PICKLE = "processed/processed/st_corpus.pkl"
EMBEDDINGS_PATH  = "processed/processed/st_embeddings.npy"
GRAPH_PATH       = "processed/processed/knowledge_graph.pkl"
MODEL_NAME       = "all-MiniLM-L6-v2"
TOP_K            = 10
 
st.set_page_config(
    page_title="MoodMatch",
    page_icon="🎵",
    layout="wide"
)
 
st.markdown("""
    <style>
    .main { background-color: #0f0f0f; }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
        font-size: 18px;
        padding: 12px;
    }
    .result-card {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .song-title {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }
    .song-meta {
        font-size: 13px;
        color: #888888;
        margin-top: 4px;
    }
    .mood-tag {
        display: inline-block;
        background-color: #8B6914;
        color: white;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        margin-top: 6px;
    }
    .lyrics-snippet {
        font-size: 13px;
        color: #aaaaaa;
        margin-top: 8px;
        font-style: italic;
        border-left: 3px solid #333;
        padding-left: 10px;
    }
    </style>
""", unsafe_allow_html=True)
 
@st.cache_resource
def load_bm25():
    with open(BM25_PICKLE, "rb") as f:
        corpus, bm25 = pickle.load(f)
    return corpus, bm25
 
 
@st.cache_resource
def load_st():
    with open(ST_CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return corpus, embeddings, model
 
 
@st.cache_resource
def load_graph():
    try:
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
        return G
    except FileNotFoundError:
        return None
 
 
def get_mood(G, song_idx: int) -> str:
    """Look up mood for a song from the knowledge graph."""
    if G is None:
        return "unknown"
    song_id = f"song_{song_idx}"
    if song_id not in G:
        return "unknown"
    for neighbor in G.successors(song_id):
        data = G.nodes[neighbor]
        if data.get("type") == "mood":
            return data["mood"]
    return "unknown"
 
 
def search_bm25(query: str, corpus: list, bm25: BM25Okapi, G, top_k: int) -> list:
    tokens = preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        doc = corpus[idx]
        results.append({
            "title": doc["title"],
            "artist": doc["artist"],
            "year": doc.get("year", ""),
            "mood": get_mood(G, idx),
            "lyrics": doc.get("lyrics", "")[:200],
            "score": round(scores[idx], 4),
        })
    return results
 
 
def search_st(query: str, corpus: list, embeddings: np.ndarray, model: SentenceTransformer, G, top_k: int) -> list:
    query_embedding = model.encode(query, convert_to_numpy=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = scores.topk(top_k).indices
    results = []
    for idx in top_indices:
        idx = idx.item()
        doc = corpus[idx]
        results.append({
            "title": doc["title"],
            "artist": doc["artist"],
            "year": doc.get("year", ""),
            "mood": get_mood(G, idx),
            "lyrics": doc.get("lyrics", "")[:200],
            "score": round(scores[idx].item(), 4),
        })
    return results
 
 
def render_results(results: list):
    for r in results:
        st.markdown(f"""
        <div class="result-card">
            <div class="song-title">{r['title']}</div>
            <div class="song-meta">{r['artist']} &nbsp;·&nbsp; {r['year']} &nbsp;·&nbsp; score: {r['score']}</div>
            <span class="mood-tag">{r['mood']}</span>
            <div class="lyrics-snippet">{r['lyrics']}...</div>
        </div>
        """, unsafe_allow_html=True)
 
 
# --- UI ---
st.markdown("# 🎵 MoodMatch")
st.markdown("*Find songs that match your mood, feeling or theme*")
st.divider()
 
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("", placeholder="e.g. heartbreak crying moving on, nostalgic summer, contemplating life...")
with col2:
    method = st.selectbox("Retrieval method", ["BM25", "SentenceTransformers"])
 
if query:
    G = load_graph()
 
    with st.spinner("Searching..."):
        if method == "BM25":
            corpus, bm25 = load_bm25()
            results = search_bm25(query, corpus, bm25, G, TOP_K)
        else:
            corpus, embeddings, model = load_st()
            results = search_st(query, corpus, embeddings, model, G, TOP_K)
 
    st.markdown(f"**Top {TOP_K} results for:** *{query}*")
    render_results(results)