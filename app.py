import html
import pickle
import numpy as np
import faiss
import streamlit as st
from collections import Counter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from retrieval_modes.query_expansion import expand_query, expand_and_preprocess
from retrieval_modes.rrf_retrieval import search_rrf

BM25_PICKLE      = "processed/bm25_index.pkl"
ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
FAISS_INDEX_PATH = "processed/faiss_index.bin"
GRAPH_PATH       = "processed/knowledge_graph.pkl"
MODEL_NAME       = "all-MiniLM-L6-v2"
TOP_K            = 10

st.set_page_config(
    page_title="MoodMatch",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #2C3E50; }
    .main { background-color: #2C3E50; }
    .stTextInput > div > div > input {
        background-color: #253444;
        color: #E8D5B0;
        border: 1px solid #4A6274;
        border-radius: 6px;
        font-size: 16px;
        padding: 10px 14px;
    }
    .stTextInput > div > div > input::placeholder { color: #7A9AB0; }
    .result-card {
        background-color: #253444;
        border: 1px solid #4A6274;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
    }
    .card-left { flex: 1; min-width: 0; }
    .card-right {
        flex-shrink: 0;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
        min-width: 130px;
    }
    .song-title { font-size: 16px; font-weight: bold; color: #E8D5B0; }
    .song-meta { font-size: 12px; color: #7A9AB0; margin-top: 3px; }
    .lyrics-snippet {
        font-size: 12px; color: #9AB0C0; margin-top: 8px;
        font-style: italic; border-left: 2px solid #4A6274; padding-left: 8px;
    }
    .tags-row { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; }
    .mood-label {
        border: 1px solid #B87333; color: #B87333;
        border-radius: 4px; padding: 3px 10px; font-size: 12px; white-space: nowrap;
    }
    .theme-label {
        border: 1px solid #4A6274; color: #9AB0C0;
        border-radius: 4px; padding: 3px 8px; font-size: 11px; white-space: nowrap;
    }
    .confidence-wrap { width: 120px; }
    .confidence-bar-bg {
        background-color: #1E2E3D; border-radius: 4px; height: 6px; width: 100%; overflow: hidden;
    }
    .confidence-bar-fill { background-color: #B87333; height: 6px; border-radius: 4px; }
    .confidence-pct { font-size: 11px; color: #7A9AB0; margin-top: 3px; text-align: right; }
    .stButton > button {
        background-color: #253444; color: #B87333;
        border: 1px solid #B87333; border-radius: 20px; font-size: 12px; padding: 3px 12px;
    }
    hr { border-color: #4A6274; }
    p, .stMarkdown { color: #E8D5B0; }
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
    index = faiss.read_index(FAISS_INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    return corpus, index, model


@st.cache_resource
def load_graph():
    try:
        with open(GRAPH_PATH, "rb") as f:
            G = pickle.load(f)
        return G
    except FileNotFoundError:
        return None


@st.cache_resource
def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def get_mood(G, song_idx: int) -> str:
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


def get_themes(G, song_idx: int, top_n: int = 2) -> list:
    if G is None:
        return []
    song_id = f"song_{song_idx}"
    if song_id not in G:
        return []
    themes = []
    for neighbor in G.successors(song_id):
        data = G.nodes[neighbor]
        if data.get("type") == "theme":
            weight = G.edges[song_id, neighbor].get("weight", 0)
            themes.append((data["theme"], weight))
    themes.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in themes[:top_n]]


def get_refinement_tags(results: list, G, top_n: int = 4) -> list:
    """Get top moods + themes across results for disambiguation chips."""
    tags = []
    moods = [r["mood"] for r in results if r.get("mood") and r["mood"] != "unknown"]
    tags.extend([m for m, _ in Counter(moods).most_common(2)])
    all_themes = []
    for r in results:
        all_themes.extend(get_themes(G, r["idx"]))
    tags.extend([t for t, _ in Counter(all_themes).most_common(2)])
    return list(dict.fromkeys(tags))[:top_n]


def filter_by_tag(results: list, G, tag: str) -> list:
    """Re-rank results: matching mood or theme first."""
    def has_tag(r):
        if r.get("mood") == tag:
            return True
        if tag in get_themes(G, r["idx"]):
            return True
        return False
    matching = [r for r in results if has_tag(r)]
    others = [r for r in results if not has_tag(r)]
    return matching + others


def search_bm25(query: str, corpus: list, bm25: BM25Okapi, G, top_k: int) -> list:
    tokens = expand_and_preprocess(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        doc = corpus[idx]
        results.append({
            "idx":    idx,
            "title":  doc["title"],
            "artist": doc["artist"],
            "year":   doc.get("year", ""),
            "mood":   get_mood(G, idx),
            "lyrics": doc.get("lyrics", "")[:200],
            "score":  round(scores[idx], 4),
        })
    return results


def search_st(query: str, corpus: list, index: faiss.Index, model: SentenceTransformer, G, top_k: int) -> list:
    expanded = expand_query(query)
    query_embedding = model.encode([expanded], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = corpus[idx]
        results.append({
            "idx":    int(idx),
            "title":  doc["title"],
            "artist": doc["artist"],
            "year":   doc.get("year", ""),
            "mood":   get_mood(G, idx),
            "lyrics": doc.get("lyrics", "")[:200],
            "score":  round(float(score), 4),
        })
    return results


def render_results(results: list, method: str, G):
    all_scores = [r["score"] for r in results]
    min_s = min(all_scores) if all_scores else 0
    max_s = max(all_scores) if all_scores else 1
    for r in results:
        norm = (r["score"] - min_s) / (max_s - min_s) if max_s != min_s else 0.75
        pct = int(norm * 100)
        bar_width = int(norm * 120)
        mood = html.escape(r.get("mood", "unknown"))
        title = html.escape(r.get("title", ""))
        artist = html.escape(r.get("artist", ""))
        year = html.escape(str(r.get("year", "")))
        lyrics = html.escape(r.get("lyrics", ""))
        themes = get_themes(G, r["idx"])
        theme_html = "".join([f'<span class="theme-label">{html.escape(t)}</span>' for t in themes])
        st.markdown(f"""
        <div class="result-card">
            <div class="card-left">
                <div class="song-title">▷ {title}</div>
                <div class="song-meta">{artist} &nbsp;·&nbsp; {year}</div>
                <div class="lyrics-snippet">{lyrics}...</div>
                <div class="tags-row">
                    <span class="mood-label">{mood}</span>
                    {theme_html}
                </div>
            </div>
            <div class="card-right">
                <div class="confidence-wrap">
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width:{bar_width}px"></div>
                    </div>
                    <div class="confidence-pct">{pct}% match</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# --- UI ---
st.image("/mnt/nfs/home_backup/mlt_ml1/seh/seh/assets/MoodMatch.png", width=300)
st.markdown("*Find songs that match your mood, feeling or theme*")
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("", placeholder="e.g. heartbreak crying moving on, nostalgic summer, contemplating life...")
with col2:
    method = st.selectbox("Retrieval method", ["BM25", "SentenceTransformers", "RRF", "RRF + Reranking", "HyDE"])

if query:
    with st.spinner("Searching..."):
        if method == "BM25":
            corpus, bm25 = load_bm25()
            G = load_graph()
            results = search_bm25(query, corpus, bm25, G, TOP_K)

        elif method == "SentenceTransformers":
            corpus, index, model = load_st()
            G = load_graph()
            results = search_st(query, corpus, index, model, G, TOP_K)

        elif method == "RRF":
            bm25_corpus, bm25 = load_bm25()
            st_corpus, index, model = load_st()
            G = load_graph()
            results = search_rrf(query, bm25_corpus, bm25, st_corpus, index, model, TOP_K)
            for r in results:
                r["mood"] = get_mood(G, r["idx"])

        elif method == "RRF + Reranking":
            from retrieval_modes.cross_encoder_reranking import search_with_reranking
            bm25_corpus, bm25 = load_bm25()
            st_corpus, index, model = load_st()
            cross_encoder = load_cross_encoder()
            G = load_graph()
            results = search_with_reranking(query, bm25_corpus, bm25, st_corpus, index, model, cross_encoder)
            for r in results:
                r["mood"] = get_mood(G, r["idx"])

        elif method == "HyDE":
            from retrieval_modes.hyde import search_hyde
            corpus, index, model = load_st()
            G = load_graph()
            results = search_hyde(query, corpus, index, model, TOP_K)
            for r in results:
                r["mood"] = get_mood(G, r["idx"])

    # show HyDE hypothesis
    if method == "HyDE" and results:
        st.info(f"🎵 Searching for songs like:\n\n*{results[0]['hypothesis']}*")

    # --- Disambiguation ---
    top_tags = get_refinement_tags(results, G)
    if top_tags:
        st.markdown("**Also feeling?**")
        cols = st.columns(len(top_tags) + 1)
        selected_tag = st.session_state.get("selected_tag", None)
        with cols[0]:
            if st.button("all", key="clear_tag"):
                st.session_state["selected_tag"] = None
                selected_tag = None
        for i, tag in enumerate(top_tags):
            with cols[i + 1]:
                if st.button(tag, key=f"tag_{tag}"):
                    st.session_state["selected_tag"] = tag
                    selected_tag = tag
        if selected_tag and selected_tag in top_tags:
            results = filter_by_tag(results, G, selected_tag)

    st.markdown(f"**Top {TOP_K} results for:** *{query}*")
    render_results(results, method, G)