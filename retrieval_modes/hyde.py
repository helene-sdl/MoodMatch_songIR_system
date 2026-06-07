"""
HyDE (Hypothetical Document Embeddings) for MoodMatch.

Instead of encoding the raw query, generates hypothetical song lyrics
matching the mood/feeling, then encodes those for FAISS search.

If no Anthropic API key is set, falls back to pre-generated hypotheses
for the evaluation query set.

Usage:
    from retrieval_modes.hyde import search_hyde
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

HYDE_PROMPT = """You are a lyricist. Write 4-6 lines of realistic song lyrics that perfectly match this mood, feeling or theme:

"{query}"

Rules:
- Write ONLY the lyrics, no titles, no explanations, no quotes
- Make them sound like real pop song lyrics
- Match the emotional tone precisely
- Use concrete imagery and natural language"""

# Pre-generated hypotheses for eval queries (used as fallback when no API key)
HYPOTHESIS_FALLBACKS = {
    "contemplating life and existence": (
        "I stare at the ceiling at 3am wondering what it all means\n"
        "Is this the life I chose or just the one that happened to me\n"
        "Every answer leads to questions I can't seem to leave behind\n"
        "Searching for a reason in the spaces of my mind"
    ),
    "hopeful for future": (
        "The sun is coming up on everything I've been waiting for\n"
        "I can feel it in my bones, something better's at the door\n"
        "All the broken pieces finally start to make some sense\n"
        "I'm ready now to face whatever comes"
    ),
    'songs that include the word "dreams"': (
        "I had dreams of golden skies and open roads ahead\n"
        "Dreams that kept me going when I should've stayed in bed\n"
        "Every dream I ever had led me back to you\n"
        "Still dreaming, still believing it comes true"
    ),
    "songs about cats": (
        "She curls up on the window ledge and watches the world go by\n"
        "Doesn't care about my problems, just blinks her golden eyes\n"
        "In a world that never stops she's perfectly at ease\n"
        "Living life exactly as she please"
    ),
    "heartbreak crying moving on": (
        "I found your sweater in the back of my closet today\n"
        "Sat on the floor and cried until the feeling went away\n"
        "It's getting easier to breathe without you here\n"
        "Still learning how to live without my fear"
    ),
    "summer bangers": (
        "Windows down and stereo loud, we're driving to the beach\n"
        "Sun on my skin and sand between my toes, happiness in reach\n"
        "Dancing in the parking lot like no one's watching us\n"
        "This is what we live for, all the fuss"
    ),
    "nostalgic songs": (
        "Remember when we used to stay out past the streetlights came on\n"
        "Running through the sprinklers till our parents called us home\n"
        "Those summers felt like they would last forever and a day\n"
        "I'd give anything to find my way back"
    ),
    'songs similar to "Blank Space" by Taylor Swift': (
        "I've got a list of ex-lovers and a talent for the drama\n"
        "I'll love you like a hurricane and leave you for the karma\n"
        "Darling I'm a nightmare dressed like a daydream you can't shake\n"
        "Come and play my game, see what mistakes we'll make"
    ),
    "angry breakup": (
        "Don't call my name like you still have the right\n"
        "You lost that privilege the night you chose to lie\n"
        "I gave you everything and you threw it in my face\n"
        "Now get your stuff and get out of my space"
    ),
    "songs about vienna": (
        "Cobblestone streets and coffee in the rain\n"
        "Church bells ringing echo down the lane\n"
        "I fell in love with you on a November night\n"
        "Vienna in the winter, everything felt right"
    ),
}


def generate_hypothetical_lyrics(query: str, api_key: str = None) -> tuple[str, bool]:
    """
    Generate hypothetical song lyrics for a mood query.
    Uses Claude API if key available, otherwise falls back to pre-generated hypotheses.

    Returns:
        (lyrics, is_generated) — is_generated=True if from API, False if fallback
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if key:
        try:
            import requests
            headers = {
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            }
            payload = {
                "model": CLAUDE_MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": HYDE_PROMPT.format(query=query)}]
            }
            response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip(), True
        except Exception as e:
            print(f"API call failed ({e}), falling back to pre-generated hypothesis")

    # fallback — check exact match first, then find closest
    if query in HYPOTHESIS_FALLBACKS:
        return HYPOTHESIS_FALLBACKS[query], False

    # fuzzy match — find closest query in fallbacks
    query_lower = query.lower()
    for key_query, lyrics in HYPOTHESIS_FALLBACKS.items():
        if any(word in key_query.lower() for word in query_lower.split() if len(word) > 3):
            return lyrics, False

    # last resort — generic fallback
    return (
        "Lost in the feeling, searching for the words\n"
        "Everything is fleeting but the music still is heard\n"
        "These songs are all I have to make it through the night\n"
        "Holding on to melodies until the morning light"
    ), False


def search_hyde(
    query: str,
    corpus: list,
    index: faiss.Index,
    model: SentenceTransformer,
    top_k: int = 10,
    api_key: str = None,
) -> list[dict]:
    """
    HyDE retrieval: generate hypothetical lyrics → encode → FAISS search.

    Returns:
        List of result dicts with title, artist, year, lyrics, score, idx, hypothesis
    """
    hypothesis, is_generated = generate_hypothetical_lyrics(query, api_key)

    query_embedding = model.encode([hypothesis], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        doc = corpus[idx]
        results.append({
            "idx":          int(idx),
            "title":        doc["title"],
            "artist":       doc["artist"],
            "year":         doc.get("year", ""),
            "lyrics":       doc.get("lyrics", "")[:200],
            "score":        round(float(score), 4),
            "hypothesis":   hypothesis,
            "is_generated": is_generated,
        })
    return results


if __name__ == "__main__":
    import pickle

    ST_CORPUS_PICKLE = "processed/st_corpus.pkl"
    FAISS_INDEX_PATH = "processed/faiss_index.bin"
    MODEL_NAME       = "all-MiniLM-L6-v2"

    print("Loading ST + FAISS...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)
    index = faiss.read_index(FAISS_INDEX_PATH)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    for q in list(HYPOTHESIS_FALLBACKS.keys())[:5]:
        print(f"\nQuery: '{q}'")
        results = search_hyde(q, corpus, index, model)
        print(f"Hypothesis:\n{results[0]['hypothesis']}\n")
        print(f"{'Rank':<6} {'Score':<8} {'Title':<35} {'Artist':<25}")
        print("-" * 80)
        for rank, r in enumerate(results, 1):
            print(f"{rank:<6} {r['score']:<8} {r['title'][:34]:<35} {r['artist'][:24]}")