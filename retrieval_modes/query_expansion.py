import nltk
nltk.data.path.insert(0, '/home/mlt_ml1/nltk_data')

from nltk.corpus import wordnet


def expand_query(query: str, max_synonyms: int = 3) -> str:
    tokens = query.lower().split()
    extra = []

    for token in tokens:
        synonyms = set()
        for syn in wordnet.synsets(token):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != token.lower():
                    synonyms.add(candidate.lower())
            if len(synonyms) >= max_synonyms:
                break
        extra.extend(list(synonyms)[:max_synonyms])

    expanded = query + " " + " ".join(extra) if extra else query
    return expanded


def expand_and_preprocess(query: str, max_synonyms: int = 3) -> list[str]:
    """
    Expand query with synonyms then preprocess.
    Use this as a drop-in replacement for preprocess() in BM25 search.
    """
    from retrieval_modes.preprocessing import preprocess  # lazy import avoids circular network call
    expanded = expand_query(query, max_synonyms)
    return preprocess(expanded)


if __name__ == "__main__":
    test_queries = [
        "contemplating life and existence",
        "hopeful for future",
        "heartbreak crying moving on",
        "nostalgic songs",
        "angry breakup",
        "summer bangers",
    ]

    for q in test_queries:
        expanded = expand_query(q)
        print(f"\nOriginal:  {q}")
        print(f"Expanded:  {expanded}")