import pickle
import json

ST_CORPUS_PICKLE = "processed/processed/st_corpus.pkl"

EVAL_SET = {
    "contemplating life and existence": [
        ("The Night Will Always Win", "Manchester Orchestra"),
        ("Holocene", "Bon Iver"),
        ("Motion Picture Soundtrack", "Radiohead"),
        ("In My Life", "The Beatles"),
        ("How to Disappear Completely", "Radiohead"),
        ("Someone Great", "LCD Soundsystem"),
        ("I Found", "Amber Run"),
    ],
    "hopeful for future": [
        ("Dog Days Are Over", "Florence + The Machine"),
        ("Here Comes the Sun", "The Beatles"),
        ("Shake It Out", "Florence + The Machine"),
        ("From Eden", "Hozier"),
        ("Keep Your Head Up", "Ben Howard"),
        ("Better Days", "Ant Clemons"),
        ("It's Time", "Imagine Dragons"),
    ],
    'songs that include the word "dreams"': [
        ("Dreams", "Fleetwood Mac"),
        ("Wildest Dreams", "Taylor Swift"),
        ("In Your Dreams", "One Direction"),
        ("Sweet Dreams (Are Made of This)", "Eurythmics"),
        ("Dreams", "Grouplove"),
        ("California Dreamin'", "The Mamas & The Papas"),
        ("Empire State of Mind", "Jay-Z"),
    ],
    "songs about cats": [
        ("Cat's in the Cradle", "Harry Chapin"),
        ("Kitten", "SALES"),
        ("Like a Cat", "f(x)"),
        ("Mr. Mistoffelees", "Cats Musical"),
        ("Cool Cat", "Queen"),
    ],
    "heartbreak crying moving on": [
        ("ceilings", "Lizzy McAlpine"),
        ("Let Me Down Slowly", "Alec Benjamin"),
        ("cardigan", "Taylor Swift"),
        ("Good 4 U", "Olivia Rodrigo"),
        ("Happier", "Olivia Rodrigo"),
        ("The Night We Met", "Lord Huron"),
        ("Liability", "Lorde"),
    ],
    "summer bangers": [
        ("Espresso", "Sabrina Carpenter"),
        ("Sunshine", "OneRepublic"),
        ("Cruel Summer", "Taylor Swift"),
        ("Summertime Sadness", "Lana Del Rey"),
        ("Watermelon Sugar", "Harry Styles"),
        ("Good as Hell", "Lizzo"),
        ("Levitating", "Dua Lipa"),
    ],
    "nostalgic songs": [
        ("Chasing Cars", "Snow Patrol"),
        ("Counting Stars", "OneRepublic"),
        ("Take On Me", "a-ha"),
        ("Mamma Mia", "ABBA"),
        ("The Less I Know the Better", "Tame Impala"),
        ("Mr. Brightside", "The Killers"),
        ("Video Killed the Radio Star", "The Buggles"),
    ],
    'songs similar to "Blank Space" by Taylor Swift': [
        ("Style", "Taylor Swift"),
        ("Bad Blood", "Taylor Swift"),
        ("Wildest Dreams", "Taylor Swift"),
        ("Clean", "Taylor Swift"),
        ("Look What You Made Me Do", "Taylor Swift"),
        ("7 Rings", "Ariana Grande"),
        ("Boyfriend", "Ariana Grande"),
    ],
    "angry breakup": [
        ("10 Things I Hate About You", "Leah Kate"),
        ("good 4 u", "Olivia Rodrigo"),
        ("You Oughta Know", "Alanis Morissette"),
        ("Before He Cheats", "Carrie Underwood"),
        ("Picture to Burn", "Taylor Swift"),
        ("Misery Business", "Paramore"),
        ("Fighter", "Christina Aguilera"),
    ],
    "songs about vienna": [
        ("Vienna", "Billy Joel"),
        ("Vienna Calling", "Falco"),
        ("The Sound of Music", "Julie Andrews"),
        ("Budapest", "George Ezra"),
        ("Suitcase", "Brandi Carlile"),
    ],
}


def find_song(corpus, title, artist):
    title_lower  = title.lower().strip()
    artist_lower = artist.lower().strip()
    for i, doc in enumerate(corpus):
        if doc["title"].lower().strip() == title_lower and artist_lower in doc["artist"].lower():
            return i
    for i, doc in enumerate(corpus):
        if doc["title"].lower().strip() == title_lower:
            return i
    return None


def main():
    print("Loading corpus...")
    with open(ST_CORPUS_PICKLE, "rb") as f:
        corpus = pickle.load(f)
    print(f"Loaded {len(corpus)} documents\n")

    results = []
    total = 0
    found_count = 0
    not_found = []

    for query, songs in EVAL_SET.items():
        print(f"\n── {query} ──")
        relevant_ids = []
        for title, artist in songs:
            total += 1
            idx = find_song(corpus, title, artist)
            if idx is not None:
                found_count += 1
                relevant_ids.append(idx)
                print(f"  ✅ {title} — {artist} (id: {idx})")
            else:
                not_found.append((query, title, artist))
                print(f"  ❌ {title} — {artist}")
        results.append({"query": query, "relevant_ids": relevant_ids})

    print(f"\n{'='*50}")
    print(f"Found: {found_count}/{total} songs")
    if not_found:
        print("\nNot found:")
        for q, t, a in not_found:
            print(f"  '{t}' by {a}  [{q}]")

    with open("evaluation/queries_personal.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to evaluation/queries_personal.json")


if __name__ == "__main__":
    main()