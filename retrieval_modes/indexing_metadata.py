import json
import sqlite3
import os

CORPUS_PATH = "data/processed/corpus.json"
DB_PATH = "data/processed/metadata.db"

def build_metadata_db():
    if os.path.exists(DB_PATH):
        print("Database already exists, skipping.")
        return

    print("Loading corpus...")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    print(f"Building metadata database for {len(corpus)} songs...")
    os.makedirs("data/processed", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id INTEGER PRIMARY KEY,
            title TEXT,
            artist TEXT,
            genre TEXT,
            year TEXT
        )
    """)

    cur.executemany(
        "INSERT INTO songs (id, title, artist, genre, year) VALUES (?, ?, ?, ?, ?)",
        [
            (i, doc.get("title", ""), doc.get("artist", ""), doc.get("genre", ""), doc.get("year", ""))
            for i, doc in enumerate(corpus)
        ]
    )

    conn.commit()
    conn.close()
    print(f"Saved metadata.db with {len(corpus)} songs")

if __name__ == "__main__":
    build_metadata_db()

