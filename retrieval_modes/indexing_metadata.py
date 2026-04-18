import os
import sqlite3
 
import ijson
 
CORPUS_PATH = "data/processed/corpus.json"
DB_PATH     = "data/processed/metadata.db"
 
 
def build_metadata_db() -> None:
    if os.path.exists(DB_PATH):
        print("Database already exists, skipping.")
        return
 
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"Corpus not found at {CORPUS_PATH}.")
 
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
 
    cur.execute("""
        CREATE TABLE IF NOT EXISTS songs (
            id     INTEGER PRIMARY KEY,
            title  TEXT,
            artist TEXT,
            genre  TEXT,
            year   TEXT
        )
    """)
 
    print("Streaming corpus and building metadata DB...")
    count = 0
    with open(CORPUS_PATH, "rb") as f:
        for i, doc in enumerate(ijson.items(f, "item")):
            cur.execute(
                "INSERT INTO songs (id, title, artist, genre, year) VALUES (?, ?, ?, ?, ?)",
                (i, doc.get("title", ""), doc.get("artist", ""), doc.get("genre", ""), doc.get("year", ""))
            )
            count += 1
            if count % 100_000 == 0:
                conn.commit()
                print(f"  ...inserted {count:,} songs")
 
    conn.commit()
    conn.close()
    print(f"Saved metadata.db with {count:,} songs")
 
 
if __name__ == "__main__":
    build_metadata_db()

