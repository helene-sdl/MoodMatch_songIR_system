import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python main.py [streamlit|bm25|st|faiss|index|graph]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "streamlit":
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])

    elif mode == "bm25":
        from retrieval_modes.BM25_retrieval import main as bm25_main
        bm25_main()

    elif mode == "st":
        from retrieval_modes.SentenceTransformer_IR import main as st_main
        st_main()

    elif mode == "faiss":
        from retrieval_modes.faiss_indexing import main as faiss_main
        faiss_main()

    elif mode == "index":
        from retrieval_modes.indexing_metadata import build_metadata_db
        build_metadata_db()

    elif mode == "graph":
        from retrieval_modes.knowledge_graph import main as graph_main
        graph_main()

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()