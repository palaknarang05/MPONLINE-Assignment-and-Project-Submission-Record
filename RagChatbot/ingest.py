"""
ingest.py

Builds a FAISS vector index from documents in ./documents.
Splits text into overlapping chunks, embeds with sentence-transformers,
saves the index + chunk metadata for retrieval at query time.

Author: Palak Narang
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("documents")
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("chunks.pkl")

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_documents(docs_dir: Path):
    texts = []
    for path in docs_dir.glob("**/*"):
        if path.suffix.lower() in (".txt", ".md"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append((path.name, f.read()))
    return texts


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index():
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No .txt/.md files found in {DOCS_DIR}/")

    model = SentenceTransformer(EMBED_MODEL)

    all_chunks = []
    for filename, text in docs:
        for chunk in chunk_text(text):
            if chunk.strip():
                all_chunks.append({"source": filename, "text": chunk})

    print(f"Loaded {len(docs)} documents -> {len(all_chunks)} chunks")

    embeddings = model.encode(
        [c["text"] for c in all_chunks],
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Saved index -> {INDEX_PATH}, metadata -> {META_PATH}")


if __name__ == "__main__":
    build_index()
