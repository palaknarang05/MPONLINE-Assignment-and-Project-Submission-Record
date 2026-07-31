"""
app.py

RAG Chatbot - Streamlit UI.
Retrieves relevant chunks from the FAISS index built by ingest.py,
then queries a local Ollama model with the retrieved context to
produce a grounded answer.

Run:
    ollama pull llama3.2        (once, locally)
    python ingest.py            (build index)
    streamlit run app.py

Author: Palak Narang
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("chunks.pkl")
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
TOP_K = 3


@st.cache_resource
def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        chunks = pickle.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, chunks, embedder


def retrieve(query, index, chunks, embedder, k=TOP_K):
    query_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def query_ollama(prompt, model=OLLAMA_MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("🤖 RAG Chatbot")
    st.caption("Retrieval-Augmented Generation over local documents, powered by Ollama.")

    index, chunks, embedder = load_index()
    if index is None:
        st.error("No index found. Run `python ingest.py` first to build the index from ./documents.")
        return

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    query = st.chat_input("Ask a question about your documents...")
    if query:
        st.session_state.history.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        retrieved = retrieve(query, index, chunks, embedder)
        prompt = build_prompt(query, retrieved)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = query_ollama(prompt)
                except requests.exceptions.ConnectionError:
                    answer = (
                        "Could not reach Ollama at localhost:11434. "
                        "Make sure `ollama serve` is running and the model is pulled."
                    )
                st.markdown(answer)
                with st.expander("Sources"):
                    for c in retrieved:
                        st.markdown(f"**{c['source']}**: {c['text'][:200]}...")

        st.session_state.history.append(("assistant", answer))


if __name__ == "__main__":
    main()
