# RAG Chatbot (Capstone Project)

A Retrieval-Augmented Generation chatbot: answers questions grounded in a
local document collection, using FAISS for retrieval and a local Ollama LLM
for generation (no external API calls, documents never leave the machine).

## Developer Info
**Name:** Palak Narang  
**Registration Number:** 23BCE11819  
**Application Number:** IN26011657  
**Batch:** 1A  
**Department:** Computer Science and Engineering  
**Institution:** VIT Bhopal University



## Tech Stack
Python, Streamlit, Sentence-Transformers (`all-MiniLM-L6-v2`), FAISS, Ollama

## Project Structure
```
RagChatbot/
├── documents/            # source .txt/.md files to index
│   ├── rag_overview.txt
│   └── faiss_notes.txt
├── ingest.py             # chunks + embeds documents, builds FAISS index
├── app.py                # Streamlit chat UI, retrieval + Ollama generation
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt

# pull a local model once
ollama pull llama3.2
ollama serve   # if not already running
```

## Usage
```bash
# 1. Build the vector index from ./documents
python ingest.py

# 2. Launch the chatbot
streamlit run app.py
```

## How It Works
1. **Ingestion** (`ingest.py`): documents are split into overlapping 500-char
   chunks, embedded with `all-MiniLM-L6-v2`, and stored in a FAISS
   `IndexFlatL2` index (`faiss_index.bin` + `chunks.pkl`).
2. **Retrieval** (`app.py`): the user's question is embedded and the top-3
   nearest chunks are pulled from the index.
3. **Generation**: retrieved chunks are inserted into a prompt template and
   sent to a local Ollama model (`llama3.2`) via its REST API. The model is
   instructed to answer only from the given context, reducing hallucination.
4. Sources for each answer are shown in an expandable panel in the UI.

## Notes
- Add your own `.txt`/`.md` files to `documents/` and re-run `ingest.py` to
  index new material.
- Swap `OLLAMA_MODEL` in `app.py` for any locally pulled Ollama model.
