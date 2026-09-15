# PDF Doubt-Solving Chatbot (RAG)

A single-file, monolithic Streamlit app. Upload a PDF and ask questions that are
answered **strictly from that document** using Retrieval-Augmented Generation.
If the answer isn't in the PDF, it says so instead of guessing.

Pipeline (all in `app.py`): extract → clean → chunk → embed → FAISS index →
retrieve top-k → grounded LLM answer → chat history.

## Answer backends (auto-selected in this order)

1. **Gemini** (free hosted LLM) — used if `GEMINI_API_KEY` is set. Best for cloud.
2. **Ollama** (local LLM) — used if running locally with Ollama.
3. **Extractive fallback** — returns the most relevant passage; no setup needed.

## Run locally

> Python 3.11–3.13 recommended.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# optional local LLM:  install Ollama, then  ollama pull llama3.2:3b
# or use Gemini:        export GEMINI_API_KEY=...
streamlit run app.py            # http://localhost:8501
```

## Deploy free on Streamlit Community Cloud

1. Get a free API key at <https://aistudio.google.com/apikey>.
2. Push this repo to GitHub.
3. Go to <https://share.streamlit.io>, connect the repo, set main file `app.py`.
4. In **Settings → Secrets**, add:
   ```
   GEMINI_API_KEY = "..."
   ```
5. Deploy → public `https://<you>.streamlit.app` URL.

If the ~1 GB RAM limit trips on boot (torch), deploy the same repo on
**Hugging Face Spaces** (free CPU, 16 GB RAM) instead — no code change.

## Config (top of `app.py`)

| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Larger = more context, blurrier retrieval |
| `CHUNK_OVERLAP` | 100 | Protects ideas at chunk boundaries |
| `TOP_K` | 4 | How many passages feed the answer |
| `EMBED_MODEL` | all-MiniLM-L6-v2 | Embedding model |
| `GEMINI_MODEL` | gemini-2.0-flash | Hosted LLM (cloud) |
| `LLM_MODEL` | llama3.2:3b | Local Ollama model |
