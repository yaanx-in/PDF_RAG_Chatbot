"""PDF-Based AI Doubt-Solving Chatbot using RAG.

Upload a PDF, then ask questions answered strictly from that document.
Run with:  streamlit run app.py
"""

import re

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"      # free, offline (~80 MB, downloaded once)
LLM_MODEL = "llama3.2:3b"               # local model served via Ollama (optional)
GEMINI_MODEL = "gemini-flash-latest"    # -latest alias: auto-tracks current flash, survives model retirements
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_MODEL = "openai/gpt-oss-120b"      # free hosted open-weights LLM (Groq)
GROQ_BASE = "https://api.groq.com/openai/v1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
REFUSAL = "This is not covered in the document."


@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


# ---------- Module 2: extraction ----------
def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    return [{"page": n, "text": (p.extract_text() or "")}
            for n, p in enumerate(reader.pages, start=1)]


# ---------- Module 3: clean + chunk ----------
def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Sliding character window with overlap (Appendix B algorithm)."""
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step)
            if text[i:i + size].strip()]


def make_chunks(pages):
    chunks = []
    for p in pages:
        for i, piece in enumerate(split_text(clean(p["text"]))):
            chunks.append({"id": f'{p["page"]}-{i}',
                           "page": p["page"], "text": piece})
    return chunks


# ---------- Module 4: embed + index ----------
def build_index(chunks):
    embedder = get_embedder()
    vecs = embedder.encode([c["text"] for c in chunks],
                           normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])     # cosine over normalized vecs
    index.add(np.array(vecs, dtype="float32"))
    return index, chunks


# ---------- Module 5: retrieve ----------
def retrieve(question, index, chunks, k=TOP_K):
    embedder = get_embedder()
    v = embedder.encode([question], normalize_embeddings=True)
    scores, ids = index.search(np.array(v, dtype="float32"), k)
    return [chunks[i] for i in ids[0] if i != -1], scores[0]


# ---------- Module 6: generate ----------
PROMPT = """You are a study assistant. Answer the question using ONLY the
context below. If the answer is not in the context, reply exactly:
"{refusal}"

Context:
{context}

Question: {question}
Answer:"""


def _gemini_answer(prompt):
    """Free hosted LLM (Google Gemini). Returns None if no key is configured."""
    import os
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")    # cloud: set in app Secrets
    except Exception:
        pass
    key = key or os.environ.get("GEMINI_API_KEY")  # local: export the env var
    if not key:
        return None
    from openai import OpenAI
    client = OpenAI(base_url=GEMINI_BASE, api_key=key)
    resp = client.chat.completions.create(
        model=GEMINI_MODEL, temperature=0.2,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def _groq_answer(prompt):
    """Free hosted open-weights LLM (Groq). Returns None if no key is configured."""
    import os
    key = None
    try:
        key = st.secrets.get("GROQ_API_KEY")      # cloud: set in app Secrets
    except Exception:
        pass
    key = key or os.environ.get("GROQ_API_KEY")   # local: export the env var
    if not key:
        return None
    from openai import OpenAI
    client = OpenAI(base_url=GROQ_BASE, api_key=key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.2,
        messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def _ollama_answer(prompt):
    """Local LLM via Ollama. Returns None if not available."""
    try:
        import ollama
        r = ollama.chat(model=LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"]
    except Exception:
        return None


def generate_answer(question, hits):
    if not hits:
        return REFUSAL
    ctx = "\n\n".join(f'[p.{c["page"]}] {c["text"]}' for c in hits)
    prompt = PROMPT.format(refusal=REFUSAL, context=ctx, question=question)
    for backend in (_groq_answer, _gemini_answer, _ollama_answer):  # Groq -> Gemini -> Ollama -> extractive
        try:
            ans = backend(prompt)
        except Exception:
            ans = None
        if ans:
            return ans
    # ponytail: no LLM at all -> extractive fallback so the app still runs
    # end to end. Set GEMINI_API_KEY (cloud) or run Ollama (local) for phrased answers.
    top = hits[0]
    return (f"(No LLM configured — showing the most relevant passage.)\n\n"
            f'{top["text"]}\n\n(from p.{top["page"]})')


# ---------- Modules 1 & 7: UI + chat ----------
st.set_page_config(page_title="PDF Doubt-Solving Chatbot (RAG)", page_icon="📄")
st.title("PDF Doubt-Solving Chatbot (RAG)")

pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf and st.session_state.get("pdf_name") != pdf.name:
    with st.spinner("Indexing your PDF..."):
        chunks = make_chunks(extract_text(pdf))
        if not chunks:
            st.error("No extractable text found — is this a scanned/image PDF?")
            st.stop()
        st.session_state.index, st.session_state.chunks = build_index(chunks)
        st.session_state.history = []
        st.session_state.pdf_name = pdf.name
    st.success(f"Ready! Indexed {len(chunks)} chunks. Ask your doubt below.")

if "index" in st.session_state:
    for q, a in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    question = st.chat_input("Ask a question about your PDF...")
    if question:
        hits, _ = retrieve(question, st.session_state.index,
                           st.session_state.chunks)
        answer = generate_answer(question, hits)
        st.session_state.history.append((question, answer))
        st.rerun()
