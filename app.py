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


# ---------- streaming + follow-ups ----------
def _get_key(name):
    import os
    try:
        k = st.secrets.get(name)
    except Exception:
        k = None
    return k or os.environ.get(name)


def answer_stream(question, hits):
    """Yield the answer token-by-token so the UI never looks frozen."""
    if not hits:
        yield REFUSAL
        return
    ctx = "\n\n".join(f'[p.{c["page"]}] {c["text"]}' for c in hits)
    prompt = PROMPT.format(refusal=REFUSAL, context=ctx, question=question)
    gk = _get_key("GROQ_API_KEY")
    if gk:
        try:
            from openai import OpenAI
            stream = OpenAI(base_url=GROQ_BASE, api_key=gk).chat.completions.create(
                model=GROQ_MODEL, temperature=0.2, stream=True,
                messages=[{"role": "user", "content": prompt}])
            streamed = False
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    streamed = True
                    yield delta
            if streamed:
                return
        except Exception:
            pass
    # Fallback (Gemini/Ollama/extractive) — reveal word-by-word for a live feel.
    for word in generate_answer(question, hits).split(" "):
        yield word + " "


def suggest_followups(question, answer):
    """Three short follow-up questions. Best-effort; empty on any failure."""
    gk = _get_key("GROQ_API_KEY")
    if not gk:
        return []
    try:
        from openai import OpenAI
        r = OpenAI(base_url=GROQ_BASE, api_key=gk).chat.completions.create(
            model=GROQ_MODEL, temperature=0.6,
            messages=[{"role": "user", "content": (
                f'A user asked: "{question}"\nThe assistant answered: "{answer[:600]}"\n'
                "Suggest 3 short, distinct follow-up questions about the SAME document. "
                "Return ONLY the questions, one per line, no numbering.")}])
        lines = [re.sub(r"^[\-\d\.\)\s]+", "", ln).strip()
                 for ln in r.choices[0].message.content.splitlines()]
        return [ln for ln in lines if ln][:3]
    except Exception:
        return []


# ---------- Modules 1 & 7: UI + chat ----------
st.set_page_config(page_title="PDF Doubt-Solving Chatbot", page_icon="📖",
                   layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
:root{
  --bg:#F4F1EA; --surface:#FCFBF7; --ink:#1A1A18; --muted:#6F6B62;
  --clay:#CC785C; --clay-soft:#F1E5DE; --border:#E5DFD3;
}
.stApp{ background:var(--bg); }
html, body, [class*="css"], .stMarkdown, p, li, span, div{ font-family:'Inter',system-ui,sans-serif; }
body{ color:var(--ink); }
/* strip default chrome */
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stDecoration"]{ display:none !important; }
.block-container{ max-width:820px; padding-top:2rem; padding-bottom:7rem; }
/* hero */
.hero-title{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:2.7rem;
  line-height:1.08; letter-spacing:-0.025em; margin:0 0 .35rem; color:var(--ink); }
.hero-title .accent{ color:var(--clay); font-style:italic; }
.hero-sub{ color:var(--muted); font-size:1.04rem; line-height:1.5; margin:0 0 1.6rem; max-width:34rem; }
/* uploader */
[data-testid="stFileUploaderDropzone"]{ background:var(--surface); border:1.5px dashed var(--border);
  border-radius:16px; transition:border-color .15s; }
[data-testid="stFileUploaderDropzone"]:hover{ border-color:var(--clay); }
/* chat message: assistant becomes a warm card */
[data-testid="stChatMessage"]{ background:transparent; padding:.35rem 0; gap:.7rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]){
  background:var(--surface); border:1px solid var(--border); border-radius:18px;
  padding:1.05rem 1.2rem; box-shadow:0 1px 2px rgba(26,26,24,.03); }
[data-testid="stChatMessage"] p{ font-size:1.0rem; line-height:1.62; }
/* chat input */
[data-testid="stChatInput"]{ background:var(--bg); }
[data-testid="stChatInput"] textarea{ font-family:'Inter',sans-serif; }
[data-testid="stChatInput"] > div{ border-radius:14px; border-color:var(--border); background:var(--surface); }
/* chips / buttons */
.stButton>button{ background:var(--surface); border:1px solid var(--border); color:var(--ink);
  border-radius:999px; padding:.42rem .95rem; font-size:.87rem; font-weight:500;
  transition:all .15s; box-shadow:none; }
.stButton>button:hover{ border-color:var(--clay); color:var(--clay); background:var(--clay-soft); }
.stButton>button:focus{ box-shadow:none; color:var(--clay); border-color:var(--clay); }
/* expander (sources) */
[data-testid="stExpander"]{ border:1px solid var(--border); border-radius:12px; background:var(--bg); }
[data-testid="stExpander"] summary{ font-size:.86rem; color:var(--muted); font-weight:500; }
/* status box */
[data-testid="stStatusWidget"], [data-testid="stStatus"]{ border-radius:12px; }
/* section label */
.eyebrow{ font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:var(--clay);
  font-weight:600; margin:.2rem 0 .6rem; }
</style>
""", unsafe_allow_html=True)

USER_AV, BOT_AV = "🧑", "📖"

st.markdown(
    '<div class="hero-title">Ask your PDF <span class="accent">anything</span></div>'
    '<div class="hero-sub">Upload a document and get grounded, cited answers — '
    'streamed live, drawn only from what\'s actually written inside it.</div>',
    unsafe_allow_html=True)

pdf = st.file_uploader("Upload your PDF", type="pdf", label_visibility="collapsed")

if pdf and st.session_state.get("pdf_name") != pdf.name:
    with st.status("Preparing your document…", expanded=True) as status:
        st.write("Extracting text…")
        pages = extract_text(pdf)
        st.write("Cleaning & chunking…")
        chunks = make_chunks(pages)
        if not chunks:
            status.update(label="No extractable text found", state="error")
            st.error("This looks like a scanned/image PDF — no machine-readable text.")
            st.stop()
        st.write(f"Embedding {len(chunks)} chunks & building the search index…")
        st.session_state.index, st.session_state.chunks = build_index(chunks)
        st.session_state.messages = []
        st.session_state.pdf_name = pdf.name
        status.update(label=f"Ready · {len(chunks)} chunks indexed from {pdf.name}",
                      state="complete", expanded=False)

if "index" in st.session_state:
    msgs = st.session_state.setdefault("messages", [])

    # ---- render history ----
    for m in msgs:
        with st.chat_message(m["role"], avatar=USER_AV if m["role"] == "user" else BOT_AV):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                srcs = m["sources"]
                with st.expander(f"📄 {len(srcs)} source passage(s)"):
                    for s in srcs:
                        pct = int(s.get("score", 0) * 100)
                        st.markdown(f"**p.{s['page']}** · {pct}% match  \n{s['text'][:420]}…")

    # ---- suggested starters (empty state) or follow-up chips (after last answer) ----
    pending = st.session_state.pop("pending_q", None)
    if not msgs:
        st.markdown('<div class="eyebrow">Try asking</div>', unsafe_allow_html=True)
        starters = ["What is this document about?", "Summarize the key points",
                    "What are the main takeaways?"]
        cols = st.columns(len(starters))
        for c, s in zip(cols, starters):
            if c.button(s, key=f"start_{s}"):
                pending = s
    elif msgs[-1]["role"] == "assistant" and msgs[-1].get("followups"):
        st.markdown('<div class="eyebrow">Follow up</div>', unsafe_allow_html=True)
        cols = st.columns(len(msgs[-1]["followups"]))
        for i, (c, f) in enumerate(zip(cols, msgs[-1]["followups"])):
            if c.button(f, key=f"fu_{len(msgs)}_{i}"):
                pending = f

    typed = st.chat_input("Ask a question about your PDF…")
    question = typed or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AV):
            st.markdown(question)
        with st.chat_message("assistant", avatar=BOT_AV):
            with st.status("Searching the document…", expanded=False) as status:
                raw_hits, scores = retrieve(question, st.session_state.index,
                                            st.session_state.chunks)
                hits = [{**h, "score": float(sc)} for h, sc in zip(raw_hits, scores)]
                status.update(label="Answering…", state="complete")
            answer = st.write_stream(answer_stream(question, hits))
            if hits:
                with st.expander(f"📄 {len(hits)} source passage(s)"):
                    for s in hits:
                        pct = int(s["score"] * 100)
                        st.markdown(f"**p.{s['page']}** · {pct}% match  \n{s['text'][:420]}…")
        followups = suggest_followups(question, answer)
        st.session_state.messages.append({"role": "assistant", "content": answer,
                                          "sources": hits, "followups": followups})
        st.rerun()
else:
    st.markdown('<div class="eyebrow">Get started</div>', unsafe_allow_html=True)
    st.caption("Drop a text-based PDF above — a study guide, report, or manual — "
               "then ask questions in plain language.")
