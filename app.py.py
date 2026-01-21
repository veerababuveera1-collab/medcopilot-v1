# ============================================================
# MEDNEXUS AI — Clinical Research Intelligence Agent
# Literature • Trials • FDA • Evidence • Citations
# ============================================================

import streamlit as st
import os, io, json, requests, datetime
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="MEDNEXUS AI — Clinical Research Intelligence",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MEDNEXUS AI — Clinical Research Intelligence Agent")
st.caption("Medical Literature • Clinical Trials • FDA Intelligence • Evidence AI")

st.info(
    "ℹ MEDNEXUS AI is a Clinical Research Intelligence Platform. "
    "It supports evidence discovery and trial research. "
    "Final medical decisions must be made by qualified clinicians."
)

# ============================================================
# STORAGE
# ============================================================
BASE = os.getcwd()
DOCS_DIR = os.path.join(BASE, "medical_docs")
VECTOR_DIR = os.path.join(BASE, "vector_db")

INDEX_FILE = os.path.join(VECTOR_DIR, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_DIR, "cache.json")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ============================================================
# PDF PROCESSING
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages[:150]:
        text = page.extract_text()
        if text and len(text) > 100:
            pages.append(text)
    return pages

def build_index():
    texts, sources = [], []

    for pdf in os.listdir(DOCS_DIR):
        if pdf.endswith(".pdf"):
            with open(os.path.join(DOCS_DIR, pdf), "rb") as f:
                pages = extract_text(f.read())
                for i, p in enumerate(pages):
                    texts.append(p)
                    sources.append(f"{pdf} — Page {i+1}")

    if not texts:
        return None, [], []

    embeddings = model.encode(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_FILE)

    cache = {"texts": texts, "sources": sources}
    json.dump(cache, open(CACHE_FILE, "w"), indent=2)

    return index, texts, sources

# ============================================================
# LOAD INDEX
# ============================================================
index, texts, sources = None, [], []

if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
    index = faiss.read_index(INDEX_FILE)
    cache = json.load(open(CACHE_FILE))
    texts = cache["texts"]
    sources = cache["sources"]

# ============================================================
# EXTERNAL MEDICAL APIs
# ============================================================
def fetch_pubmed(query):
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 5}
        r = requests.get(url, params=params, timeout=15)
        return r.json()["esearchresult"]["idlist"]
    except:
        return []

def fetch_trials(query):
    try:
        url = "https://clinicaltrials.gov/api/v2/studies"
        params = {"query.term": query, "pageSize": 5}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        trials = []
        for s in data.get("studies", []):
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            design = proto.get("designModule", {})

            trials.append({
                "Trial ID": ident.get("nctId", ""),
                "Title": ident.get("briefTitle", ""),
                "Phase": ", ".join(design.get("phases", ["N/A"])),
                "Status": status.get("overallStatus", "Unknown")
            })
        return trials
    except:
        return []

def fetch_fda_alerts():
    try:
        url = "https://api.fda.gov/drug/enforcement.json?limit=5"
        r = requests.get(url, timeout=15)
        data = r.json()
        alerts = []
        for item in data.get("results", []):
            alerts.append({
                "Drug": item.get("product_description", ""),
                "Reason": item.get("reason_for_recall", "")
            })
        return alerts
    except:
        return []

# ============================================================
# UI SIDEBAR
# ============================================================
menu = st.sidebar.radio(
    "Research Intelligence Center",
    ["📊 Dashboard", "📁 Evidence Library", "🔍 Research AI Console", "🧪 Trial Intelligence", "⚠ FDA Intelligence"]
)

# ============================================================
# DASHBOARD
# ============================================================
if menu == "📊 Dashboard":
    st.header("📊 Clinical Research Command Center")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Medical PDFs", len(os.listdir(DOCS_DIR)))
    col2.metric("Indexed Pages", len(texts))
    col3.metric("AI Engine", "Online")
    col4.metric("Global Feeds", "Live")

    st.success("All Research Intelligence Systems Operational")

# ============================================================
# EVIDENCE LIBRARY
# ============================================================
if menu == "📁 Evidence Library":
    st.header("📁 Medical Literature Library")

    uploaded_files = st.file_uploader(
        "Upload Medical PDFs (Guidelines, Papers, Protocols)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for pdf in uploaded_files:
            with open(os.path.join(DOCS_DIR, pdf.name), "wb") as f:
                f.write(pdf.getbuffer())
        st.success("PDFs uploaded successfully")

    if st.button("🧠 Build Evidence Index"):
        with st.spinner("Indexing medical literature..."):
            index, texts, sources = build_index()
        st.success("Evidence AI Index Built Successfully")

# ============================================================
# RESEARCH AI CONSOLE
# ============================================================
if menu == "🔍 Research AI Console":
    st.header("🔍 Clinical Research AI Console")

    query = st.text_area("Ask a clinical research question", height=120)

    if st.button("🚀 Run Research Intelligence"):
        if index is None:
            st.error("Evidence index not built yet.")
        else:
            q_emb = model.encode([query]).astype("float32")
            D, I = index.search(q_emb, 3)

            st.subheader("🧠 AI Evidence Answer")

            for i in I[0]:
                st.info(texts[i][:800])
                st.caption(f"📚 Source: {sources[i]}")

            st.divider()

            st.subheader("📚 PubMed Literature")
            pubmed = fetch_pubmed(query)
            st.write(pubmed)

# ============================================================
# TRIAL INTELLIGENCE
# ============================================================
if menu == "🧪 Trial Intelligence":
    st.header("🧪 Clinical Trial Intelligence")

    query = st.text_input("Search clinical trials")

    if st.button("Search Trials"):
        trials = fetch_trials(query)
        if trials:
            st.table(pd.DataFrame(trials))
        else:
            st.info("No trials found.")

# ============================================================
# FDA INTELLIGENCE
# ============================================================
if menu == "⚠ FDA Intelligence":
    st.header("⚠ FDA Drug Safety Intelligence")

    alerts = fetch_fda_alerts()
    if alerts:
        st.table(pd.DataFrame(alerts))
    else:
        st.info("No FDA alerts found.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🧠 MEDNEXUS AI — Clinical Research Intelligence Agent | Evidence • Trials • FDA")
