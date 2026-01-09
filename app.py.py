import streamlit as st
import os
import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="MedCopilot V2 — Clinical Research Copilot",
    layout="wide"
)

st.title("🧠 MedCopilot V2 — Clinical Research Copilot")
st.caption("Evidence-based medical Q&A (Cloud-Safe, Low-Memory)")
st.warning("⚠ This AI system is for research support only. Not medical advice.")

# -----------------------------
# LOAD LIGHTWEIGHT EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embedding_model():
    # Lightweight model for Streamlit Cloud
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_faiss_index():
    if not os.path.exists("medical_faiss.index"):
        st.error("❌ medical_faiss.index not found in repo.")
        st.stop()
    return faiss.read_index("medical_faiss.index")

@st.cache_resource
def load_chunks():
    if not os.path.exists("chunked_docs.pkl"):
        st.error("❌ chunked_docs.pkl not found in repo.")
        st.stop()
    with open("chunked_docs.pkl", "rb") as f:
        return pickle.load(f)

embedding_model = load_embedding_model()
index = load_faiss_index()
chunked_docs = load_chunks()

# -----------------------------
# LLM VIA API (NO LOCAL MODEL)
# -----------------------------
# Add this in Streamlit Cloud → App Settings → Secrets:
# HF_API_KEY = "your_huggingface_api_key"
HF_API_KEY = st.secrets.get("HF_API_KEY", "")

if not HF_API_KEY:
    st.info("🔐 Add your HF_API_KEY in Streamlit Secrets to enable answers.")

def ask_llm(prompt: str) -> str:
    """Call HuggingFace Inference API (no local model load)."""
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data[0]["generated_text"]

# -----------------------------
# UI — QUESTION INPUT
# -----------------------------
question = st.text_input(
    "💬 Ask a clinical research question:",
    placeholder="Example: What are the symptoms and causes of diabetes?"
)

# -----------------------------
# ASK BUTTON
# -----------------------------
if st.button("Ask MedCopilot") and question.strip():

    if not HF_API_KEY:
        st.error("❌ Missing HF_API_KEY. Add it in Streamlit Secrets.")
        st.stop()

    with st.spinner("🔍 Searching medical knowledge..."):

        # Vector search
        q_embedding = embedding_model.encode([question])
        distances, indices = index.search(np.array(q_embedding), 5)

        # Build context from top-k chunks
        context = ""
        sources = []

        for idx in indices[0]:
            chunk = chunked_docs[idx]
            context += chunk["text"] + "\n"
            sources.append(f'{chunk["metadata"]["source"]} (page {chunk["metadata"]["page"]})')

        # LLM prompt (strict: use only context)
        prompt = f"""
You are a medical research assistant.
Answer the question using ONLY the context below.
Use simple English.
Do not add new information.

Question:
{question}

Context:
{context}

Answer:
"""

        try:
            answer = ask_llm(prompt)
        except Exception as e:
            st.error(f"LLM API error: {e}")
            st.stop()

    # -----------------------------
    # DISPLAY OUTPUT
    # -----------------------------
    st.subheader("✅ Clinical Answer")
    st.write(answer)

    st.subheader("📚 Evidence Sources")
    for s in sorted(set(sources)):
        st.write("•", s)
