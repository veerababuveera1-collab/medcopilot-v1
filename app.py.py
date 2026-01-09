import streamlit as st
import os
import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="MedCopilot V2 — Clinical Research Copilot",
    layout="wide"
)

st.title("🧠 MedCopilot V2 — Clinical Research Copilot")
st.caption("Evidence-based medical Q&A (Global Standard AI)")
st.warning("⚠ This AI system is for research support only. Not medical advice.")

# =============================
# LOAD EMBEDDING MODEL
# =============================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

@st.cache_resource
def load_faiss_index():
    if not os.path.exists("medical_faiss.index"):
        st.error("❌ medical_faiss.index not found in repository.")
        st.stop()
    return faiss.read_index("medical_faiss.index")

@st.cache_resource
def load_chunks():
    if not os.path.exists("chunked_docs.pkl"):
        st.error("❌ chunked_docs.pkl not found in repository.")
        st.stop()
    with open("chunked_docs.pkl", "rb") as f:
        return pickle.load(f)

embedding_model = load_embedding_model()
index = load_faiss_index()
chunked_docs = load_chunks()

# =============================
# LOAD API KEY
# =============================
HF_API_KEY = (
    st.secrets.get("HF_API_KEY") or
    st.secrets.get("HF_TOKEN") or
    os.environ.get("HF_API_KEY") or
    os.environ.get("HF_TOKEN")
)

if not HF_API_KEY:
    st.error("❌ HuggingFace API key not found. Add HF_API_KEY in Streamlit Secrets.")
    st.stop()

# =============================
# HUGGINGFACE TEXT GENERATION API (FREE TIER)
# =============================
def ask_llm(prompt: str) -> str:
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.2
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)

        if response.status_code == 503:
            return "⚠ AI server is busy. Please try again in 30 seconds."

        if response.status_code == 429:
            return "⚠ API rate limit reached. Please try again later."

        if response.status_code != 200:
            return f"❌ AI Error {response.status_code}: {response.text}"

        result = response.json()

        if isinstance(result, list):
            return result[0]["generated_text"]
        else:
            return result.get("generated_text", "No answer generated.")

    except requests.exceptions.Timeout:
        return "⚠ AI request timed out. Please try again."

    except Exception as e:
        return f"❌ AI Connection Error: {str(e)}"

# =============================
# USER INPUT
# =============================
question = st.text_input(
    "💬 Ask a clinical research question:",
    placeholder="Example: What are the symptoms and causes of asthma?"
)

# =============================
# ASK BUTTON
# =============================
if st.button("Ask MedCopilot") and question.strip():

    with st.spinner("🔍 Searching medical knowledge..."):

        q_embedding = embedding_model.encode([question])
        distances, indices = index.search(np.array(q_embedding), 5)

        context = ""
        sources = []

        for idx in indices[0]:
            chunk = chunked_docs[idx]
            context += chunk["text"] + "\n"
            sources.append(
                f'{chunk["metadata"]["source"]} (page {chunk["metadata"]["page"]})'
            )

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

        answer = ask_llm(prompt)

    # =============================
    # DISPLAY OUTPUT
    # =============================
    st.subheader("✅ Clinical Answer")
    st.write(answer)

    st.subheader("📚 Evidence Sources")
    for s in sorted(set(sources)):
        st.write("•", s)
