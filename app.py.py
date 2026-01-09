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
st.caption("Evidence-based medical Q&A (Groq Powered AI)")
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
# LOAD GROQ API KEY
# =============================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ Groq API key not found. Add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# =============================
# GROQ CHAT API (NEW MODEL)
# =============================
def ask_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",   # ✅ new supported model
        "messages": [
            {"role": "system", "content": "You are a clinical research assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 429:
            return "⚠ API rate limit reached. Please try again later."

        if response.status_code != 200:
            return f"❌ AI Error {response.status_code}: {response.text}"

        data = response.json()
        return data["choices"][0]["message"]["content"]

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
        _, indices = index.search(np.array(q_embedding), 5)

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
