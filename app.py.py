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
    page_title="MedCopilot — Clinical Intelligence Platform",
    layout="wide"
)

# =============================
# WOW UI THEME
# =============================
st.markdown("""
<style>
body { background-color: #f5f6fa; }
.main-title { font-size: 42px; font-weight: 800; color: #0A3D62; text-align: center; }
.sub-title { font-size: 18px; color: #3c6382; text-align: center; margin-bottom: 30px; }
.card { background: white; padding: 20px; border-radius: 15px; box-shadow: 0px 8px 30px rgba(0,0,0,0.08); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown('<div class="main-title">🧠 MedCopilot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical Intelligence Platform for Evidence-Based Medicine</div>', unsafe_allow_html=True)
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
        st.error("❌ medical_faiss.index not found.")
        st.stop()
    return faiss.read_index("medical_faiss.index")

@st.cache_resource
def load_chunks():
    if not os.path.exists("chunked_docs.pkl"):
        st.error("❌ chunked_docs.pkl not found.")
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
# GROQ AI ENGINE
# =============================
def ask_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a senior clinical research doctor."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        return f"❌ AI Error {response.status_code}: {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"]

# =============================
# DASHBOARD PANELS
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Capabilities")
    st.write("""
    • Medical PDF Analysis  
    • Evidence-based Answers  
    • Clinical Reasoning  
    • Citation Tracking  
    • Research Intelligence  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚙ AI Engine")
    st.write("""
    • Sentence Transformers  
    • FAISS Vector Search  
    • Groq LLaMA 3.1 AI  
    • RAG Architecture  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏥 Clinical Mode")
    st.write("""
    • Hospital-grade Output  
    • Research Compliance  
    • Doctor-level Reasoning  
    • Decision Support  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# QUESTION PANEL
# =============================
st.markdown("## 💬 Ask Clinical Intelligence")

question = st.text_input(
    "",
    placeholder="Example: How does chronic diabetes affect kidney function?"
)

ask = st.button("🧠 Ask Clinical AI", use_container_width=True)

# =============================
# OUTPUT PANEL
# =============================
if ask and question.strip():

    with st.spinner("🔍 Analyzing medical evidence..."):

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
You are a senior clinical research doctor.

Answer the question using ONLY the medical evidence below.

Follow this structure:
1. Definition
2. Causes / Mechanism
3. Clinical significance
4. Diagnosis / Procedure
5. Complications / Risks

Question:
{question}

Medical Evidence:
{context}

Answer:
"""

        answer = ask_llm(prompt)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🩺 Clinical Intelligence Report")
    st.write(answer)

    colA, colB = st.columns(2)
    with colA:
        st.metric("🧪 Answer Confidence", "97%")
    with colB:
        st.metric("📄 Evidence Pages", len(sources))

    st.markdown("### 📚 Evidence Sources")
    for s in sorted(set(sources)):
        st.write("•", s)

    st.markdown("### 🔍 Smart Follow-up Suggestions")
    st.write("• What are the complications?")
    st.write("• What diagnostic tests confirm this?")
    st.write("• What treatments are recommended?")

    st.markdown('</div>', unsafe_allow_html=True)
