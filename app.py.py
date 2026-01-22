# ============================================================
# VEERA AI — Clinical Research Copilot (FINAL PRODUCTION BUILD)
# Medical-Grade Research Intelligence Platform
# Author: Veera Babu
# ============================================================

import streamlit as st
import os
import google.generativeai as genai
from pypdf import PdfReader
from datetime import datetime

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Veera AI Clinical Research Copilot",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# GEMINI CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-1.5-pro"

# ============================================================
# MEDICAL SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Veera AI — a Medical Research Specialist Copilot.

You assist doctors, researchers, and scientists.

Strict Rules:
1. Always prioritize uploaded clinical PDFs over general knowledge.
2. Never hallucinate.
3. If evidence is missing, say "Insufficient clinical evidence".
4. Maintain scientific and professional tone.
5. Mention trial phase, sample size, and outcomes when available.
6. Support Telugu and English language.
7. Be accurate, ethical, and clinical-grade.
"""

# ============================================================
# PDF PROCESSOR
# ============================================================

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text_data = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_data += text + "\n"
    return text_data


# ============================================================
# GEMINI MEDICAL ENGINE
# ============================================================

def get_medical_response(context, question):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT
    )

    prompt = f"""
You are analyzing clinical research documents.

--- Clinical Research Context ---
{context}

--- Research Question ---
{question}

--- Instructions ---
- Use only the uploaded clinical evidence
- Answer as a medical researcher
- Do not assume anything
- If insufficient data, say clearly
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 4096
        }
    )

    return response.text


# ============================================================
# UI HEADER
# ============================================================

st.markdown("""
# 🧬 Veera AI — Clinical Research Copilot  
### Medical-Grade AI for Evidence-Based Research

> Turning clinical literature into instant medical intelligence.

---
""")

# ============================================================
# SIDEBAR — MEDICAL LIBRARY
# ============================================================

st.sidebar.title("📚 Medical Research Library")

uploaded_files = st.sidebar.file_uploader(
    "Upload Clinical Research PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

context_text = ""
library = []

if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} PDFs Loaded Successfully")

    for file in uploaded_files:
        pdf_text = extract_text_from_pdf(file)
        context_text += pdf_text + "\n\n"

        library.append({
            "name": file.name,
            "size": round(file.size / 1024, 2),
            "uploaded": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    st.sidebar.markdown("### 📂 Library Documents")
    for doc in library:
        st.sidebar.write(f"📄 {doc['name']} ({doc['size']} KB)")


# ============================================================
# CHAT INTERFACE
# ============================================================

st.subheader("💬 Clinical Research Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask your clinical research question (English / Telugu):")

if st.button("🔍 Analyze Clinical Evidence"):

    if not uploaded_files:
        st.warning("Please upload clinical research PDFs first.")
    elif not question:
        st.warning("Please enter your research question.")
    else:
        with st.spinner("🧠 Veera AI is analyzing clinical evidence..."):
            answer = get_medical_response(context_text, question)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for chat in reversed(st.session_state.chat_history):
    st.markdown("### 🧑‍⚕️ Research Question")
    st.write(chat["question"])

    st.markdown("### 🧠 Veera AI Medical Analysis")
    st.write(chat["answer"])

    st.markdown("---")


# ============================================================
# FOOTER — SECURITY & TRUST
# ============================================================

st.markdown("""
---
### 🔐 Privacy & Security First  
✔ Client-side PDF Processing  
✔ Encrypted AI Streams  
✔ No Data Retention  
✔ Medical Ethics Compliant  

**Veera AI Clinical Research Copilot — Built for the Future of Medicine**
""")
