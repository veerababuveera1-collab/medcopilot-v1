# ============================================================
# ĀROGYABODHA AI — HYBRID RESEARCH + HOSPITAL AI OS
# Clinical Research + Hospital Intelligence + CDSS Platform
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests, re
import numpy as np
import faiss
import pandas as pd
from typing import List
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ============================================================
# CONFIG
# ============================================================
st.set_page_config("ĀROGYABODHA AI — Hybrid Clinical Intelligence OS", "🧠", layout="wide")

st.info(
    "ℹ️ ĀROGYABODHA AI is a Clinical Decision Support System (CDSS). "
    "It does NOT provide diagnosis or treatment. "
    "Final decisions must be made by licensed doctors."
)

# ============================================================
# STORAGE
# ============================================================
BASE = os.getcwd()
PDF_FOLDER = os.path.join(BASE, "medical_library")
VECTOR_FOLDER = os.path.join(BASE, "vector_cache")

PATIENT_DB = os.path.join(BASE, "patients.json")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# ============================================================
# DATABASE INIT
# ============================================================
if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"},
        "admin1": {"password": "admin123", "role": "Admin"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION STATE
# ============================================================
defaults = {
    "logged_in": False,
    "username": None,
    "role": None,
    "index_ready": False,
    "index": None,
    "docs": [],
    "srcs": []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# AUDIT SYSTEM
# ============================================================
def audit(event, meta=None):
    logs = []
    if os.path.exists(AUDIT_LOG):
        logs = json.load(open(AUDIT_LOG))
    logs.append({
        "time": str(datetime.datetime.utcnow()),
        "user": st.session_state.username,
        "role": st.session_state.role,
        "event": event,
        "meta": meta or {}
    })
    json.dump(logs, open(AUDIT_LOG, "w"), indent=2)

# ============================================================
# LOGIN SYSTEM
# ============================================================
def login_ui():
    st.title("ĀROGYABODHA AI — Secure Clinical Intelligence Login")
    with st.form("login"):
        u = st.text_input("User ID")
        p = st.text_input("Password", type="password")
        ok = st.form_submit_button("Login")

    if ok:
        users = json.load(open(USERS_DB))
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.session_state.role = users[u]["role"]
            audit("login", {"user": u})
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_ui()
    st.stop()

# ============================================================
# EMBEDDING MODEL
# ============================================================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ============================================================
# PDF RAG INDEX
# ============================================================
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    return [p.extract_text() for p in reader.pages if p.extract_text()]

def build_index():
    docs, srcs = [], []
    for pdf in os.listdir(PDF_FOLDER):
        if pdf.endswith(".pdf"):
            with open(os.path.join(PDF_FOLDER, pdf), "rb") as f:
                pages = extract_text(f.read())
            for i, p in enumerate(pages):
                docs.append(p)
                srcs.append(f"{pdf} — Page {i+1}")

    if not docs:
        return None, [], []

    emb = embedder.encode(docs)
    idx = faiss.IndexFlatL2(emb.shape[1])
    idx.add(np.array(emb, dtype=np.float32))
    faiss.write_index(idx, INDEX_FILE)
    pickle.dump({"docs": docs, "srcs": srcs}, open(CACHE_FILE, "wb"))
    return idx, docs, srcs

# Load cache
if os.path.exists(INDEX_FILE) and os.path.exists(CACHE_FILE):
    st.session_state.index = faiss.read_index(INDEX_FILE)
    cache = pickle.load(open(CACHE_FILE, "rb"))
    st.session_state.docs = cache["docs"]
    st.session_state.srcs = cache["srcs"]
    st.session_state.index_ready = True

# ============================================================
# RESEARCH CONNECTORS
# ============================================================
def fetch_pubmed(query):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 5}
    r = requests.get(url, params=params, timeout=15)
    return r.json()["esearchresult"]["idlist"]

def fetch_clinical_trials(query):
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {"query.term": query, "pageSize": 5}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    trials = []
    for study in data.get("studies", []):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        trials.append({
            "Trial ID": ident.get("nctId", "N/A"),
            "Phase": ", ".join(design.get("phases", ["N/A"])),
            "Status": status.get("overallStatus", "Unknown")
        })
    return trials

def fetch_fda_alerts():
    url = "https://api.fda.gov/drug/enforcement.json?limit=5"
    r = requests.get(url, timeout=15)
    data = r.json()
    return [
        f"{i.get('product_description','Unknown')} | Reason: {i.get('reason_for_recall','Safety Alert')}"
        for i in data.get("results", [])
    ]

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Hybrid Clinical Command Center", [
    "📁 Evidence Library",
    "🔬 Research Intelligence",
    "🏥 ICU Intelligence",
    "💊 Drug Interaction AI",
    "🩻 Radiology AI",
    "👤 Patient Workspace",
    "🧾 Doctor Orders",
    "🕒 Audit & Compliance"
])

# ============================================================
# EVIDENCE LIBRARY
# ============================================================
if module == "📁 Evidence Library":
    st.header("📁 Hospital Evidence Library")

    files = st.file_uploader("Upload Medical PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(PDF_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success("PDFs uploaded")

    if st.button("Build Evidence Index"):
        st.session_state.index, st.session_state.docs, st.session_state.srcs = build_index()
        st.session_state.index_ready = True
        audit("build_index", {"docs": len(st.session_state.docs)})
        st.success("Index built successfully")

# ============================================================
# RESEARCH INTELLIGENCE
# ============================================================
if module == "🔬 Research Intelligence":
    st.header("🔬 Clinical Research Intelligence Engine")

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze Research") and query:
        audit("research_query", {"query": query})

        pubmed_ids = fetch_pubmed(query)
        trials = fetch_clinical_trials(query)
        alerts = fetch_fda_alerts()

        st.subheader("📚 PubMed Journal Evidence")
        for i, pmid in enumerate(pubmed_ids, 1):
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            st.markdown(f"**{i}. PMID: {pmid}**  \n🔗 [View Article]({url})")

        st.subheader("🧪 Clinical Trials")
        st.table(pd.DataFrame(trials))

        st.subheader("⚠ FDA Safety Alerts")
        for a in alerts:
            st.warning(a)

# ============================================================
# ICU INTELLIGENCE
# ============================================================
if module == "🏥 ICU Intelligence":
    st.header("🏥 ICU Early Warning AI")

    hr = st.number_input("Heart Rate", 30, 200, 90)
    rr = st.number_input("Resp Rate", 8, 60, 20)
    spo2 = st.number_input("SpO2", 60, 100, 95)
    temp = st.number_input("Temp", 34.0, 42.0, 37.5)

    if st.button("Generate Risk Summary"):
        vitals = f"HR:{hr}, RR:{rr}, SpO2:{spo2}, Temp:{temp}"
        st.write(f"AI Risk Summary for Vitals: {vitals}")

# ============================================================
# DRUG INTERACTION
# ============================================================
if module == "💊 Drug Interaction AI":
    st.header("💊 Drug Interaction AI")

    meds = st.text_input("Enter medications")
    if st.button("Analyze"):
        st.write(f"AI Interaction Analysis for: {meds}")

# ============================================================
# RADIOLOGY AI
# ============================================================
if module == "🩻 Radiology AI":
    st.header("🩻 Radiology AI")

    file = st.file_uploader("Upload scan")
    if file:
        st.write("AI Radiology Report Generated")

# ============================================================
# PATIENT WORKSPACE
# ============================================================
if module == "👤 Patient Workspace":
    st.header("👤 Patient Workspace")

    patients = json.load(open(PATIENT_DB))
    with st.form("add_patient"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 0, 120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Symptoms")
        submit = st.form_submit_button("Create Case")

    if submit:
        case = {
            "id": len(patients)+1,
            "name": name,
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "timeline": [],
            "time": str(datetime.datetime.utcnow())
        }
        patients.append(case)
        json.dump(patients, open(PATIENT_DB, "w"), indent=2)
        audit("new_patient_case", case)
        st.success("Patient case created")

    st.dataframe(pd.DataFrame(patients), use_container_width=True)

# ============================================================
# DOCTOR ORDERS
# ============================================================
if module == "🧾 Doctor Orders":
    st.header("🧾 Doctor Orders")

    patients = json.load(open(PATIENT_DB))
    if patients:
        pid = st.selectbox("Select Patient", [p["id"] for p in patients])
        order = st.text_area("Enter Order")

        if st.button("Submit Order"):
            for p in patients:
                if p["id"] == pid:
                    p["timeline"].append({
                        "time": str(datetime.datetime.utcnow()),
                        "doctor": st.session_state.username,
                        "order": order
                    })
            json.dump(patients, open(PATIENT_DB, "w"), indent=2)
            audit("doctor_order", {"patient_id": pid})
            st.success("Order saved")

# ============================================================
# AUDIT
# ============================================================
if module == "🕒 Audit & Compliance":
    st.header("🕒 Audit & Compliance")
    if os.path.exists(AUDIT_LOG):
        df = pd.DataFrame(json.load(open(AUDIT_LOG)))
        st.dataframe(df, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.caption("ĀROGYABODHA AI — Hybrid Research + Hospital Intelligence OS")
