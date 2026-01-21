# ============================================================
# ĀROGYABODHA AI — Phase-3+ PRODUCTION Medical Intelligence OS
# Journal + Trials + FDA + Health Economics + PDF Reports
# ============================================================

import streamlit as st
import os, json, pickle, datetime, io, requests, re
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ============================================================
# CONFIG
# ============================================================
st.set_page_config("ĀROGYABODHA AI — Medical Intelligence OS", "🧠", layout="wide")

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
REPORT_FOLDER = os.path.join(BASE, "reports")

PATIENT_DB = os.path.join(BASE, "patients.json")
AUDIT_LOG = os.path.join(BASE, "audit_log.json")
USERS_DB = os.path.join(BASE, "users.json")

INDEX_FILE = os.path.join(VECTOR_FOLDER, "index.faiss")
CACHE_FILE = os.path.join(VECTOR_FOLDER, "cache.pkl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ============================================================
# DATABASE INIT
# ============================================================
if not os.path.exists(PATIENT_DB):
    json.dump([], open(PATIENT_DB, "w"), indent=2)

if not os.path.exists(USERS_DB):
    json.dump({
        "doctor1": {"password": "doctor123", "role": "Doctor"},
        "researcher1": {"password": "research123", "role": "Researcher"}
    }, open(USERS_DB, "w"), indent=2)

# ============================================================
# SESSION
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
# AUDIT
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
# LOGIN
# ============================================================
def login_ui():
    st.title("ĀROGYABODHA AI — Secure Medical Intelligence Login")
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
# PUBMED CONNECTORS (Abstract + Authors)
# ============================================================
def fetch_pubmed_ids(query):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 5}
    r = requests.get(url, params=params, timeout=15)
    return r.json()["esearchresult"]["idlist"]

def fetch_pubmed_details(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    r = requests.get(url, params=params, timeout=15)
    return r.text[:1500]

# ============================================================
# HEALTH ECONOMICS MODEL
# ============================================================
def cost_effectiveness_model(modality_costs):
    """
    Simple cost-effectiveness comparison model
    """
    df = pd.DataFrame(modality_costs, columns=["Modality", "Annual Cost ($)", "QALY"])
    df["Cost per QALY"] = df["Annual Cost ($)"] / df["QALY"]
    return df

# ============================================================
# PDF REPORT GENERATOR
# ============================================================
def generate_pdf_report(title, content):
    filename = f"{REPORT_FOLDER}/{title.replace(' ','_')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph(title, styles["Title"]))
    flow.append(Spacer(1, 12))

    for section in content.split("\n"):
        flow.append(Paragraph(section, styles["Normal"]))
        flow.append(Spacer(1, 12))

    doc.build(flow)
    return filename

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f"👨‍⚕️ **{st.session_state.username}** ({st.session_state.role})")

if st.sidebar.button("Logout"):
    audit("logout")
    st.session_state.logged_in = False
    st.rerun()

module = st.sidebar.radio("Medical Intelligence Center", [
    "🔬 Phase-3 Research Copilot",
    "📊 Cost-Effectiveness Lab",
    "📄 Research Report Generator",
    "🕒 Audit & Compliance"
])

# ============================================================
# RESEARCH COPILOT (JOURNAL INTELLIGENCE)
# ============================================================
if module == "🔬 Phase-3 Research Copilot":
    st.header("🔬 Phase-3 Clinical Research Intelligence Engine")

    query = st.text_input("Ask a clinical research question")

    if st.button("Analyze Research") and query:
        pubmed_ids = fetch_pubmed_ids(query)

        st.subheader("📚 Journal Evidence — PubMed Indexed Articles")

        for i, pmid in enumerate(pubmed_ids, 1):
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            st.markdown(f"**{i}. PMID: {pmid}**  \n🔗 [View Journal Article]({pubmed_url})")

            with st.expander("View Abstract & Authors"):
                xml_data = fetch_pubmed_details(pmid)
                st.code(xml_data[:1200])

# ============================================================
# HEALTH ECONOMICS LAB
# ============================================================
if module == "📊 Cost-Effectiveness Lab":
    st.header("📊 Health Economics — Cost Effectiveness Model")

    modalities = [
        ["Hemodialysis", 25000, 0.75],
        ["Peritoneal Dialysis", 18000, 0.72],
        ["Transplant Follow-up", 12000, 0.9]
    ]

    df = cost_effectiveness_model(modalities)

    st.subheader("Cost-Effectiveness Comparison (Dialysis)")
    st.dataframe(df, use_container_width=True)

# ============================================================
# PDF REPORT GENERATOR
# ============================================================
if module == "📄 Research Report Generator":
    st.header("📄 Clinical Research PDF Generator")

    title = st.text_input("Report Title")
    content = st.text_area("Paste Research Summary / Evidence")

    if st.button("Generate PDF Report") and title and content:
        file_path = generate_pdf_report(title, content)
        audit("generate_report", {"title": title})
        st.success("PDF Report Generated")
        st.download_button("Download Report", open(file_path, "rb"), file_name=os.path.basename(file_path))

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
st.caption("ĀROGYABODHA AI — Phase-3+ PRODUCTION Medical Intelligence OS")
