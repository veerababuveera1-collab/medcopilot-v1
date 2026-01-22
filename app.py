# ============================================================
# VEERA AI — Defence Intelligence OS (DEMO PLATFORM)
# Strategy, Simulation & Advisory Copilot (Non-Operational)
# Author: Veera Babu
# ============================================================

import streamlit as st
import os
import google.generativeai as genai
from datetime import datetime

# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="VEERA AI — Defence Intelligence OS",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# GEMINI CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-1.5-pro"

# ============================================================
# SYSTEM PROMPT (ADVISORY / NON-OPERATIONAL)
# ============================================================

SYSTEM_PROMPT = """
You are VEERA AI — a Defence Strategy & Intelligence Advisory Copilot.

You provide high-level, non-operational analysis for:
- strategic planning
- risk assessment
- readiness evaluation
- scenario simulation
- decision support

Rules:
1. Do not provide tactical combat instructions or targeting guidance.
2. Do not generate real-world attack plans.
3. Focus on strategy, risk, preparedness, resilience, and policy.
4. Maintain professional defence-analyst tone.
5. Support English and Telugu.
6. If a request is operational or sensitive, respond with a safe alternative.
"""

# ============================================================
# AI ENGINE (ADVISORY)
# ============================================================

def get_defence_advisory(question, context=""):
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT
    )

    prompt = f"""
You are a defence intelligence advisor.

Context (if any):
{context}

Question:
{question}

Instructions:
- Provide strategic, non-operational analysis
- Focus on risk, readiness, resilience, and policy
- Offer options and considerations (not commands)
- Use clear, professional language
"""

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "max_output_tokens": 2048
        }
    )

    return response.text


# ============================================================
# HEADER
# ============================================================

st.markdown("""
# 🛡️ VEERA AI — Defence Intelligence OS (Demo)
### Strategy, Simulation & Decision Support Copilot

> Advisory platform for defence planning, readiness and resilience.

---
""")

# ============================================================
# SIDEBAR — MODULES
# ============================================================

module = st.sidebar.selectbox(
    "Select Defence Intelligence Module",
    [
        "🛰 Strategic Threat Assessment",
        "📊 Readiness & Risk Analysis",
        "🧭 Scenario Simulation (What-If)",
        "📡 Cyber & Space Resilience",
        "🗺 Border & Maritime Security (Policy View)",
        "🔐 Governance & Compliance"
    ]
)

# ============================================================
# MODULE 1: Strategic Threat Assessment
# ============================================================

if module == "🛰 Strategic Threat Assessment":
    st.subheader("🛰 Strategic Threat Assessment (Advisory)")

    topic = st.text_area(
        "Describe the strategic environment or concern (high-level):",
        placeholder="Example: Regional stability, hybrid threats, supply chain resilience..."
    )

    if st.button("Generate Assessment"):
        with st.spinner("Analyzing strategic landscape..."):
            answer = get_defence_advisory(topic)

        st.markdown("### 🧠 VEERA AI — Strategic Assessment")
        st.write(answer)


# ============================================================
# MODULE 2: Readiness & Risk Analysis
# ============================================================

if module == "📊 Readiness & Risk Analysis":
    st.subheader("📊 Readiness & Risk Analysis")

    inputs = st.text_area(
        "Enter readiness factors (logistics, training, cyber posture, supply chain, etc.):",
        placeholder="Example: Force readiness, logistics, cyber hygiene, satellite redundancy..."
    )

    if st.button("Evaluate Readiness & Risks"):
        with st.spinner("Evaluating readiness and risk posture..."):
            answer = get_defence_advisory(inputs)

        st.markdown("### 📊 VEERA AI — Readiness & Risk Summary")
        st.write(answer)


# ============================================================
# MODULE 3: Scenario Simulation (What-If)
# ============================================================

if module == "🧭 Scenario Simulation (What-If)":
    st.subheader("🧭 Scenario Simulation (What-If)")

    scenario = st.text_area(
        "Describe a hypothetical scenario for policy/strategy simulation:",
        placeholder="Example: Communication disruption, logistics bottleneck, cyber incident..."
    )

    if st.button("Run What-If Simulation"):
        with st.spinner("Simulating scenario (policy/strategy level)..."):
            answer = get_defence_advisory(scenario)

        st.markdown("### 🧭 VEERA AI — Scenario Insights")
        st.write(answer)


# ============================================================
# MODULE 4: Cyber & Space Resilience
# ============================================================

if module == "📡 Cyber & Space Resilience":
    st.subheader("📡 Cyber & Space Resilience (Advisory)")

    query = st.text_area(
        "Ask about cyber, satellite, and communication resilience:",
        placeholder="Example: Redundancy, incident response, zero-trust posture..."
    )

    if st.button("Generate Resilience Guidance"):
        with st.spinner("Preparing resilience guidance..."):
            answer = get_defence_advisory(query)

        st.markdown("### 📡 VEERA AI — Resilience Guidance")
        st.write(answer)


# ============================================================
# MODULE 5: Border & Maritime Security (Policy View)
# ============================================================

if module == "🗺 Border & Maritime Security (Policy View)":
    st.subheader("🗺 Border & Maritime Security (Policy View)")

    policy_query = st.text_area(
        "Enter policy/strategy question (no operational details):",
        placeholder="Example: Surveillance policy, coordination frameworks, capacity building..."
    )

    if st.button("Generate Policy Advisory"):
        with st.spinner("Generating policy advisory..."):
            answer = get_defence_advisory(policy_query)

        st.markdown("### 🗺 VEERA AI — Policy Advisory")
        st.write(answer)


# ============================================================
# MODULE 6: Governance & Compliance
# ============================================================

if module == "🔐 Governance & Compliance":
    st.subheader("🔐 Governance, Ethics & Compliance")

    gov_query = st.text_area(
        "Ask about governance, ethics, audits, and compliance:",
        placeholder="Example: AI governance, audit trails, data protection, procurement..."
    )

    if st.button("Get Governance Guidance"):
        with st.spinner("Preparing governance guidance..."):
            answer = get_defence_advisory(gov_query)

        st.markdown("### 🔐 VEERA AI — Governance Guidance")
        st.write(answer)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
---
### 🔐 Safety & Ethics by Design  
✔ Advisory-only (non-operational)  
✔ Strategy & policy focused  
✔ Audit-friendly outputs  
✔ Multi-language ready  

**VEERA AI — Defence Intelligence OS (Demo Platform)**
""")
