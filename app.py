import json
import os
import re
from pathlib import Path

import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
APP_DIR = Path(__file__).parent
KB_PATH = APP_DIR / "knowledge_base" / "machine_manual.json"
MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="AI Machine Diagnosis Assistant", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding:1.5rem 1.7rem;border-radius:18px;border:1px solid rgba(128,128,128,.25);background:linear-gradient(135deg,rgba(70,130,180,.12),rgba(120,90,180,.08));margin-bottom:1.2rem;}
.machine-card {padding:1.1rem;border:1px solid rgba(128,128,128,.25);border-radius:16px;min-height:145px;}
.result-card {padding:1rem 1.2rem;border:1px solid rgba(128,128,128,.25);border-radius:14px;margin-bottom:.8rem;}
.safety-card {padding:1rem 1.2rem;border-radius:14px;border:1px solid rgba(220,140,40,.45);background:rgba(220,140,40,.08);}
.small-muted {color:#777;font-size:.92rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())

def retrieve_cases(machine, problem, symptoms, top_k=4):
    data = load_knowledge_base()
    query_words = set(normalize(f"{machine} {problem} {symptoms}").split())
    scored = []
    for entry in data.get("entries", []):
        haystack = normalize(" ".join([
            entry.get("machine", ""), entry.get("problem", ""),
            " ".join(entry.get("symptoms", [])),
            " ".join(entry.get("possible_causes", [])),
            " ".join(entry.get("keywords", [])),
        ]))
        score = sum(1 for word in query_words if len(word) > 2 and word in haystack)
        if normalize(machine) in haystack: score += 3
        if normalize(problem) in normalize(entry.get("problem", "")): score += 5
        if score > 0: scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:top_k]]

def get_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try: key = st.secrets["GROQ_API_KEY"]
        except Exception: key = None
    return Groq(api_key=key) if key else None

def ask_model(machine, problem, symptoms, matched_cases):
    client = get_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env locally or Streamlit Secrets when deploying.")
    system = """You are an industrial machine troubleshooting assistant. Use the supplied engineering knowledge base as the primary technical reference. Do not invent machine-specific limits, part numbers, measurements, or OEM procedures. If unsupported, label it as an AI-based possibility. Always include safety precautions. Never advise bypassing guards, interlocks, pressure protection, or electrical safety procedures.
Return ONLY valid JSON with keys: likely_problem, confidence (High|Medium|Low), possible_causes (array), recommended_actions (array), troubleshooting_steps (array), safety_precautions (array), basis (Manual-Based Finding|AI-Based Possibility|Mixed)."""
    user = f"Machine: {machine}\nSelected problem: {problem}\nAdditional symptoms: {symptoms or 'None provided'}\n\nRelevant knowledge-base cases:\n{json.dumps(matched_cases, ensure_ascii=False, indent=2)}"
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        reasoning_effort="low", temperature=0.2,
        response_format={"type":"json_object"},
    )
    return json.loads(response.choices[0].message.content)

data = load_knowledge_base()
machines = data.get("machines", {})

st.markdown("""<div class="hero"><h1>⚙️ AI Machine Diagnosis Assistant</h1><p>AI-assisted troubleshooting for bearings, air compressors and centrifugal pumps.</p><div class="small-muted">Knowledge-base retrieval + GPT-OSS 120B analysis</div></div>""", unsafe_allow_html=True)
st.warning("Safety notice: This tool provides first-line troubleshooting guidance only. Follow site procedures, LOTO, depressurization requirements and OEM instructions.")

st.subheader("1. Select equipment")
icons = {"Bearing":"⚙️","Air Compressor":"💨","Centrifugal Pump":"🔄"}
cols = st.columns(len(machines))
for i, (name, info) in enumerate(machines.items()):
    with cols[i]:
        st.markdown(f'<div class="machine-card"><h3>{icons.get(name,"🔧")} {name}</h3><p>{info.get("description","")}</p></div>', unsafe_allow_html=True)
machine = st.radio("Equipment", list(machines.keys()), horizontal=True, label_visibility="collapsed")

st.subheader("2. Describe the problem")
problem = st.selectbox("Problem category", machines[machine].get("problems", []))
symptoms = st.text_area("Additional symptoms", placeholder="Example: The machine is noisy and vibration has increased...", height=120)

if st.button("🔍 Run Diagnosis", type="primary", use_container_width=True):
    with st.spinner("Matching engineering cases and generating diagnosis..."):
        try:
            cases = retrieve_cases(machine, problem, symptoms)
            result = ask_model(machine, problem, symptoms, cases)
            st.session_state["diagnosis"] = result
            st.session_state["matched_cases"] = cases
        except Exception as exc:
            st.error(str(exc))

if "diagnosis" in st.session_state:
    result = st.session_state["diagnosis"]
    st.divider(); st.subheader("3. Diagnosis Report")
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown(f'<div class="result-card"><h3>🔍 Likely Problem</h3><p>{result.get("likely_problem","Not determined")}</p></div>', unsafe_allow_html=True)
    with c2: st.metric("Confidence", result.get("confidence","Low"))
    st.caption(f"Basis: {result.get('basis','Mixed')}")
    left, right = st.columns(2)
    with left:
        st.markdown("### Possible Causes")
        for x in result.get("possible_causes", []): st.markdown(f"- {x}")
        st.markdown("### Recommended Actions")
        for x in result.get("recommended_actions", []): st.markdown(f"- {x}")
    with right:
        st.markdown("### Troubleshooting Steps")
        for x in result.get("troubleshooting_steps", []): st.markdown(f"- {x}")
        st.markdown('<div class="safety-card"><h3>⚠️ Safety Precautions</h3>', unsafe_allow_html=True)
        for x in result.get("safety_precautions", []): st.markdown(f"- {x}")
        st.markdown("</div>", unsafe_allow_html=True)
    with st.expander("View matched knowledge-base cases"): st.json(st.session_state.get("matched_cases", []))

st.divider(); st.subheader("How it works")
w = st.columns(4)
w[0].markdown("**1️⃣ Symptoms**\n\nUser describes the problem.")
w[1].markdown("**2️⃣ Retrieval**\n\nPython matches engineering cases.")
w[2].markdown("**3️⃣ AI Analysis**\n\nGPT-OSS 120B analyzes retrieved context.")
w[3].markdown("**4️⃣ Diagnosis**\n\nCauses, actions and safety are presented.")
st.caption("For maintenance decision support only — verify against the equipment OEM manual and site safety procedures.")
