import json
import os
import re
from pathlib import Path

import streamlit as st
from groq import Groq
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

APP_DIR = Path(__file__).parent
KB_PATH = APP_DIR / "machine_manual.json"

MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Machine Diagnosis Assistant",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge_base():

    if not KB_PATH.exists():
        raise FileNotFoundError(
            "machine_manual.json was not found. "
            "Make sure it is in the same folder as app.py."
        )

    with open(KB_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):

    if not text:
        return ""

    return re.sub(
        r"[^a-z0-9\s]",
        " ",
        str(text).lower()
    )


# ============================================================
# KNOWLEDGE BASE RETRIEVAL
# ============================================================

def retrieve_cases(machine, problem, symptoms, top_k=4):

    data = load_knowledge_base()

    query = f"{machine} {problem} {symptoms}"
    query_words = set(normalize(query).split())

    scored_cases = []

    for entry in data.get("entries", []):

        machine_text = entry.get("machine", "")
        problem_text = entry.get("problem", "")

        symptoms_list = entry.get("symptoms", [])
        causes_list = entry.get("possible_causes", [])
        keywords_list = entry.get("keywords", [])

        haystack = normalize(
            " ".join(
                [
                    machine_text,
                    problem_text,
                    " ".join(symptoms_list),
                    " ".join(causes_list),
                    " ".join(keywords_list)
                ]
            )
        )

        score = 0

        for word in query_words:

            if len(word) > 2 and word in haystack:
                score += 1

        if normalize(machine) in normalize(machine_text):
            score += 3

        if normalize(problem) in normalize(problem_text):
            score += 5

        if score > 0:
            scored_cases.append(
                (score, entry)
            )

    scored_cases.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        entry
        for _, entry in scored_cases[:top_k]
    ]


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY in Streamlit Secrets."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# AI DIAGNOSIS
# ============================================================

def generate_diagnosis(
    machine,
    problem,
    symptoms,
    matched_cases
):

    client = get_groq_client()

    system_prompt = """
You are an industrial machine diagnosis assistant.

Your task is to analyze mechanical machine problems using
the supplied engineering knowledge base.

Supported equipment:
- Bearings
- Air Compressors
- Centrifugal Pumps

Use the supplied knowledge base as the primary technical source.

Do not invent machine-specific:
- measurements
- limits
- part numbers
- OEM procedures
- unsupported technical specifications

If the knowledge base does not fully support a conclusion,
state that the diagnosis is preliminary.

Generate a professional diagnostic report.

Return ONLY valid JSON using exactly these keys:

{
    "diagnosis": "short diagnosis",
    "possible_causes": [
        "cause 1",
        "cause 2",
        "cause 3"
    ],
    "recommended_actions": [
        "action 1",
        "action 2",
        "action 3"
    ],
    "troubleshooting": [
        "step 1",
        "step 2",
        "step 3"
    ],
    "safety_precautions": [
        "safety instruction 1",
        "safety instruction 2"
    ],
    "confidence": "High|Medium|Low"
}

Do not add any other keys.
"""

    user_prompt = f"""
Machine:
{machine}

Problem:
{problem}

Additional symptoms:
{symptoms if symptoms else "No additional symptoms provided."}

Relevant engineering knowledge base:
{json.dumps(
    matched_cases,
    ensure_ascii=False,
    indent=2
)}

Generate the diagnostic report based primarily
on the supplied knowledge base.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            response_format={
                "type": "json_object"
            }
        )

    except Exception as error:

        raise RuntimeError(
            f"Unable to connect to Groq AI service: {error}"
        )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The AI model returned an empty response."
        )

    try:

        return json.loads(content)

    except json.JSONDecodeError:

        raise RuntimeError(
            "The AI returned an invalid diagnosis format."
        )


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = load_knowledge_base()

except Exception as error:

    st.error(str(error))
    st.stop()


machines = data.get("machines", {})

if not machines:

    st.error(
        "No machine information was found in machine_manual.json."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "⚙️ AI Machine Diagnosis Assistant"
)

st.write(
    "AI-assisted troubleshooting for bearings, "
    "air compressors and centrifugal pumps."
)

st.caption(
    "Knowledge-base retrieval + GPT-OSS 120B"
)


# ============================================================
# MACHINE SELECTION
# ============================================================

st.subheader("Select Equipment")

icons = {
    "Bearing": "⚙️",
    "Air Compressor": "💨",
    "Centrifugal Pump": "🔄"
}

machine_names = list(
    machines.keys()
)

machine = st.radio(
    "Equipment",
    machine_names,
    horizontal=True
)


# ============================================================
# PROBLEM SELECTION
# ============================================================

st.subheader("Describe the Problem")

problem_list = machines[machine].get(
    "problems",
    []
)

if not problem_list:

    st.error(
        f"No problem categories available for {machine}."
    )

    st.stop()


problem = st.selectbox(
    "Problem category",
    problem_list
)


# ============================================================
# SYMPTOMS
# ============================================================

symptoms = st.text_area(
    "Additional symptoms",
    placeholder=(
        "Describe any additional symptoms, "
        "such as noise, vibration, leakage, "
        "overheating, low flow or pressure..."
    ),
    height=120
)


# ============================================================
# DIAGNOSIS BUTTON
# ============================================================

if st.button(
    "🔍 Run Diagnosis",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Analyzing machine condition..."
    ):

        try:

            matched_cases = retrieve_cases(
                machine,
                problem,
                symptoms
            )

            diagnosis = generate_diagnosis(
                machine,
                problem,
                symptoms,
                matched_cases
            )

            st.session_state["diagnosis"] = diagnosis

        except Exception as error:

            st.error(
                str(error)
            )


# ============================================================
# DIAGNOSIS REPORT
# ============================================================

if "diagnosis" in st.session_state:

    diagnosis = st.session_state["diagnosis"]

    st.divider()

    st.header(
        "Diagnostic Report"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "Diagnosis"
    )

    st.write(
        diagnosis.get(
            "diagnosis",
            "No diagnosis available."
        )
    )


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = diagnosis.get(
        "confidence",
        "Low"
    )

    st.metric(
        "Confidence",
        confidence
    )


    # --------------------------------------------------------
    # POSSIBLE CAUSES
    # --------------------------------------------------------

    st.subheader(
        "Possible Causes"
    )

    causes = diagnosis.get(
        "possible_causes",
        []
    )

    for cause in causes:

        st.markdown(
            f"- {cause}"
        )


    # --------------------------------------------------------
    # RECOMMENDED ACTIONS
    # --------------------------------------------------------

    st.subheader(
        "Recommended Actions"
    )

    actions = diagnosis.get(
        "recommended_actions",
        []
    )

    for action in actions:

        st.markdown(
            f"- {action}"
        )


    # --------------------------------------------------------
    # TROUBLESHOOTING
    # --------------------------------------------------------

    st.subheader(
        "Troubleshooting Steps"
    )

    troubleshooting = diagnosis.get(
        "troubleshooting",
        []
    )

    for step in troubleshooting:

        st.markdown(
            f"{troubleshooting.index(step) + 1}. {step}"
        )


    # --------------------------------------------------------
    # SAFETY
    # --------------------------------------------------------

    st.subheader(
        "⚠️ Safety Precautions"
    )

    safety = diagnosis.get(
        "safety_precautions",
        []
    )

    for instruction in safety:

        st.markdown(
            f"- {instruction}"
        )


# ============================================================
# SAFETY NOTICE
# ============================================================

st.divider()

st.caption(
    "Safety: This tool provides preliminary troubleshooting "
    "guidance only. Follow LOTO procedures, depressurization "
    "requirements, electrical safety procedures and OEM instructions."
)
