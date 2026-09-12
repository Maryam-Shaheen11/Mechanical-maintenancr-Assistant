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
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Machine Diagnosis Assistant",
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.25);
        background: linear-gradient(
            135deg,
            rgba(70,130,180,.12),
            rgba(120,90,180,.08)
        );
        margin-bottom: 1.2rem;
    }

    .machine-card {
        padding: 1.1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 16px;
        min-height: 145px;
    }

    .result-card {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        margin-bottom: .8rem;
    }

    .safety-card {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(220,140,40,.45);
        background: rgba(220,140,40,.08);
    }

    .small-muted {
        color: #777;
        font-size: .92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge_base():
    """
    Load the machine diagnostic knowledge base.

    machine_manual.json must be in the same folder as app.py.
    """

    if not KB_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base file not found: {KB_PATH.name}. "
            "Make sure machine_manual.json is uploaded to the same "
            "GitHub folder as app.py."
        )

    with open(KB_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    """
    Convert text into simple lowercase words.
    """

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
    """
    Retrieve the most relevant diagnostic cases
    from the engineering knowledge base.
    """

    data = load_knowledge_base()

    query_text = f"{machine} {problem} {symptoms}"
    query_words = set(normalize(query_text).split())

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
                    " ".join(keywords_list),
                ]
            )
        )

        score = 0

        # Keyword matching
        for word in query_words:
            if len(word) > 2 and word in haystack:
                score += 1

        # Machine match
        if normalize(machine) in normalize(machine_text):
            score += 3

        # Problem match
        if normalize(problem) in normalize(problem_text):
            score += 5

        if score > 0:
            scored_cases.append((score, entry))

    scored_cases.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        entry
        for _, entry in scored_cases[:top_k]
    ]


# ============================================================
# GROQ CLIENT
# ============================================================

def get_client():
    """
    Get Groq API client.

    Locally:
        GROQ_API_KEY in .env

    Streamlit Cloud:
        GROQ_API_KEY in Streamlit Secrets
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# AI DIAGNOSIS
# ============================================================

def ask_model(machine, problem, symptoms, matched_cases):

    client = get_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    system_prompt = """
You are an industrial machine troubleshooting assistant.

Your task is to provide preliminary troubleshooting guidance
for bearings, air compressors, and centrifugal pumps.

Use the supplied engineering knowledge base as the PRIMARY
technical reference.

Do not invent:
- Machine-specific limits
- Part numbers
- Measurements
- OEM procedures
- Unsupported technical specifications

If something is not directly supported by the supplied
knowledge base, clearly treat it as an AI-based possibility.

Always include safety precautions.

Never advise users to:
- Bypass guards
- Bypass interlocks
- Bypass pressure protection
- Ignore electrical safety
- Perform unsafe maintenance

Return ONLY valid JSON.

Use exactly these keys:

{
    "likely_problem": "string",
    "confidence": "High|Medium|Low",
    "possible_causes": ["string"],
    "recommended_actions": ["string"],
    "troubleshooting_steps": ["string"],
    "safety_precautions": ["string"],
    "basis": "Manual-Based Finding|AI-Based Possibility|Mixed"
}
"""

    user_prompt = f"""
Machine: {machine}

Selected problem:
{problem}

Additional symptoms:
{symptoms if symptoms else "None provided"}

Relevant engineering knowledge-base cases:

{json.dumps(
    matched_cases,
    ensure_ascii=False,
    indent=2
)}

Analyze the reported problem using the supplied
knowledge-base information.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        reasoning_effort="low",
        temperature=0.2,
        response_format={
            "type": "json_object"
        },
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError(
            "The AI model returned an empty response."
        )

    return json.loads(content)


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:
    data = load_knowledge_base()
except Exception as error:
    st.error(
        f"Unable to load the knowledge base: {error}"
    )
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

st.markdown(
    """
    <div class="hero">
        <h1>⚙️ AI Machine Diagnosis Assistant</h1>
        <p>
            AI-assisted troubleshooting for bearings,
            air compressors and centrifugal pumps.
        </p>
        <div class="small-muted">
            Knowledge-base retrieval + GPT-OSS 120B analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAFETY NOTICE
# ============================================================

st.warning(
    "Safety notice: This tool provides first-line troubleshooting "
    "guidance only. Follow site procedures, LOTO, depressurization "
    "requirements and OEM instructions."
)


# ============================================================
# MACHINE SELECTION
# ============================================================

st.subheader("1. Select Equipment")

icons = {
    "Bearing": "⚙️",
    "Air Compressor": "💨",
    "Centrifugal Pump": "🔄",
}

machine_names = list(machines.keys())

cols = st.columns(len(machine_names))

for index, name in enumerate(machine_names):

    info = machines.get(name, {})

    with cols[index]:

        st.markdown(
            f"""
            <div class="machine-card">
                <h3>
                    {icons.get(name, "🔧")} {name}
                </h3>

                <p>
                    {info.get("description", "")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


machine = st.radio(
    "Equipment",
    machine_names,
    horizontal=True,
    label_visibility="collapsed",
)


# ============================================================
# PROBLEM SELECTION
# ============================================================

st.subheader("2. Describe the Problem")

problem_list = machines[machine].get(
    "problems",
    []
)

if not problem_list:
    st.error(
        f"No problem categories found for {machine}."
    )
    st.stop()


problem = st.selectbox(
    "Problem category",
    problem_list,
)


# ============================================================
# SYMPTOMS
# ============================================================

symptoms = st.text_area(
    "Additional symptoms",
    placeholder=(
        "Example: The machine is noisy and "
        "vibration has increased..."
    ),
    height=120,
)


# ============================================================
# RUN DIAGNOSIS
# ============================================================

if st.button(
    "🔍 Run Diagnosis",
    type="primary",
    use_container_width=True,
):

    with st.spinner(
        "Matching engineering cases and generating diagnosis..."
    ):

        try:

            # Retrieve relevant engineering cases
            cases = retrieve_cases(
                machine,
                problem,
                symptoms,
            )

            if not cases:
                st.warning(
                    "No closely matching knowledge-base case "
                    "was found. The AI will have limited "
                    "reference information."
                )

            # Generate AI diagnosis
            result = ask_model(
                machine,
                problem,
                symptoms,
                cases,
            )

            # Save results
            st.session_state["diagnosis"] = result
            st.session_state["matched_cases"] = cases

        except Exception as error:

            st.error(
                f"Diagnosis failed: {error}"
            )


# ============================================================
# DIAGNOSIS REPORT
# ============================================================

if "diagnosis" in st.session_state:

    result = st.session_state["diagnosis"]

    st.divider()

    st.subheader("3. Diagnosis Report")

    # --------------------------------------------------------
    # Likely Problem + Confidence
    # --------------------------------------------------------

    column1, column2 = st.columns(
        [3, 1]
    )

    with column1:

        st.markdown(
            f"""
            <div class="result-card">
                <h3>🔍 Likely Problem</h3>
                <p>
                    {result.get(
                        "likely_problem",
                        "Not determined"
                    )}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with column2:

        st.metric(
            "Confidence",
            result.get(
                "confidence",
                "Low"
            ),
        )

    st.caption(
        f"Basis: {result.get('basis', 'Mixed')}"
    )


    # --------------------------------------------------------
    # Causes + Actions
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### Possible Causes"
        )

        causes = result.get(
            "possible_causes",
            []
        )

        if causes:
            for item in causes:
                st.markdown(
                    f"- {item}"
                )
        else:
            st.write(
                "No causes provided."
            )


        st.markdown(
            "### Recommended Actions"
        )

        actions = result.get(
            "recommended_actions",
            []
        )

        if actions:
            for item in actions:
                st.markdown(
                    f"- {item}"
                )
        else:
            st.write(
                "No recommended actions provided."
            )


    # --------------------------------------------------------
    # Troubleshooting + Safety
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### Troubleshooting Steps"
        )

        steps = result.get(
            "troubleshooting_steps",
            []
        )

        if steps:
            for item in steps:
                st.markdown(
                    f"- {item}"
                )
        else:
            st.write(
                "No troubleshooting steps provided."
            )


        st.markdown(
            """
            <div class="safety-card">
                <h3>⚠️ Safety Precautions</h3>
            """,
            unsafe_allow_html=True,
        )

        safety_items = result.get(
            "safety_precautions",
            []
        )

        if safety_items:

            for item in safety_items:

                st.markdown(
                    f"- {item}"
                )

        else:

            st.write(
                "Follow site safety procedures and OEM instructions."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # Knowledge Base Evidence
    # --------------------------------------------------------

    with st.expander(
        "📚 View Matched Knowledge-Base Cases"
    ):

        matched_cases = st.session_state.get(
            "matched_cases",
            []
        )

        if matched_cases:

            st.json(
                matched_cases
            )

        else:

            st.info(
                "No matching knowledge-base cases were found."
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("How It Works")

workflow = st.columns(4)

workflow[0].markdown(
    """
    **1️⃣ Symptoms**

    User selects equipment and
    describes the problem.
    """
)

workflow[1].markdown(
    """
    **2️⃣ Retrieval**

    Python matches the symptoms
    with engineering cases.
    """
)

workflow[2].markdown(
    """
    **3️⃣ AI Analysis**

    GPT-OSS 120B analyzes the
    retrieved technical context.
    """
)

workflow[3].markdown(
    """
    **4️⃣ Diagnosis**

    Causes, actions, troubleshooting
    and safety guidance are presented.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI Machine Diagnosis Assistant • "
    "For preliminary maintenance troubleshooting assistance only."
)
