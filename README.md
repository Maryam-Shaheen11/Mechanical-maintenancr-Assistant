# AI Machine Diagnosis Assistant

AI-assisted troubleshooting for Bearings, Air Compressors and Centrifugal Pumps.

## Architecture
User symptoms -> Python keyword retrieval -> engineering knowledge base -> relevant cases -> GPT-OSS 120B via Groq -> structured diagnosis.

## Model
This app uses `openai/gpt-oss-120b` through Groq. GPT-OSS is an OpenAI open-weight model; it is not served through the OpenAI API. Groq currently lists this model on its Free Plan with rate limits, which can change.

## Run locally
1. `python -m venv .venv`
2. Windows: `.venv\\Scripts\\activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add `GROQ_API_KEY`
5. `streamlit run app.py`

## Deployment
Push to GitHub, create a Streamlit app using `app.py`, and add `GROQ_API_KEY` in Streamlit Secrets.

## Safety
This is first-line decision support, not a replacement for a qualified maintenance engineer, OEM manual, LOTO procedure, or site risk assessment.
