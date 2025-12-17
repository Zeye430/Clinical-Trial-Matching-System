# semantic_llm.py
import os
import json
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI() 

MODEL_NAME = "gpt-4.1-mini"


def _safe_json_from_text(text: str) -> Dict[str, Any]:

    if not text:
        return {}

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}

    candidate = m.group(0)

    candidate = re.sub(r"```(?:json)?", "", candidate)
    candidate = candidate.replace("```", "")

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def extract_patient_info(description: str) -> Dict[str, Any]:
    """
    Use ChatGPT model to parse patient natural language description into structured information:

    Returns:
    {
        "age": int or None,
        "sex": "MALE" / "FEMALE" / "UNKNOWN",
        "conditions": [str, ...],
        "keywords": [str, ...],
    }
    """
    prompt = f"""
You help match patients to clinical trials.

Read the patient description below and extract a concise JSON object with:
- age: integer number of years, or null if not mentioned
- sex: one of "MALE", "FEMALE", or "UNKNOWN"
- conditions: list of main diagnoses or conditions (diseases)
- keywords: list of 5-10 important medical keywords (diseases, symptoms, treatments, risk factors)

Return ONLY valid JSON. Do not include any explanations or extra text.

Patient description:
\"\"\"{description}\"\"\"
"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content
        data = _safe_json_from_text(content)
    except Exception as e:
        print("[WARN] extract_patient_info error:", e)
        data = {}


    # 1. Age
    age = data.get("age")
    try:
        age = int(age) if age is not None else None
        if age < 0 or age > 120:
            age = None
    except (ValueError, TypeError):
        age = None

    # 2. Sex
    sex = str(data.get("sex", "UNKNOWN")).upper()
    if sex not in {"MALE", "FEMALE"}:
        sex = "UNKNOWN"

    # 3. conditions / keywords (as lists)
    conditions = data.get("conditions") or []
    if isinstance(conditions, str):
        conditions = [conditions]

    keywords = data.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]

    return {
        "age": age,
        "sex": sex,
        "conditions": [c.strip() for c in conditions if c],
        "keywords": [k.strip() for k in keywords if k],
    }
