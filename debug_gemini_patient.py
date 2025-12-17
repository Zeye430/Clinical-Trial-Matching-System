import os
import json
from datetime import datetime
import google.generativeai as genai

# ============================================================
# CONFIG
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=API_KEY)

# ✅ Free-tier friendly model
MODEL_NAME = "models/gemini-2.5-flash-lite"

# ============================================================
# TEST INPUT
# ============================================================

patient_text = "65 year old male kidney disease anemia"

prompt = f"""
You are a medical information extraction system.

Output ONLY valid JSON.
Do NOT include markdown, explanations, or extra text.

Required keys:
- age
- sex
- conditions
- keywords

Text:
{patient_text}
"""

# ============================================================
# GEMINI CALL
# ============================================================

print("\n===== CALLING GEMINI =====\n")

model = genai.GenerativeModel(MODEL_NAME)
response = model.generate_content(prompt)

raw = response.text

# ============================================================
# RAW OUTPUT (TERMINAL)
# ============================================================

print("===== RAW GEMINI OUTPUT =====")
print(repr(raw))
print("===== END RAW OUTPUT =====\n")

# ============================================================
# RAW OUTPUT (FILE)
# ============================================================

with open("gemini_raw_output.txt", "a", encoding="utf-8") as f:
    f.write("\n---------------------------------\n")
    f.write(str(datetime.now()) + "\n")
    f.write("RAW OUTPUT:\n")
    f.write(repr(raw) + "\n")

# ============================================================
# JSON CLEANING + PARSING
# ============================================================

print("===== JSON PARSING ATTEMPT =====")

try:
    clean = raw.strip()

    # 🔧 Remove markdown fences if present
    if clean.startswith("```"):
        clean = clean.replace("```json", "")
        clean = clean.replace("```", "")
        clean = clean.strip()

    parsed = json.loads(clean)

    print("JSON parsed successfully:")
    print(parsed)

    with open("gemini_raw_output.txt", "a", encoding="utf-8") as f:
        f.write("PARSED JSON:\n")
        f.write(json.dumps(parsed, indent=2))
        f.write("\n")

except Exception as e:
    print("JSON parsing FAILED:")
    print(e)

    with open("gemini_raw_output.txt", "a", encoding="utf-8") as f:
        f.write("JSON PARSING FAILED:\n")
        f.write(str(e) + "\n")

print("\n===== DEBUG COMPLETE =====\n")
