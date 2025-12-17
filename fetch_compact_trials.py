import requests
import csv
import time
import json
from pathlib import Path
import re


# -----------------------------
# ClinicalTrials.gov v2 Base URL
# -----------------------------
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# -----------------------------
# Required Fields (paths in JSON)
# -----------------------------
FIELDS = {
    "nctId": ["protocolSection", "identificationModule", "nctId"],
    "briefTitle": ["protocolSection", "identificationModule", "briefTitle"],

    # basic description
    "briefSummary": ["protocolSection", "descriptionModule", "briefSummary"],
    "detailedDescription": ["protocolSection", "descriptionModule", "detailedDescription"],

    # condition / phase / status
    "conditions": ["protocolSection", "conditionsModule", "conditions"],
    "overallStatus": ["protocolSection", "statusModule", "overallStatus"],
    "phases": ["protocolSection", "designModule", "phases"],

    # design / enrollment
    "studyType": ["protocolSection", "designModule", "studyType"],
    "enrollmentCount": ["protocolSection", "designModule", "enrollmentInfo", "count"],
    "targetDuration": ["protocolSection", "designModule", "targetDuration"],

    # eligibility
    "eligibilityCriteria": ["protocolSection", "eligibilityModule", "eligibilityCriteria"],
    "minAge": ["protocolSection", "eligibilityModule", "minAge"],
    "maxAge": ["protocolSection", "eligibilityModule", "maxAge"],
    "sexes": ["protocolSection", "eligibilityModule", "sexes"],
}

# ===========================================================
# TEXT CLEANING
# ===========================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ===========================================================
# SAFE GET FOR NESTED JSON
# ===========================================================

def safe_get(obj, path):
    cur = obj
    for p in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(p)
        if cur is None:
            return ""

    if isinstance(cur, dict):
        for key in ("text", "value", "narrative"):
            if key in cur:
                return cur[key]
        return json.dumps(cur)
 
    if isinstance(cur, list):
 
        flat = []
        for x in cur:
            if isinstance(x, dict):

                vals = [x.get(k, "") for k in ("text", "value", "narrative")]
                flat.append(" ".join(v for v in vals if v))
            else:
                flat.append(str(x))
        return " | ".join([x for x in flat if x])
    return cur

# ===========================================================
# MAIN EXTRACTOR
# ===========================================================

def extract_compact(study):
    compact = {}
    for field, path in FIELDS.items():
        val = safe_get(study, path)
        if field in ("briefSummary", "detailedDescription", "eligibilityCriteria"):
            val = normalize_text(val)
        compact[field] = val
    return compact

# ===========================================================
# FETCH + WRITE TO CSV
# ===========================================================

def fetch_and_save_csv(
    out_csv="data/compact_trials.csv",
    max_studies=5000,

    statuses=("RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"),
    page_size=1000,
    sleep=0.2,
    condition_term: str | None = None,
):
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FIELDS.keys()))
        writer.writeheader()

        total = 0
        session = requests.Session()

        for status in statuses:
            page_token = None

            while total < max_studies:
                params = {
                    "filter.overallStatus": status,
                    "pageSize": page_size,
                    "format": "json",
                }
                if condition_term:

                    params["query.cond"] = condition_term
                if page_token:
                    params["pageToken"] = page_token

                resp = session.get(BASE_URL, params=params)
                if resp.status_code != 200:
                    print(f"[WARNING] HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                data = resp.json()
                studies = data.get("studies", [])
                if not studies:
                    break

                for s in studies:
                    writer.writerow(extract_compact(s))
                    total += 1
                    if total >= max_studies:
                        break

                print(f"[INFO] status={status} fetched {len(studies)} → total={total}")

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

                time.sleep(sleep)

    print(f"[DONE] Saved {total} studies → {out_csv}")


if __name__ == "__main__":
    fetch_and_save_csv(
        out_csv="data/compact_trials_small.csv",
        max_studies=10000,
        condition_term=None,
    )