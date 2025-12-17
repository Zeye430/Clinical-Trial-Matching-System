# evaluation.py
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from data_loader import load_trials, build_full_text
from tfidf_model import TFIDFModel
from hybrid_model import HybridModel


# -----------------------------
# Define patient-trial relevance rules
# -----------------------------
def trial_relevant_for_patient(patient: Dict[str, Any], row: pd.Series) -> bool:
    age = patient.get("age")
    cond_keywords = [c.lower() for c in (patient.get("conditions") or []) if c]

    # Condition 1: at least one condition keyword matches trial conditions
    trial_conds = str(row.get("conditions", "")).lower()
    if cond_keywords:
        if not any(k in trial_conds for k in cond_keywords):
            return False

    # Condition 2: age compatible (if trial has age limits)
    if age is not None:
        min_age = row.get("minAgeYears", np.nan)
        max_age = row.get("maxAgeYears", np.nan)

        if not np.isnan(min_age) and age < min_age:
            return False
        if not np.isnan(max_age) and age > max_age:
            return False

    return True


# -----------------------------
# Few synthetic patient cases
# -----------------------------
def build_synthetic_patients() -> List[Dict[str, Any]]:
    
    patients = [
        {
            "name": "Case 1: T2D middle-aged male",
            "age": 60,
            "sex": "MALE",
            "conditions": ["type 2 diabetes"],
            "keywords": ["diabetes", "hyperglycemia"],
            "query_text": "60 year old male with type 2 diabetes and high blood sugar",
        },
        {
            "name": "Case 2: Breast cancer female",
            "age": 52,
            "sex": "FEMALE",
            "conditions": ["breast cancer"],
            "keywords": ["breast tumor", "HER2"],
            "query_text": "52 year old female with breast cancer",
        },
        {
            "name": "Case 3: Chronic kidney disease",
            "age": 70,
            "sex": "MALE",
            "conditions": ["chronic kidney disease"],
            "keywords": ["CKD", "renal failure"],
            "query_text": "70 year old male with chronic kidney disease stage 3",
        },
    ]
    return patients


# -----------------------------
# Simple Precision@K evaluation
# -----------------------------
def precision_at_k(
    ranked_indices: List[int],
    df: pd.DataFrame,
    patient: Dict[str, Any],
    k: int,
) -> float:
    top_idx = ranked_indices[:k]
    if not top_idx:
        return 0.0

    relevant_count = 0
    for idx in top_idx:
        row = df.iloc[idx]
        if trial_relevant_for_patient(patient, row):
            relevant_count += 1

    return relevant_count / len(top_idx)


def run_evaluation(
    csv_path: str = "data/compact_trials_small.csv",
    top_k: int = 5,
    candidate_k: int = 50,
):
    # 1. read data + build fullText
    df = load_trials(csv_path)
    df = build_full_text(df)

    print(f"[INFO] Loaded {len(df)} trials from {csv_path}")

    # 2. TF-IDF baseline + Hybrid model
    tfidf = TFIDFModel(df)
    tfidf.fit()
    hybrid = HybridModel(tfidf, alpha=0.7, beta=0.3)

    # 3. Build synthetic patients
    patients = build_synthetic_patients()

    # 4. Compare patient cases baseline vs hybrid
    baseline_scores = []
    hybrid_scores = []

    for p in patients:
        print(f"\n=== {p['name']} ===")
        query_text = p["query_text"]

        # ---- baseline: TF-IDF only ----
        tfidf_results = tfidf.query(query_text, top_k=candidate_k)
        tfidf_indices = [r["index"] for r in tfidf_results]
        prec_tfidf = precision_at_k(tfidf_indices, df, p, top_k)
        baseline_scores.append(prec_tfidf)
        print(f"TF-IDF Precision@{top_k}: {prec_tfidf:.3f}")

        # ---- hybrid: TF-IDF + rules ----
        hybrid_results = hybrid.query(
            patient=p,
            query_text=query_text,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        hybrid_indices = [r["index"] for r in hybrid_results]
        prec_hybrid = precision_at_k(hybrid_indices, df, p, top_k)
        hybrid_scores.append(prec_hybrid)
        print(f"Hybrid Precision@{top_k}: {prec_hybrid:.3f}")

        print("\nTop-3 Hybrid Trials:")
        for r in hybrid_results[:3]:
            print(f"- {r['nctId']} | {r['briefTitle'][:80]}...")
            print(f"  tfidf={r['tfidf_score']:.3f}, semantic={r['semantic_score']:.3f}, hybrid={r['hybrid_score']:.3f}")

    if baseline_scores:
        print("\n=== Average over patients ===")
        print(f"Avg TF-IDF Precision@{top_k}: {np.mean(baseline_scores):.3f}")
        print(f"Avg Hybrid Precision@{top_k}: {np.mean(hybrid_scores):.3f}")


if __name__ == "__main__":
    run_evaluation()
