# hybrid_model.py
from typing import Dict, Any, List, Tuple

import numpy as np

from tfidf_model import TFIDFModel


def rule_based_semantic_score_with_explanation(
    patient: Dict[str, Any],
    row
) -> Tuple[float, str]:
    """
    Based on simple rules, compute a semantic matching score between patient and trial.
    returns:
      score (0~1)
      explanation
    """
    explanations = []
    score = 0.0
    max_score = 0.0

    # ---------- 1. Age ----------
    age = patient.get("age")
    min_age = row.get("minAgeYears", np.nan)
    max_age = row.get("maxAgeYears", np.nan)

    if age is not None and not np.isnan(age):
        max_score += 1.0
        ok = True
        if not np.isnan(min_age) and age < min_age:
            ok = False
        if not np.isnan(max_age) and age > max_age:
            ok = False

        if ok:
            score += 1.0
            explanations.append(
                f"✅ Age {age} year olds within trial requirements"
                + (
                    f" [{min_age:.0f}, {max_age:.0f}] years old"
                    if not np.isnan(min_age) or not np.isnan(max_age)
                    else ""
                )
            )
        else:
            explanations.append(
                f"⚠️ Age {age} year olds not within trial requirements"
                + (
                    f" [{min_age:.0f}, {max_age:.0f}] years old"
                    if not np.isnan(min_age) or not np.isnan(max_age)
                    else ""
                )
            )
    else:
        explanations.append("ℹ️ No age information provided by patient")

    # ---------- 2. Sex ----------
    sex = str(patient.get("sex", "UNKNOWN")).upper()
    trial_sex = str(row.get("sexes", "ALL")).upper()

    if trial_sex in {"MALE", "FEMALE"} and sex in {"MALE", "FEMALE"}:
        max_score += 1.0
        if sex == trial_sex:
            score += 1.0
            explanations.append(
                f"✅ Sex matches: patient {sex}, trial only recruits {trial_sex}"
            )
        else:
            explanations.append(
                f"⚠️ Sex mismatch: patient {sex}, trial only recruits {trial_sex}"
            )
    else:
        explanations.append(
            f"ℹ️ Sex restriction is loose: trial recruits {trial_sex} (typically means ALL or no strict restriction)"
        )

    # ---------- 3. Conditions / Keywords ----------
    patient_terms = set(
        t.lower()
        for t in (patient.get("conditions") or []) + (patient.get("keywords") or [])
        if t
    )

    if patient_terms:
        max_score += 1.0
        trial_text = (
            str(row.get("conditions", "")) + " " +
            str(row.get("eligibilityCriteria", ""))
        ).lower()

        hits = [term for term in patient_terms if term in trial_text]
        if hits:
            score += len(hits) / len(patient_terms)
            explanations.append(
                f"✅ Conditions/keywords matched: {len(hits)}/{len(patient_terms)} keywords found in trial conditions "
                f"（hits: {', '.join(hits)}）"
            )
        else:
            explanations.append(
                "⚠️ Patient-provided conditions/keywords not found in trial conditions"
            )
    else:
        explanations.append("ℹ️ Patient did not provide explicit conditions/keywords, cannot do text matching")

    # ---------- Final Score ----------
    if max_score > 0:
        final_score = score / max_score
    else:
        final_score = 0.0

    explanation_text = " | ".join(explanations)
    return final_score, explanation_text


def rule_based_semantic_score(patient: Dict[str, Any], row) -> float:
    """
    Keep for compatibility: only return the score without explanation.
    """
    score, _ = rule_based_semantic_score_with_explanation(patient, row)
    return score


class HybridModel:
    """
    Hybrid model combining TF-IDF scores with simple rule-based semantic matching.
    """

    def __init__(self, tfidf_model: TFIDFModel, alpha: float = 0.7, beta: float = 0.3):
        """
        tfidf_model: Fitted TFIDFModel
        alpha: TF-IDF weight
        beta: semantic weight
        """
        self.tfidf_model = tfidf_model
        self.df = tfidf_model.df
        self.alpha = alpha
        self.beta = beta

    def query(
        self,
        patient: Dict[str, Any],
        query_text: str | None = None,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Given a patient profile, return a sorted list of trials.

        patient: structured patient information (age, sex, conditions, keywords)
        query_text: query text for TF-IDF; if None, will be constructed from conditions+keywords
        top_k: number of final results to return
        candidate_k: number of candidates to fetch from TF-IDF before hybrid re-ranking
        """
        if query_text is None:
            conds = patient.get("conditions") or []
            kws = patient.get("keywords") or []
            query_text = " ".join(conds + kws) or " ".join(conds) or " ".join(kws)

        tfidf_results = self.tfidf_model.query(query_text, top_k=candidate_k)

        enriched: List[Dict[str, Any]] = []

        for r in tfidf_results:
            idx = r["index"]
            row = self.df.iloc[idx]

            semantic_score, explanation = rule_based_semantic_score_with_explanation(
                patient, row
            )
            tfidf_score = r["tfidf_score"]

            hybrid_score = self.alpha * tfidf_score + self.beta * semantic_score

            enriched.append(
                {
                    "index": idx,
                    "nctId": r.get("nctId"),
                    "briefTitle": r.get("briefTitle", ""),
                    "tfidf_score": tfidf_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": hybrid_score,
                    "explanation": explanation,
                }
            )

        enriched_sorted = sorted(
            enriched, key=lambda x: x["hybrid_score"], reverse=True
        )

        return enriched_sorted[:top_k]

