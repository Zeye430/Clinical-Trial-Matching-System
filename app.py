# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_trials, build_full_text
from tfidf_model import TFIDFModel
from hybrid_model import HybridModel
from semantic_llm import extract_patient_info


# -----------------------------
# Load data & models (cache)
# -----------------------------
@st.cache_resource
def load_model():
    csv_path = "data/compact_trials_small.csv"

    df = load_trials(csv_path)
    df = build_full_text(df)

    tfidf = TFIDFModel(df)
    tfidf.fit()

    hybrid = HybridModel(tfidf, alpha=0.7, beta=0.3)
    return df, hybrid


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Clinical Trial Matcher", layout="wide")
    st.title("🧪 Clinical Trial Matching Demo")

    st.markdown(
        "This demo showcases: Data Extraction → TF-IDF Retrieval → Hybrid Rule-Based Model → LLM Patient Parsing → Visualization & Explanation."
    )

    df, hybrid = load_model()

    col_left, col_right = st.columns([1, 2])

    # -------- Left: Input --------
    with col_left:
        st.subheader("1️⃣ Input Patient Information")

        st.markdown("**(A) Paste Natural Language Description (Optional)**")
        raw_text = st.text_area(
            "Patient description (English works best)",
            height=150,
            placeholder=(
                "e.g. 60 year old male with long-standing type 2 diabetes and "
                "chronic kidney disease stage 3..."
            ),
        )

        use_llm = st.checkbox("Use LLM to parse description", value=True)

        st.markdown("**(B) Manually Supplement/Override Structured Info**")

        age_input = st.number_input("Age (years)", min_value=0, max_value=120, value=60)
        sex_input = st.selectbox("Sex", ["UNKNOWN", "MALE", "FEMALE"], index=1)
        cond_input = st.text_input(
            "Conditions (comma separated)",
            value="type 2 diabetes, chronic kidney disease",
        )

        kw_input = st.text_input(
            "Additional keywords (comma separated, optional)",
            value="",
        )

        top_k = st.slider("Top-K trials to show", min_value=3, max_value=20, value=5)

        run_btn = st.button("🔍 Match Trials")

    # -------- Right: Results --------
    with col_right:
        st.subheader("2️⃣ Matching Results & Explanations")

        # Score Explanation
        with st.expander("ℹ️ Score Explanation (Click to expand)", expanded=False):
            st.markdown(
                """
**Score Definitions (0 ~ 1, Higher is Better):**

- `tfidf_score`:  
  Score based solely on text similarity (Patient Description vs. Trial Text). Higher values indicate the trial is easier to retrieve using keyword search.

- `semantic_score`:  
  Semantic match score based on rule-based logic, considering:
  - Age falling within min/max requirements;
  - Sex matching eligibility constraints;
  - Patient conditions/keywords appearing in trial conditions or eligibility criteria.
  Higher values indicate better alignment with Age/Sex/Condition constraints.

- `hybrid_score`:  
  Linear combination: `hybrid = 0.7 * tfidf_score + 0.3 * semantic_score`.  
  Balances "text relevance" with "hard constraints" for the final ranking.
"""
            )

        if run_btn:
            # 1. Init patient dict
            patient = {
                "age": None,
                "sex": "UNKNOWN",
                "conditions": [],
                "keywords": [],
            }

            # 2. LLM Parsing (Optional)
            if use_llm and raw_text.strip():
                st.markdown("**Using LLM to parse patient description...**")
                llm_patient = extract_patient_info(raw_text)
                st.write("LLM Parsed Result:", llm_patient)
                patient.update(llm_patient)

            # 3. Manual Override
            patient["age"] = int(age_input)
            patient["sex"] = sex_input

            manual_conditions = [c.strip() for c in cond_input.split(",") if c.strip()]
            manual_keywords = [k.strip() for k in kw_input.split(",") if k.strip()]

            patient["conditions"] = list(
                {*(patient.get("conditions") or []), *manual_conditions}
            )
            patient["keywords"] = list(
                {*(patient.get("keywords") or []), *manual_keywords}
            )

            st.markdown("**Final Structured Patient Data for Matching:**")
            st.json(patient)

            # 4. Construct query_text
            if raw_text.strip():
                query_text = raw_text
            else:
                query_text = " ".join(patient["conditions"] + patient["keywords"])

            # 5. Call Hybrid Model
            results = hybrid.query(
                patient=patient,
                query_text=query_text,
                top_k=top_k,
                candidate_k=50,
            )

            if not results:
                st.warning("No matching trials found. Please try modifying criteria.")
                return

            # 6. Display Table
            st.markdown("**Top Matched Trials:**")
            df_res = pd.DataFrame(results)

            show_cols = [
                "nctId",
                "briefTitle",
                "tfidf_score",
                "semantic_score",
                "hybrid_score",
            ]
            show_cols = [c for c in show_cols if c in df_res.columns]

            st.dataframe(
                df_res[show_cols].style.format(
                    {
                        "tfidf_score": "{:.3f}",
                        "semantic_score": "{:.3f}",
                        "hybrid_score": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

            # 7. Visualization
            st.subheader("📊 Hybrid Score Distribution")

            titles = [
                (t[:60] + "...") if len(t) > 60 else t
                for t in df_res["briefTitle"].tolist()
            ]
            scores = df_res["hybrid_score"].tolist()

            fig, ax = plt.subplots()
            ax.barh(titles, scores)
            ax.invert_yaxis()
            ax.set_xlabel("Hybrid Score")
            ax.set_title("Top Trials Relevance (Higher is Better)")

            st.pyplot(fig)

            # 8. Detailed Explanations
            st.subheader("📝 Trial Explanations")

            for i, r in enumerate(results, start=1):
                title = r.get("briefTitle", "")
                nct_id = r.get("nctId", "")
                expander_label = f"{i}. {title[:80]}..." if len(title) > 80 else f"{i}. {title}"

                with st.expander(expander_label, expanded=(i == 1)):
                    st.markdown(f"**NCT ID:** `{nct_id}`")
                    st.markdown(
                        f"- **tfidf_score**: {r.get('tfidf_score', 0.0):.3f}\n"
                        f"- **semantic_score**: {r.get('semantic_score', 0.0):.3f}\n"
                        f"- **hybrid_score**: {r.get('hybrid_score', 0.0):.3f}"
                    )
                    st.markdown("**Explanation:**")
                    st.markdown(r.get("explanation", "(No explanation available)"))


if __name__ == "__main__":
    main()