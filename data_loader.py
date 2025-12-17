import re
from typing import List

import numpy as np
import pandas as pd


# -----------------------------
# helper: parse age like "18 Years", "6 Months" → float(years)
# -----------------------------

def _parse_age_to_years(s: str) -> float:
    """
    Convert ClinicalTrials.gov age strings to years.

    Examples:
        "18 Years"   -> 18.0
        "6 Months"   -> 0.5
        "3 Weeks"    -> ~0.057
        "N/A" / NaN  -> np.nan
    """
    if pd.isna(s):
        return np.nan

    s = str(s).strip()
    if s.upper() in {"N/A", "NA", ""}:
        return np.nan

    m = re.match(r"(\d+)\s+(\w+)", s)
    if not m:
        return np.nan

    value = float(m.group(1))
    unit = m.group(2).lower()

    if "year" in unit:
        return value
    if "month" in unit:
        return value / 12.0
    if "week" in unit:
        return value / 52.0
    if "day" in unit:
        return value / 365.0

    return np.nan


# -----------------------------
# main loader
# -----------------------------

def load_trials(csv_path: str) -> pd.DataFrame:
    """
    Load compact trial CSV and perform basic cleaning / feature engineering.

    - Ensure important text columns exist and fill NaN with "".
    - Convert enrollmentCount to numeric.
    - Parse minAge / maxAge to minAgeYears / maxAgeYears.
    - Normalize sexes to ALL / MALE / FEMALE.
    """
    df = pd.read_csv(csv_path)

    # 1) make sure important string columns exist
    string_cols: List[str] = [
        "nctId",
        "briefTitle",
        "briefSummary",
        "detailedDescription",
        "conditions",
        "overallStatus",
        "phases",
        "studyType",
        "targetDuration",
        "eligibilityCriteria",
        "minAge",
        "maxAge",
        "sexes",
    ]

    for col in string_cols:
        if col not in df.columns:
            df[col] = ""

    # string columns NaN → ""
    df[string_cols] = df[string_cols].fillna("")

    # 2) enrollmentCount conversion
    if "enrollmentCount" in df.columns:
        df["enrollmentCount"] = pd.to_numeric(
            df["enrollmentCount"], errors="coerce"
        )
    else:
        df["enrollmentCount"] = np.nan

    # 3) parse minAge / maxAge to numeric years
    df["minAgeYears"] = df["minAge"].apply(_parse_age_to_years)
    df["maxAgeYears"] = df["maxAge"].apply(_parse_age_to_years)

    # 4) normalize sexes
    df["sexes"] = (
        df["sexes"]
        .fillna("ALL")
        .astype(str)
        .str.upper()
        .replace({"BOTH": "ALL"})
    )

    return df


def build_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge relevant text fields into a single 'fullText' field for TF-IDF / embeddings.

    We concatenate:
        briefTitle + briefSummary + detailedDescription + eligibilityCriteria + conditions
    """
    text_cols = [
        "briefTitle",
        "briefSummary",
        "detailedDescription",
        "eligibilityCriteria",
        "conditions",
    ]

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""

    df["fullText"] = (
        df[text_cols]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return df

