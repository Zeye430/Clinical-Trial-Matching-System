# tfidf_model.py
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFModel:
    """
    Simple TF-IDF retrieval model over trial texts.
    """

    def __init__(self, df, text_col: str = "fullText"):
        """
        df: pandas DataFrame of trials
        text_col: column name containing the concatenated text for each trial
        """
        self.df = df
        self.text_col = text_col
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
        self.tfidf_matrix = None

    def fit(self):
        """
        Fit TF-IDF on all trial texts.
        """
        texts = self.df[self.text_col].fillna("").astype(str).tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        return self

    def query(self, text: str, top_k: int = 10) -> List[Dict]:
        """
        Rank trials by cosine similarity between the query text
        and each trial's fullText.

        Returns a list of dicts with at least:
            - index (row index in df)
            - nctId
            - briefTitle
            - tfidf_score
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("TFIDFModel is not fitted. Call .fit() first.")

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top_k indices
        top_idx = np.argsort(scores)[::-1][:top_k]

        results: List[Dict] = []
        for idx in top_idx:
            row = self.df.iloc[idx]
            results.append(
                {
                    "index": int(idx),
                    "nctId": row.get("nctId", None),
                    "briefTitle": row.get("briefTitle", ""),
                    "tfidf_score": float(scores[idx]),
                }
            )
        return results

