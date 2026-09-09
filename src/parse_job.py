# src/parse_job.py
"""
Parsing helpers for job postings.
Provides:
 - clean_html_description(text)
 - extract_text_fields(job_row)
 - extract_skills_from_text(text)
Optional: a small spaCy-based noun chunk extractor (if spaCy installed).
"""

import re
import html
from typing import List, Set, Any
import numpy as np
import pandas as pd

# Optional spaCy (not required)
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except Exception:
    _nlp = None

# Curated skill keywords (expand as needed)
SKILL_KEYWORDS: Set[str] = {
    # languages & frameworks
    "python", "java", "javascript", "typescript", "react", "reactjs", "node", "node.js",
    "express", "django", "flask", "angular", "vue", "nextjs",
    # cloud & infra
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    # db & data
    "sql", "postgres", "postgresql", "mysql", "mongodb", "redis",
    # ml / data stack
    "tensorflow", "pytorch", "scikit-learn", "spark", "airflow",
    # misc
    "graphql", "rest api", "rest", "ci/cd", "git", "redis"
}


def clean_html_description(html_text: Any) -> str:
    """
    Safely remove HTML tags, decode entities, collapse whitespace.
    Accepts None, numeric, lists, arrays, and converts them to text.
    """
    if html_text is None:
        return ""
    # If array-like or series, join
    if isinstance(html_text, (list, tuple, np.ndarray, pd.Series)):
        try:
            html_text = " ".join([str(x) for x in html_text if x is not None])
        except Exception:
            html_text = str(html_text)
    text = str(html_text)
    text = html.unescape(text)
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_fields(job_row: Any) -> str:
    """
    Combine common job fields into a single cleaned text blob.
    job_row: pandas.Series or dict-like
    """
    parts = []

    def _get_val(r, k):
        if hasattr(r, "get"):
            return r.get(k, None)
        try:
            return r[k]
        except Exception:
            return None

    for key in ("position", "company", "location", "description", "tags"):
        v = _get_val(job_row, key)
        if v is None:
            continue
        # skip NaN
        if isinstance(v, float) and pd.isna(v):
            continue
        # if list-like, join
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            items = []
            for it in v:
                if it is None:
                    continue
                items.append(str(it))
            if items:
                parts.append(" ".join(items))
        else:
            s = str(v).strip()
            if s and s.lower() != "nan":
                parts.append(s)

    joined = " \n ".join(parts)
    return clean_html_description(joined)


def extract_skills_from_text(text: Any, keywords: Set[str] = SKILL_KEYWORDS) -> List[str]:
    """
    Keyword-based skill extraction.
    Returns a sorted list of matched keywords (lowercase).
    Works with text that may be list/array by converting to string.
    """
    if text is None:
        return []
    if isinstance(text, (list, tuple, np.ndarray, pd.Series)):
        try:
            text = " ".join([str(x) for x in text if x is not None])
        except Exception:
            text = str(text)
    t = str(text).lower()
    found = {kw for kw in keywords if kw in t}
    return sorted(found)


def extract_noun_chunks(text: str, max_chunks: int = 20) -> List[str]:
    """
    Optional helper: extract noun chunks using spaCy (if available).
    """
    if _nlp is None or not text:
        return []
    try:
        doc = _nlp(text[:20000])
        chunks = [c.text.strip().lower() for c in doc.noun_chunks][:max_chunks]
        return chunks
    except Exception:
        return []
