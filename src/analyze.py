# src/analyze.py
"""
Aggregation helpers: normalize dates, count top skills, compute trends.
"""
import pandas as pd
from collections import Counter
from typing import List
from datetime import datetime


def normalize_dates(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Normalize various date fields into a single posted_at (UTC) and posted_date.
    """
    df = df.copy()
    if "date" in df.columns:
        df["posted_at"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    elif "created_at" in df.columns:
        df["posted_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    elif "epoch" in df.columns:
        df["posted_at"] = pd.to_datetime(df["epoch"], unit="s", utc=True, errors="coerce")
    elif "time" in df.columns:
        df["posted_at"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        # fallback: use current time
        df["posted_at"] = pd.Timestamp.utcnow()
    df["posted_date"] = df["posted_at"].dt.date
    return df


def top_skills(df: pd.DataFrame, skills_col: str = "extracted_skills", top_n: int = 30) -> pd.DataFrame:
    """
    Count most common skills in the DataFrame. extracted_skills can be list-like or comma-separated.
    """
    all_skills: List[str] = []
    for s in df.get(skills_col, pd.Series([])).dropna():
        if isinstance(s, (list, tuple)):
            all_skills.extend(s)
        else:
            all_skills.extend([x.strip() for x in str(s).split(",") if x.strip()])
    cnt = Counter(all_skills)
    most = cnt.most_common(top_n)
    return pd.DataFrame(most, columns=["skill", "count"])


def skill_trend(df: pd.DataFrame, skill: str, freq: str = "W") -> pd.DataFrame:
    """
    Return time series (resampled by freq) of how many postings mention `skill`.
    """
    df = df.copy()
    df = normalize_dates(df)
    def has_skill(row):
        s = row.get("extracted_skills", [])
        if isinstance(s, (list, tuple)):
            return skill in s
        return skill in str(s)
    df["has_skill"] = df.apply(has_skill, axis=1)
    ts = df.set_index("posted_at").resample(freq)["has_skill"].sum().rename("count").reset_index()
    return ts
