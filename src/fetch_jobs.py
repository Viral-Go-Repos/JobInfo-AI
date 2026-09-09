# src/fetch_jobs.py
"""
Fetch job postings from RemoteOK public API and save to parquet for caching.
No API key required.
"""
import requests
from pathlib import Path
from typing import Optional
import pandas as pd
import time
import logging

REMOTEOK_API = "https://remoteok.com/api"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fetch_jobs")


def fetch_remoteok_jobs(save_path: str = "data/raw/remoteok_jobs.parquet",
                        max_jobs: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch job postings from RemoteOK public JSON endpoint and save as parquet.
    Returns a pandas DataFrame.
    """
    log.info("Requesting RemoteOK API...")
    resp = requests.get(REMOTEOK_API, headers={"User-Agent": "ai-job-dashboard/0.1"})
    resp.raise_for_status()
    data = resp.json()

    # RemoteOK's JSON contains metadata in the first element sometimes; filter entries with 'id'
    jobs = [j for j in data if isinstance(j, dict) and j.get("id")]
    df = pd.DataFrame(jobs)

    if max_jobs:
        df = df.head(max_jobs)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(save_path, index=False)
    log.info(f"Saved {len(df)} jobs to {save_path}")
    return df


if __name__ == "__main__":
    df = fetch_remoteok_jobs(max_jobs=500)
    print("Fetched:", len(df))
    print(df.columns.tolist())
