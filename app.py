# app.py
"""
Streamlit dashboard for AI Job Postings Dashboard (RemoteOK).
Robust version with safer handling of array-like fields and missing columns.
Run with: streamlit run app.py
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import numpy as np

from src.fetch_jobs import fetch_remoteok_jobs
from src.parse_job import extract_text_fields, extract_skills_from_text, clean_html_description
from src.analyze import top_skills, normalize_dates, skill_trend

# Paths
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw"
RAW_PATH.mkdir(parents=True, exist_ok=True)
CACHE_PATH = RAW_PATH / "remoteok_jobs.parquet"

# Streamlit page config
st.set_page_config(layout="wide", page_title="AI Job Postings Dashboard")
st.title("🚀 AI Job Postings Dashboard")

# Sidebar controls
st.sidebar.header("Data controls")
if st.sidebar.button("Fetch latest RemoteOK jobs"):
    with st.spinner("Fetching RemoteOK data..."):
        try:
            df = fetch_remoteok_jobs(save_path=str(CACHE_PATH), max_jobs=2000)
            st.sidebar.success(f"Fetched {len(df)} jobs")
        except Exception as e:
            st.sidebar.error(f"Fetch failed: {e}")
            st.stop()
else:
    if CACHE_PATH.exists():
        try:
            df = pd.read_parquet(CACHE_PATH)
            st.sidebar.info(f"Loaded cached {len(df)} jobs")
        except Exception as e:
            st.sidebar.error(f"Failed to load cache: {e}")
            df = pd.DataFrame()
    else:
        st.info("No cached data yet. Click 'Fetch latest RemoteOK jobs' in the sidebar.")
        st.stop()

# If empty DataFrame, stop early
if df is None or df.empty:
    st.warning("Dataset is empty. Fetch data or check cache.")
    st.stop()

# --- Preprocessing (defensive) ---
df = df.copy()

# Ensure description column exists
if "description" not in df.columns:
    df["description"] = ""

# Clean description safely
df["description_clean"] = df["description"].apply(clean_html_description)

# Build a "full_text" for extraction - using safe extract_text_fields
def safe_full_text(row):
    try:
        return extract_text_fields(row)
    except Exception:
        # fallback: try to combine some common fields
        parts = []
        for k in ("position", "company", "location", "description", "tags"):
            v = row.get(k) if hasattr(row, "get") else row.get(k, "")
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                parts.append(" ".join([str(x) for x in v if x is not None]))
            elif pd.isna(v):
                continue
            else:
                parts.append(str(v))
        return " \n ".join(parts)

df["full_text"] = df.apply(lambda r: safe_full_text(r), axis=1)

# Extract skills (safe)
def safe_extract_skills(text):
    try:
        return extract_skills_from_text(text)
    except Exception:
        return []

df["extracted_skills"] = df["full_text"].apply(safe_extract_skills)

# --- Sidebar filters ---
st.sidebar.header("Filters")
# Build safe all_skills list
all_skills_set = set()
for lst in df["extracted_skills"].dropna():
    if isinstance(lst, (list, tuple, set, pd.Series, np.ndarray)):
        for s in lst:
            if s:
                all_skills_set.add(s)
    else:
        for s in str(lst).split(","):
            s = s.strip()
            if s:
                all_skills_set.add(s)
all_skills = sorted(all_skills_set)

skill_filter = st.sidebar.multiselect("Filter by skill", options=all_skills, default=[])
company_filter = st.sidebar.text_input("Company contains")
loc_filter = st.sidebar.text_input("Location contains")

# Apply filters defensively
filtered = df.copy()
if skill_filter:
    def has_all_skills(row_skills):
        if isinstance(row_skills, (list, tuple, set, pd.Series, np.ndarray)):
            row_set = set([str(x) for x in row_skills])
            return all(k in row_set for k in skill_filter)
        else:
            txt = str(row_skills)
            return all(k in txt for k in skill_filter)
    filtered = filtered[filtered["extracted_skills"].apply(has_all_skills)]

if company_filter:
    filtered = filtered[filtered["company"].astype(str).str.contains(company_filter, case=False, na=False)]

if loc_filter:
    filtered = filtered[filtered["location"].astype(str).str.contains(loc_filter, case=False, na=False)]

# --- Visualization & UI ---
st.header("Top skills (current selection)")
ts = top_skills(filtered, top_n=30)
if ts.empty:
    st.info("No skills found in selection.")
else:
    fig = px.bar(ts, x="skill", y="count", title="Top skills")
    st.plotly_chart(fig, use_container_width=True)

st.header("Job listings (This week's postings)")
cols_to_show = []
# pick columns that exist
for c in ["date", "company", "position", "location", "tags", "extracted_skills"]:
    if c in filtered.columns:
        cols_to_show.append(c)

if cols_to_show:
    st.dataframe(filtered[cols_to_show].head(300))
else:
    st.info("No standard columns to show in table.")

# Skill trend plotting (guarded)
st.header("Skill trend")
if not ts.empty:
    skill_example = st.selectbox("Pick skill to plot", ts["skill"].tolist())
    if skill_example:
        try:
            trend_df = skill_trend(filtered, skill_example, freq="W")
            if not trend_df.empty:
                fig2 = px.line(trend_df, x="posted_at", y="count", title=f"Weekly count of jobs mentioning {skill_example}")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No trend data available for this selection.")
        except Exception as e:
            st.error(f"Failed to compute trend: {e}")
else:
    st.info("No skill trend to show.")

# Export processed dataset
st.subheader("Export data")
try:
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download current selection (CSV)", csv, file_name="jobs_filtered.csv", mime="text/csv")
except Exception as e:
    st.error(f"Failed to prepare export: {e}")
