
# Job Info AI Dashboard

An open, interpretable pipeline that collects job postings, extracts in-demand technical skills using rule-based NLP, and visualizes hiring trends in a Streamlit dashboard.


**Why this project**
- Help learners and professionals discover which technical skills employers are hiring for.
- Turn unstructured job descriptions into actionable, time-aware insights.

**Highlights**
- Lightweight, rule-based NLP for transparent skill extraction
- Simple data pipeline: fetch → clean → extract → analyze → visualize
- Streamlit dashboard for interactive exploration and exports

**Quick Start**

1. Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\\Scripts\\activate    # Windows
source venv/bin/activate # macOS / Linux
```

2. Install dependencies and download spaCy model:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the dashboard:

```bash
streamlit run app.py
```

Project layout

```
./
├─ app.py                # Streamlit dashboard
├─ requirements.txt
├─ src/
│  ├─ fetch_jobs.py      # fetches and caches job postings
│  ├─ parse_job.py       # text cleaning and skill extraction
│  └─ analyze.py         # aggregation and trend analysis
└─ data/raw/             # cached raw job JSON
```

How it works

- Data ingestion: pulls job postings from a public API and writes cached JSON to `data/raw/`.
- Cleaning: strips HTML, normalizes text, and concatenates relevant fields for parsing.
- Skill extraction: uses a curated skill dictionary and rule-based matching to tag each posting.
- Analysis: aggregates skill counts and computes weekly trends.
- Visualization: interactive plots and a filterable job table with CSV export.

Key features

- Top skills leaderboard and time-series trends
- Filter by skills, company, and location
- Browse job listings and export filtered results as CSV

Development notes

- The extraction logic is intentionally rule-based for interpretability; swap in ML models later if desired.
- Store larger datasets in Parquet when scaling beyond CSV.

Contributing

- Bug reports and PRs welcome. Please open issues or PRs with clear descriptions.

License

This repository is provided for educational and research purposes.

Credits

- RemoteOK (or configured data source) for public job data
- Open-source Python ecosystem (pandas, spaCy, Streamlit, Plotly)
