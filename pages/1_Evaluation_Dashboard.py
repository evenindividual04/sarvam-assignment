import os
import sqlite3

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Evaluation Dashboard")
st.caption("Deep Research Agent Evaluation Metrics & Failure Taxonomy")

db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "research.db")

if not os.path.exists(db_path):
    st.warning("Database not found. Please run the evaluation script first.")
    st.stop()

conn = sqlite3.connect(db_path)
# Check if eval_runs table exists
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eval_runs'")
if not cursor.fetchone():
    st.warning("No evaluation runs table found. Please run the evaluation script first.")
    st.stop()

df = pd.read_sql_query("SELECT * FROM eval_runs ORDER BY run_at DESC", conn)
conn.close()

if df.empty:
    st.info("No evaluation runs recorded yet.")
    st.stop()

# Group by unique runs using a rough timestamp prefix or just exact run_at timestamp
runs = df['run_at'].unique()
# We will just show the most recent run automatically, or let user select
selected_run = st.selectbox("Select Evaluation Run timestamp", runs)

run_df = df[df['run_at'] == selected_run]

st.header("Metrics Summary")
cols = st.columns(5)

passes = len(run_df[run_df['failure_class'] == 'PASS'])
total = len(run_df)
pass_rate = passes / total if total else 0

cols[0].metric("Pass Rate", f"{pass_rate*100:.1f}%", f"{passes}/{total} Passed")
cols[1].metric("Avg Faithfulness", f"{run_df['faithfulness_score'].mean():.2f}")
cols[2].metric("Avg Relevance", f"{run_df['answer_relevance_score'].mean():.2f}")
cols[3].metric("Avg Ctx Precision", f"{run_df['context_precision_score'].mean():.2f}")
cols[4].metric("Avg Citation Integrity", f"{run_df['citation_integrity_score'].mean():.2f}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Failure Taxonomy")
    taxonomy = run_df['failure_class'].value_counts()
    st.bar_chart(taxonomy)

with col2:
    st.subheader("Scores by Category")
    avg_scores = run_df.groupby('category')[
        ['faithfulness_score', 'answer_relevance_score', 'citation_integrity_score', 'context_precision_score']
    ].mean()
    st.bar_chart(avg_scores)

st.subheader("Detailed Run Data")
display_cols = [
    'question_id', 'category', 'question', 'faithfulness_score', 
    'answer_relevance_score', 'context_precision_score', 
    'citation_integrity_score', 'failure_class', 'latency_ms'
]
# Only display columns that exist in the dataframe (in case some runs didn't have context precision)
valid_cols = [c for c in display_cols if c in run_df.columns]
st.dataframe(run_df[valid_cols], use_container_width=True)
