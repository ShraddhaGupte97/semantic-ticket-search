import streamlit as st
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
from inference import (
    load_embeddings,
    load_ticket_ids,
    load_faiss_index,
    load_dataframe,
    build_ticket_text,
    encode_texts,
    get_reranker,
    rerank_candidates,
    find_similar_tickets
)
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Ticket Search & Analysis",
    page_icon="🔍",
    layout="wide"
)

# Load cached artifacts
@st.cache_resource
def load_resources():
    return {
        "embedding_matrix": load_embeddings("artifacts/embedding_matrix.npy"),
        "ticket_ids": load_ticket_ids("artifacts/ticket_ids.json"),
        "index": load_faiss_index("artifacts/faiss.index"),
        "reranker": get_reranker("./rank_model"),
        "ticket_df": load_dataframe("data/raw_data.json")
    }

# Load all resources
resources = load_resources()
ticket_map = {row["ticket_number"]: row for _, row in resources["ticket_df"].iterrows()}

# Title and description
st.title("🔍 Ticket Search & Analysis")
st.markdown("""
This tool helps you find similar tickets and analyze patterns in your ticket data.
Enter a ticket below to find similar tickets and view analysis.
""")

# Input form
with st.form("ticket_input"):
    st.subheader("Enter Ticket Details")
    
    col1, col2 = st.columns(2)
    with col1:
        ticket_number = st.text_input("Ticket Number")
        title = st.text_input("Title")
        metrics = st.text_input("Metrics")
        dimensions = st.text_input("Dimensions (comma separated)")
        labels = st.text_input("Labels")
        create_date = st.text_input("Create Date (YYYY-MM-DD)")
        resolve_date = st.text_input("Resolve Date (YYYY-MM-DD)")
    
    with col2:
        time_period = st.text_input("Time Period")
        status = st.selectbox("Status", ["Resolved", "Open", "In Progress"])
        assigned_folder = st.text_input("Assigned Folder")
        priority = st.selectbox("Priority", ["High", "Medium", "Low"])
        link = st.text_input("Ticket Link")
    
    st.subheader("Comments")
    comment_author = st.text_input("Comment Author")
    comment_date = st.text_input("Comment Date (YYYY-MM-DD)")
    comment_text = st.text_area("Comment Text")
    
    submitted = st.form_submit_button("Search Similar Tickets")

if submitted:
    # Create query ticket
    query_ticket = {
        "ticket_number": ticket_number,
        "title": title,
        "metrics": metrics,
        "dimensions": [d.strip() for d in dimensions.split(",") if d.strip()],
        "time_period": time_period,
        "status": status,
        "assigned_folder": assigned_folder,
        "priority": priority,
        "labels": labels,
        "create_date": create_date,
        "resolve_date": resolve_date,
        "link": link,
        "comments": [{
            "author": comment_author,
            "date": comment_date,
            "comment": comment_text
        }]
    }
    
    reranked = find_similar_tickets(query_ticket, top_k=3)
    
    # Convert scores to 0-100%
    scores = [r['score'] for r in reranked]
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        for r in reranked:
            if max_score > min_score:
                r['score_pct'] = 100 * (r['score'] - min_score) / (max_score - min_score)
            else:
                r['score_pct'] = 100.0  # All scores are the same
    else:
        for r in reranked:
            r['score_pct'] = 0.0

    # Filter out tickets with score below 40%
    reranked = [r for r in reranked if r['score_pct'] >= 40.0]

    # Calculate average resolution time (in days) for displayed tickets
    resolution_times = []
    for r in reranked:
        ticket_data = ticket_map.get(r['ticket_number'], {})
        create_date = ticket_data.get('create_date')
        resolve_date = ticket_data.get('resolve_date')
        try:
            if create_date and resolve_date:
                d1 = datetime.fromisoformat(create_date)
                d2 = datetime.fromisoformat(resolve_date)
                resolution_times.append((d2 - d1).days)
        except Exception:
            pass

    col1, col2, col3 = st.columns(3)

    with col1:
        if resolution_times:
            avg_resolution = sum(resolution_times) / len(resolution_times)
            st.info(f"**Average Resolution Time:** {avg_resolution:.1f} days")
        else:
            st.info("**Average Resolution Time:** N/A")

    with col2:
        folders = [ticket_map.get(r['ticket_number'], {}).get('assigned_folder', 'N/A') for r in reranked]
        if folders:
            most_common_folder = Counter(folders).most_common(1)[0][0]
            st.info(f"**Most Common Assigned Folder:** {most_common_folder}")

    with col3:
        dates = []
        for r in reranked:
            ticket_data = ticket_map.get(r['ticket_number'], {})
            create_date = ticket_data.get('create_date')
            try:
                if create_date:
                    dates.append(datetime.fromisoformat(create_date))
            except Exception:
                pass
        if dates:
            st.info(f"**Oldest:** {min(dates).date()} | **Newest:** {max(dates).date()}")

    # Display results
    st.subheader("Top Similar Tickets")
    for i, r in enumerate(reranked, 1):
        with st.expander(f"{i}. Ticket: {r['ticket_number']} | Similarity: {r['score_pct']:.2f}%"):
            st.write(f"**Text:** {r['text']}")
            ticket_data = ticket_map.get(r['ticket_number'], {})
            st.write(f"**Priority:** {ticket_data.get('priority', 'N/A')}")
            st.write(f"**Status:** {ticket_data.get('status', 'N/A')}")
            st.write(f"**Assigned Folder:** {ticket_data.get('assigned_folder', 'N/A')}") 