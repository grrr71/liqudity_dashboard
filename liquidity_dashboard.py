import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from urllib.parse import quote
######## tester
# Set Streamlit app page title and layout
st.set_page_config(page_title="Polymarket Liquidity Toolkit", layout="wide")

# Load snapshot data from parquet file and convert timestamp
@st.cache_data
def load_data(path="/home/kaibrusch/code/liquidity_snapshots.parquet"):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

# Parse query parameters to enable page linking and state pre-fill
params = st.experimental_get_query_params()
market_id_override = params.get("market_id", [None])[0]
outcome_override = params.get("outcome", [None])[0]
page_override = params.get("page", [None])[0]

now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
latest_snapshot = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')

# Sidebar navigation and timestamp info
with st.sidebar:
    st.markdown("### ⏱️ Timestamps")
    st.markdown(
        f"<span style='font-size:13px; font-family:monospace;'>🕒 Current UTC Time:<br>{now_utc} UTC</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='font-size:13px; font-family:monospace;'>📸 Latest Snapshot:<br>{latest_snapshot} UTC</span>",
        unsafe_allow_html=True
    )
    # Set the page based on query parameters or user selection
    if page_override in ["Liquidity Explorer", "Liquidity Analyzer"]:
        page = page_override
    else:
        page = st.radio("📂 Navigate", ["Liquidity Explorer", "Liquidity Analyzer"], index=0, key="page_selector")

# -----------------------------
# PAGE 1: LIQUIDITY EXPLORER
# -----------------------------
if page == "Liquidity Explorer":
    st.markdown("# Liquidity Explorer")
    st.markdown("Use the filters to explore the latest snapshot of all token outcomes.")

    # Generate latest snapshot per token ID
    latest_df = df.sort_values("timestamp").groupby("token_id").tail(1).copy()

    # Render filter controls
    st.markdown("#### 🔎 Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        spread_slider = st.slider("Normalized Spread", 0.0, 2.0, (0.0, 2.0))
    with col2:
        liquidity_slider = st.slider("Liquidity", 0.0, latest_df["liquidityNum"].max(), (0.0, latest_df["liquidityNum"].max()))
    with col3:
        imbalance_slider = st.slider("Order Book Imbalance", -1.0, 1.0, (-1.0, 1.0), step=0.01)

    active_only = st.checkbox("Only Active Markets (last hour)", value=True)
    keyword = st.text_input("Search outcome or market (keyword match):", "")

    filtered = latest_df[
        (latest_df["normalized_spread"] >= spread_slider[0]) & (latest_df["normalized_spread"] <= spread_slider[1]) &
        (latest_df["liquidityNum"] >= liquidity_slider[0]) & (latest_df["liquidityNum"] <= liquidity_slider[1]) &
        (latest_df["order_book_imbalance"] >= imbalance_slider[0]) & (latest_df["order_book_imbalance"] <= imbalance_slider[1])
    ]

    if active_only:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        filtered = filtered[filtered["timestamp"] >= cutoff]

    if keyword:
        keyword_lower = keyword.lower()
        filtered = filtered[
            filtered["outcome_name"].str.lower().str.contains(keyword_lower) |
            filtered["question"].str.lower().str.contains(keyword_lower)
        ]

    st.markdown(f"**{len(filtered):,} market outcomes match your filters.**")

    # Add a column for navigation buttons
    filtered["View"] = filtered.apply(
        lambda row: f"[🔍 View](?page=Liquidity+Analyzer&market_id={quote(str(row['market_id']))}&outcome={quote(str(row['outcome_name']))})",
        axis=1
    )

    # Display the filtered data using Streamlit's built-in table
    st.table(filtered[["market_id", "question", "outcome_name", "normalized_spread", "liquidityNum", "View"]])

# -----------------------------
# PAGE 2: LIQUIDITY ANALYZER
# -----------------------------
elif page == "Liquidity Analyzer":
    st.markdown("# Liquidity Analyzer")
    st.markdown("Search for a market outcome to analyze its liquidity over time.")

    # Pre-fill inputs with query parameters
    market_id_input = st.text_input("Market ID", value=market_id_override or "")
    outcome_input = st.text_input("Outcome keyword", value=outcome_override or "")

    # Filter the data based on inputs
    search_results = df.copy()
    if market_id_input:
        search_results = search_results[search_results["market_id"].astype(str).str.contains(market_id_input)]
    if outcome_input:
        search_results = search_results[search_results["outcome_name"].str.lower().str.contains(outcome_input.lower())]

    df_token = search_results.sort_values("timestamp")

    if not df_token.empty:
        st.markdown("### 📊 Liquidity Metrics")
        st.dataframe(df_token)
    else:
        st.warning("No data matches your filters or search.")
