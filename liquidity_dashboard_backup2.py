import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Polymarket Liquidity Toolkit", layout="wide")

@st.cache_data
def load_data(path="/home/kaibrusch/code/liquidity_snapshots.parquet"):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

with st.sidebar:
    reload = st.checkbox("↻ Reload snapshots", value=False)
if reload:
    st.cache_data.clear()

df = load_data()

now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
latest_ts = df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")

st.sidebar.markdown(f"🕒 **Current UTC Time:** `{now_utc}`")
st.sidebar.markdown(f"📸 **Latest Snapshot:** `{latest_ts} UTC`")

page = st.sidebar.radio("📂 Navigate", ["Liquidity Explorer", "Liquidity Analyzer"])


if page == "Liquidity Explorer":
    
    

    latest_df = df.sort_values("timestamp").groupby("token_id").tail(1).copy()

    st.markdown("#### 🔎 Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        spread_min = st.slider("Min Normalized Spread", 0.0, 2.0, 0.0, step=0.01)
    with col2:
        liquidity_min = st.slider("Min Liquidity", 1, 100000, 1)
    with col3:
        imbalance_min = st.slider("Min Order Book Imbalance", 0.0, 1.0, 0.0, step=0.01)

    active_only = st.checkbox("Only Active Markets (last hour)", value=True)
    keyword = st.text_input("Search outcome or market (keyword match):", "")

    filtered = latest_df[
        (latest_df["normalized_spread"] >= spread_min) &
        (latest_df["liquidityNum"] >= liquidity_min) &
        (abs(latest_df["order_book_imbalance"]) >= imbalance_min)
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

    all_cols = filtered.columns.tolist()
    default_cols = [
        "market_id", "question", "outcome_name",
        "normalized_spread", "liquidityNum",
        "top_of_book_depth_bid", "top_of_book_depth_ask"
    ]
    selected_cols = st.multiselect("Columns to Display", all_cols, default=default_cols)

    sort_by = st.selectbox("Sort by", ["spread_bps", "volumeNum", "order_book_imbalance"])

    st.dataframe(
        filtered.sort_values(sort_by, ascending=False)[selected_cols],
        use_container_width=True
    )

elif page == "Liquidity Analyzer":
    col1, col2 = st.columns(2)
    with col1:
        market_id_input = st.text_input("Market ID", value="525104")
    with col2:
        outcome_input = st.text_input("Outcome keyword", value="Yes")

    if not market_id_input or not outcome_input:
        st.warning("Both Market ID and Outcome keyword are required.")
    else:
        df_filtered = df.copy()
        df_filtered = df_filtered[
            df_filtered["market_id"].astype(str).str.contains(market_id_input)
        ]
        df_filtered = df_filtered[
            df_filtered["outcome_name"].str.lower().str.contains(outcome_input.lower())
        ]
        df_filtered = df_filtered.sort_values("timestamp")

        if df_filtered.empty:
            st.warning("No matching market outcomes found.")
        else:
            question = df_filtered['question'].iloc[0]
            outcome = df_filtered['outcome_name'].iloc[0]
            last_ts = df_filtered['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')

            st.markdown(f"📌 **Market:** `{question}`")
            st.markdown(f"🏷️ **Outcome:** `{outcome}`")
            st.markdown(f"🕒 **Last Snapshot:** `{last_ts} UTC`")

            metric_cols = [
                "bid_ask_spread", "normalized_spread",
                "order_book_imbalance", "price_impact_buy_100", "price_impact_sell_100"
            ]
            selected_metrics = st.multiselect(
                "Select metrics to plot",
                metric_cols,
                default=["normalized_spread", "order_book_imbalance"]
            )

            if selected_metrics:
                fig = px.line(
                    df_filtered.melt(id_vars="timestamp", value_vars=selected_metrics),
                    x="timestamp", y="value", color="variable",
                    labels={"variable": "Metric", "value": "Value"},
                    title=None
                )
                fig.update_layout(height=500, template="simple_white")
                fig.update_xaxes(tickformat="%b %d", title_text="Date")
                st.plotly_chart(fig, use_container_width=True)

            # Second plot for top of book depth
            fig_depth = px.line(
                df_filtered.melt(
                    id_vars="timestamp",
                    value_vars=["top_of_book_depth_bid", "top_of_book_depth_ask"]
                ),
                x="timestamp", y="value", color="variable",
                labels={"variable": "Side", "value": "Depth"},
                title=None
            )
            fig_depth.update_layout(height=500, template="simple_white")
            fig_depth.update_xaxes(tickformat="%b %d", title_text="Date")
            st.plotly_chart(fig_depth, use_container_width=True)
