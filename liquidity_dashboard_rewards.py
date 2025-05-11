import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Polymarket Liquidity Toolkit", layout="wide")

@st.cache_data
def load_data(path="/home/kaibrusch/code/liquidity_snapshots2.parquet"):
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
st.sidebar.markdown(f"📸 **Latest Snapshot:/n** `{latest_ts} UTC`")

page = st.sidebar.radio("📂 Navigate", ["Liquidity Explorer", "Liquidity Analyzer"])

if page == "Liquidity Explorer":# Select latest snapshot for each token_id
    
   
    latest_df = df.sort_values("timestamp").groupby("token_id").tail(1).copy()

    # Filters
    st.markdown("#### 🔎 Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        spread_platform_range = st.slider(
            "Platform Spread (Range)", 0.0, 1.0, (0.0, 1.0), step=0.01
        )
    with col2:
        bid_ask_spread_range = st.slider(
            "Bid-Ask Spread (Range)", 0.0, 1.0, (0.0, 1.0), step=0.01
        )
    with col3:
        volume_range = st.slider(
            "24h Volume (USDC)", 
            1.0, 
            1_000_000.0, 
            (1.0, 1_000_000.0), 
            step=100.0
        )

    active_only = st.checkbox("Only Active Markets (last hour)", value=True)
    

    volume_max = volume_range[1]
    if volume_max == 1_000_000.0:
        volume_filter = latest_df["volume24hr"] >= volume_range[0]
    else:
        volume_filter = latest_df["volume24hr"].between(*volume_range)
    # Filtering logic
    filtered = latest_df[
        latest_df["spread_platform"].between(*spread_platform_range) &
        latest_df["bid_ask_spread"].between(*bid_ask_spread_range) &
        volume_filter
    ]

    if active_only:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        filtered = filtered[filtered["timestamp"] >= cutoff]
    
   
    st.markdown(f"**{len(filtered):,} market outcomes match your filters.**")
    st.markdown("#### 📊 Metric Distributions")
    plot_col1, plot_col2, plot_col3 = st.columns(3)

    with plot_col1:
        fig1 = px.histogram(filtered, x="spread_platform", nbins=30, title="Platform Spread")
        fig1.update_layout(height=250, margin=dict(t=30, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with plot_col2:
        fig2 = px.histogram(filtered, x="bid_ask_spread", nbins=30, title="Bid-Ask Spread")
        fig2.update_layout(height=250, margin=dict(t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with plot_col3:
        fig3 = px.histogram(filtered, x="volume24hr", nbins=30, title="24h Volume (USDC)")
        fig3.update_layout(height=250, margin=dict(t=30, b=20))
        # removed log scale
        st.plotly_chart(fig3, use_container_width=True)

    keyword = st.text_input("Search outcome or market (keyword match):", "")
    if keyword:
        keyword_lower = keyword.lower()
        filtered = filtered[
            filtered["outcome_name"].str.lower().str.contains(keyword_lower) |
            filtered["question"].str.lower().str.contains(keyword_lower)
        ]

    # Display table
    all_cols = filtered.columns.tolist()
    default_cols = [
        "market_id", "volume24hr", "outcome_name", "spread_platform", "bid_ask_spread",
        "bid_slope_5", "ask_slope_5", "order_book_imbalance", "rewards_min_size", "slug"
    ]
    selected_cols = st.multiselect("Columns to Display", all_cols, default=default_cols)

    st.dataframe(filtered[selected_cols], use_container_width=True)

elif page == "Liquidity Analyzer":
    col1, col2 = st.columns(2)
    with col1:
        market_id_input = st.text_input("Market ID", value="525104")
    with col2:
        outcome_input = st.text_input("Outcome keyword", value="Yes")

    if not market_id_input or not outcome_input:
        st.warning("Both Market ID and Outcome keyword are required.")
    else:
        df_filtered = df[
            df["market_id"].astype(str).str.contains(market_id_input) &
            df["outcome_name"].str.lower().str.contains(outcome_input.lower())
        ].sort_values("timestamp")

        if df_filtered.empty:
            st.warning("No matching market outcomes found.")
        else:
            latest_row = df_filtered.iloc[-1]
            question = latest_row["question"]
            outcome = latest_row["outcome_name"]
            last_ts = latest_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            reward_min = latest_row.get("rewards_min_size", "N/A")
            reward_max = latest_row.get("rewards_max_spread", "N/A")
            volume_24h = latest_row.get("volume24hr", 0.0)

            st.markdown(f"""
            <table style='width:100%; font-size: 16px;'>
              <tr><td>📌 <strong>Market:</strong></td><td>{question}</td></tr>
              <tr><td>🏷️ <strong>Outcome:</strong></td><td>{outcome}</td></tr>
              <tr><td>💰 <strong>Reward Range:</strong></td><td>{reward_min} – {reward_max}</td></tr>
              <tr><td>📊 <strong>24h Volume:</strong></td><td>{volume_24h:,.2f}</td></tr>
              <tr><td>🕒 <strong>Last Snapshot:</strong></td><td>{last_ts} UTC</td></tr>
            </table>
            """, unsafe_allow_html=True)

            metric_cols = [
                col for col in df.columns 
                if col not in ["timestamp", "question", "market_id", "outcome_name"]
            ]

            # First plot
            selected_metrics = st.multiselect(
                "Select metrics for first plot (multi-select):",
                metric_cols,
                default=["normalized_spread", "order_book_imbalance"]
            )

            if selected_metrics:
                plot_df = df_filtered[["timestamp"] + selected_metrics].copy()
                plot_df = plot_df.melt(id_vars="timestamp", value_vars=selected_metrics)

                fig = px.line(
                    plot_df,
                    x="timestamp", y="value", color="variable",
                    labels={"variable": "Metric", "value": "Value"}
                )
                fig.update_layout(height=500, template="simple_white")
                fig.update_xaxes(tickformat="%b %d", title_text="Date")
                st.plotly_chart(fig, use_container_width=True)

                del plot_df
                gc.collect()

            # Second plot
            st.markdown("### 📈 Compare Two Metrics (second plot)")
            col_compare = st.multiselect(
                "Select exactly two metrics for side-by-side comparison:",
                metric_cols,
                default=["top_of_book_depth_bid", "top_of_book_depth_ask"]
            )

            if len(col_compare) == 2:
                plot_df_2 = df_filtered[["timestamp"] + col_compare].copy()
                plot_df_2 = plot_df_2.melt(id_vars="timestamp", value_vars=col_compare)

                fig_depth = px.line(
                    plot_df_2,
                    x="timestamp", y="value", color="variable",
                    labels={"variable": "Metric", "value": "Value"}
                )
                fig_depth.update_layout(height=500, template="simple_white")
                fig_depth.update_xaxes(tickformat="%b %d", title_text="Date")
                st.plotly_chart(fig_depth, use_container_width=True)

                del plot_df_2
                gc.collect()

            elif len(col_compare) > 0:
                st.warning("Please select exactly two metrics for the second plot.")