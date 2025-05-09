import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
from datetime import datetime, timedelta

st.set_page_config(page_title="Polymarket Liquidity Toolkit", layout="wide")

# ---- DuckDB Setup ---- #
parquet_path = "/home/kaibrusch/code/liquidity_snapshots2.parquet"
con = duckdb.connect()
con.execute(f"CREATE OR REPLACE VIEW liquidity AS SELECT * FROM read_parquet('{parquet_path}')")

def query_df(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

# ---- Sidebar ---- #
now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M") + " UTC"
latest_ts_raw = query_df("SELECT MAX(timestamp) AS ts FROM liquidity")['ts'][0]
latest_ts = latest_ts_raw.strftime("%Y-%m-%d %H:%M") + " UTC"

st.sidebar.markdown(f"🕒 **Current UTC Time:**<br>`{now_utc}`", unsafe_allow_html=True)
st.sidebar.markdown(f"📸 **Latest Snapshot:**<br>`{latest_ts}`", unsafe_allow_html=True)

page = st.sidebar.radio("📂 Navigate", ["Liquidity Explorer", "Liquidity Analyzer"])

# ---- Liquidity Explorer ---- #
if page == "Liquidity Explorer":
    # UI Filters
    st.markdown("#### 🔎 Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        spread_platform_range = st.slider("Platform Spread (Range)", 0.0, 1.0, (0.0, 1.0), step=0.01)
    with col2:
        bid_ask_spread_range = st.slider("Bid-Ask Spread (Range)", 0.0, 1.0, (0.0, 1.0), step=0.01)
    with col3:
        volume_range = st.slider("24h Volume (USDC)", 1.0, 1_000_000.0, (1.0, 1_000_000.0), step=100.0)

    active_only = st.checkbox("Only Active Markets (last hour)", value=True)
    

    # Dynamic SQL filter construction
    sql = f"""
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY timestamp DESC) AS rn
        FROM liquidity
    )
    WHERE rn = 1
      AND spread_platform BETWEEN {spread_platform_range[0]} AND {spread_platform_range[1]}
      AND bid_ask_spread BETWEEN {bid_ask_spread_range[0]} AND {bid_ask_spread_range[1]}
      AND volume24hr >= {volume_range[0]}
    """

    if volume_range[1] < 1_000_000:
        sql += f" AND volume24hr <= {volume_range[1]}"

    if active_only:
        cutoff = (datetime.utcnow() - timedelta(hours=1)).isoformat()
        sql += f" AND timestamp >= TIMESTAMP '{cutoff}'"
    

    filtered = query_df(sql)
    st.markdown(f"**{len(filtered):,} market outcomes match your filters.**")

    # Distribution plots
    st.markdown("#### 📊 Metric Distributions")
    plot_col1, plot_col2, plot_col3 = st.columns(3)

    with plot_col1:
        st.plotly_chart(px.histogram(filtered, x="spread_platform", nbins=30, title="Platform Spread"), use_container_width=True)
    with plot_col2:
        st.plotly_chart(px.histogram(filtered, x="bid_ask_spread", nbins=30, title="Bid-Ask Spread"), use_container_width=True)
    with plot_col3:
        st.plotly_chart(px.histogram(filtered, x="volume24hr", nbins=30, title="24h Volume (USDC)"), use_container_width=True)

    keyword = st.text_input("Search outcome or market (keyword match):", "")
    if keyword:
        keyword_lower = keyword.lower()
        sql += f" AND (LOWER(outcome_name) LIKE '%{keyword_lower}%' OR LOWER(question) LIKE '%{keyword_lower}%')"
    
    default_cols = [
        "market_id", "volume24hr", "outcome_name", "spread_platform", "bid_ask_spread",
        "bid_slope_5", "ask_slope_5", "order_book_imbalance", "rewards_min_size", "slug"
    ]
    selected_cols = st.multiselect("Columns to Display", filtered.columns.tolist(), default=default_cols)
    st.dataframe(filtered[selected_cols], use_container_width=True)

elif page == "Liquidity Analyzer":
    st.markdown("### 🔍 Liquidity Analyzer")

    # Load distinct markets from latest snapshot
    market_list = query_df("""
        SELECT DISTINCT market_id, question
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY timestamp DESC) AS rn
            FROM liquidity
        )
        WHERE rn = 1
        ORDER BY question
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_market = st.selectbox("Select Market", options=market_list["question"].tolist())
        selected_market_id = market_list[market_list["question"] == selected_market]["market_id"].values[0]
    with col2:
        time_window = st.radio("Time Range", ["24h", "3d", "7d", "30d", "All"], index=2, horizontal=True)

    # Load outcomes for selected market
    outcome_list = query_df(f"""
        SELECT DISTINCT outcome_name
        FROM liquidity
        WHERE market_id = '{selected_market_id}'
        ORDER BY outcome_name
    """)
    selected_outcome = st.selectbox("Select Outcome", options=outcome_list["outcome_name"].tolist())

    # Time filter
    now = datetime.utcnow()
    time_cutoff = {
        "24h": now - timedelta(hours=24),
        "3d": now - timedelta(days=3),
        "7d": now - timedelta(days=7),
        "30d": now - timedelta(days=30),
        "All": datetime(2000, 1, 1)
    }[time_window].isoformat()

    df_filtered = query_df(f"""
        SELECT * FROM liquidity
        WHERE market_id = '{selected_market_id}'
          AND outcome_name = '{selected_outcome}'
          AND timestamp >= TIMESTAMP '{time_cutoff}'
        ORDER BY timestamp
    """)

    if df_filtered.empty:
        st.warning("No data available for selected market and outcome.")
    else:
        latest_row = df_filtered.iloc[-1]
        active_flag = "🟢 Active" if (now - latest_row["timestamp"]).total_seconds() < 3 * 3600 else "🔴 Inactive"
        snapshot_time = latest_row["timestamp"].strftime("%Y-%m-%d %H:%M")
        reward_min = latest_row.get("rewards_min_size", "N/A")
        reward_max = latest_row.get("rewards_max_spread", "N/A")
        volume = f"{latest_row.get('volume24hr', 0.0):,.2f}"

        # Compact Header Summary
        st.markdown(f"""
        #### **{selected_outcome}** in *{selected_market}*
        | ⏱️ Last Snapshot | 💰 Reward Range | 📊 24h Volume | Status |
        |------------------|------------------|----------------|--------|
        | `{snapshot_time} UTC` | `{reward_min} – {reward_max}` | `{volume} USDC` | {active_flag} |
        """)

        # Panels
        with st.expander("📈 Price & Spread Metrics", expanded=True):
            price_metrics = [m for m in ["mid_price", "spread_platform", "bid_ask_spread"] if m in df_filtered.columns]
            if price_metrics:
                ts1 = df_filtered[["timestamp"] + price_metrics].melt(id_vars="timestamp")
                fig1 = px.line(ts1, x="timestamp", y="value", color="variable")
                fig1.update_layout(height=400, template="simple_white")
                st.plotly_chart(fig1, use_container_width=True)

        with st.expander("📊 Liquidity & Depth Metrics", expanded=False):
            liquidity_metrics = [m for m in ["top_of_book_depth_bid", "top_of_book_depth_ask", "order_book_imbalance", "rewards_min_size"] if m in df_filtered.columns]
            if liquidity_metrics:
                ts2 = df_filtered[["timestamp"] + liquidity_metrics].melt(id_vars="timestamp")
                fig2 = px.line(ts2, x="timestamp", y="value", color="variable")
                fig2.update_layout(height=400, template="simple_white")
                st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📉 Volatility / Impact Metrics", expanded=False):
            vol_metrics = [m for m in ["realized_volatility", "price_impact_buy", "price_impact_sell"] if m in df_filtered.columns]
            if vol_metrics:
                ts3 = df_filtered[["timestamp"] + vol_metrics].melt(id_vars="timestamp")
                fig3 = px.line(ts3, x="timestamp", y="value", color="variable")
                fig3.update_layout(height=400, template="simple_white")
                st.plotly_chart(fig3, use_container_width=True)

        with st.expander("🧾 Raw Data", expanded=False):
            st.dataframe(df_filtered.tail(100), use_container_width=True)
