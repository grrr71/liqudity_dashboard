import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(page_title="🧠 Liquidity Risk Monitor", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        .st-df th {
            background-color: #dee2e6;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# DATA LOADING
# -----------------------------
@st.cache_data
def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def compute_latest(df):
    latest = df.sort_values("timestamp").groupby("token_id").tail(1).copy()
    latest["severity"] = latest.apply(classify_liquidity_issue, axis=1)
    return latest

parquet_path = st.sidebar.text_input("📂 Path to Parquet File", value="/home/kaibrusch/code/liquidity_snapshots.parquet")

try:
    df = load_data(parquet_path)
except Exception as e:
    st.error(f"Failed to load Parquet: {e}")
    st.stop()

# -----------------------------
# CLASSIFY SEVERITY
# -----------------------------
def classify_liquidity_issue(row):
    issues = []
    if row["spread_bps"] > spread_thresh:
        issues.append("high_spread")
    if row["top_of_book_depth_bid"] < depth_thresh:
        issues.append("low_bid_depth")
    if row.get("price_impact_buy_100", 0) > slippage_thresh:
        issues.append("high_slippage_buy")
    if row.get("price_impact_sell_100", 0) > slippage_thresh:
        issues.append("high_slippage_sell")
    if row.get("volumeNum", 0) < volume_thresh:
        issues.append("low_volume")
    if abs(row.get("order_book_imbalance", 0)) > imbalance_thresh:
        issues.append("strong_imbalance")

    if not issues:
        return "✅ Healthy"
    elif len(issues) >= 3:
        return "🔴 Critical"
    else:
        return "🟠 At Risk"

# -----------------------------
# SIDEBAR THRESHOLDS
# -----------------------------
st.sidebar.title("🔧 Threshold Settings")
spread_thresh = st.sidebar.slider("Spread BPS Threshold", 0, 10000, 2000)
depth_thresh = st.sidebar.slider("Min Top-of-Book Depth (Bid)", 0, 100, 10)
slippage_thresh = st.sidebar.slider("Max Slippage (Buy/Sell)", 0.0, 1.0, 0.1, step=0.01)
volume_thresh = st.sidebar.slider("Min Volume", 0, 1000, 50)
imbalance_thresh = st.sidebar.slider("Max Order Book Imbalance", 0.0, 1.0, 0.8, step=0.01)

# -----------------------------
# COMPUTE LATEST SNAPSHOT
# -----------------------------
latest_df = compute_latest(df)
latest_ts = latest_df["timestamp"].max() if not latest_df.empty else datetime.utcnow()

# -----------------------------
# SEVERITY SUMMARY
# -----------------------------
st.sidebar.markdown("### 📊 Liquidity Risk Dashboard")
severity_counts = latest_df["severity"].value_counts()
st.sidebar.markdown(
    f"🔴 Critical: {severity_counts.get('🔴 Critical', 0)}  \
     🟠 At Risk: {severity_counts.get('🟠 At Risk', 0)}  \
     ✅ Healthy: {severity_counts.get('✅ Healthy', 0)}"
)

# -----------------------------
# SELECT MARKET AND METRICS
# -----------------------------
st.markdown("### 🔎 Select Market to Visualize")
selected_market = st.selectbox("Market ID", options=latest_df["market_id"].unique())

df_market = df[df["market_id"] == selected_market].sort_values("timestamp")
metric_options = [col for col in df.columns if df[col].dtype.kind in 'fc' and col not in ["market_id", "token_id"]]
metrics_to_plot = st.multiselect("📈 Select metrics to show", metric_options, default=["bid_ask_spread", "order_book_imbalance"])

# -----------------------------
# PLOT TIME SERIES
# -----------------------------
if not df_market.empty and metrics_to_plot:
    fig = px.line(
        df_market.melt(id_vars="timestamp", value_vars=metrics_to_plot),
        x="timestamp", y="value", color="variable",
        title=f"📈 Time Series for Market {selected_market}",
        labels={"variable": "Metric", "value": "Value"}
    )
    fig.update_traces(line_shape="linear")
    fig.update_yaxes(tickformat=".2f")
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TOP RISKY MARKETS
# -----------------------------
st.markdown("### 🔥 Top 10 Markets by Spread BPS")
top_spread = latest_df.sort_values("spread_bps", ascending=False).head(10)
st.dataframe(top_spread[["market_id", "spread_bps", "outcome_name", "question"]], use_container_width=True)

# -----------------------------
# TABLE VIEW
# -----------------------------
st.subheader("🚨 Flagged Market Outcomes")

highlight_color = {
    "🔴 Critical": "#f8d7da",
    "🟠 At Risk": "#fff3cd",
    "✅ Healthy": "#d4edda"
}

def highlight_severity(row):
    color = highlight_color.get(row["severity"], "white")
    return [f"background-color: {color}" if col == "severity" else "" for col in row.index]

column_order = [
    "severity", "timestamp", "spread_bps", "bid_ask_spread", "top_of_book_depth_bid", "top_of_book_depth_ask",
    "order_book_imbalance", "price_impact_buy_100", "price_impact_sell_100",
    "volumeNum", "liquidityNum", "market_id", "token_id", "outcome_name", "question"
]

rounded_df = latest_df[column_order].copy()
metric_cols = [col for col in rounded_df.columns if rounded_df[col].dtype.kind in 'fc']
rounded_df[metric_cols] = rounded_df[metric_cols].round(2)

st.dataframe(
    rounded_df.sort_values("severity").style.format({col: "{:.2f}" for col in metric_cols}).apply(highlight_severity, axis=1),
    use_container_width=True, hide_index=True
)
