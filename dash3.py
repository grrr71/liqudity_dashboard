import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import duckdb
from datetime import datetime, timedelta
import numpy as np
import requests

st.set_page_config(page_title="Polymarket Liquidity Toolkit", layout="wide")

# ---- Helper Functions ---- #
def calculate_liquidity_risk_score(row):
    """Calculate a liquidity risk score based on various metrics"""
    risk_factors = {
        'spread_platform': 0.3,
        'bid_ask_spread': 0.2,
        'order_book_imbalance': 0.2,
        'top_of_book_depth_bid': 0.15,
        'top_of_book_depth_ask': 0.15
    }
    
    score = 0
    for factor, weight in risk_factors.items():
        if factor in row:
            if factor in ['top_of_book_depth_bid', 'top_of_book_depth_ask']:
                # Higher depth is better (lower risk)
                score += weight * (1 - min(1, row[factor] / 100000))
            else:
                # Higher spread/imbalance is worse (higher risk)
                score += weight * min(1, row[factor])
    
    return score

def calculate_efficiency_ratio(data):
    """Calculate the efficiency ratio (how quickly prices adjust to new information)"""
    if len(data) < 2:
        return 0
    
    price_changes = data['mid_price'].diff().abs().sum()
    total_movement = (data['mid_price'] - data['mid_price'].iloc[0]).abs().sum()
    
    return price_changes / total_movement if total_movement > 0 else 0

def calculate_slippage_risk(row, trade_size=1000):
    """Calculate potential slippage for a given trade size based on available aggregate metrics"""
    if 'top_of_book_depth_bid' not in row or 'top_of_book_depth_ask' not in row or 'bid_ask_spread' not in row:
        return None
    
    # Calculate average depth from top of book
    avg_depth = (row['top_of_book_depth_bid'] + row['top_of_book_depth_ask']) / 2
    
    # Calculate potential slippage based on:
    # 1. Current spread (bid_ask_spread)
    # 2. Available depth at top of book
    # 3. Order book imbalance
    spread_impact = row['bid_ask_spread']  # Base impact from current spread
    
    # Depth impact - how much of the trade size relative to available depth
    depth_impact = min(1, trade_size / avg_depth) if avg_depth > 0 else 1
    
    # Combine impacts with weights
    slippage = (0.6 * spread_impact) + (0.4 * depth_impact)
    
    return min(1, slippage)  # Cap at 100%

@st.cache_data(ttl=300)
def create_time_series_plot(data, metrics, title):
    """Create a time series plot for the given metrics"""
    if not metrics:
        return None
    ts_data = data[["timestamp"] + metrics].melt(id_vars="timestamp")
    fig = px.line(ts_data, x="timestamp", y="value", color="variable")
    fig.update_layout(height=400, template="simple_white", title=title)
    return fig

@st.cache_data(ttl=60)
def fetch_trade_volume_1h(token_id):
    """Fetch 1-hour trade volume from Goldsky"""
    endpoint = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/prod/gn"
    now = int(datetime.utcnow().timestamp())
    one_hour_ago = now - 3600

    # Format token_id as a string without scientific notation
    token_id_str = str(int(token_id))

    # Query for both maker and taker sides
    query = f"""
    {{
      makerOrders: ordersMatchedEvents(
        where: {{
          makerAssetID: "{token_id_str}",
          timestamp_gt: {one_hour_ago}
        }},
        first: 1000,
        orderBy: timestamp,
        orderDirection: desc
      ) {{
        makerAmountFilled
        takerAmountFilled
        timestamp
        makerAssetID
        takerAssetID
      }}
      takerOrders: ordersMatchedEvents(
        where: {{
          takerAssetID: "{token_id_str}",
          timestamp_gt: {one_hour_ago}
        }},
        first: 1000,
        orderBy: timestamp,
        orderDirection: desc
      ) {{
        makerAmountFilled
        takerAmountFilled
        timestamp
        makerAssetID
        takerAssetID
      }}
    }}
    """

    try:
        response = requests.post(endpoint, json={"query": query})
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            trades = []
            # Process maker orders (our token is the maker)
            if "makerOrders" in data["data"]:
                for trade in data["data"]["makerOrders"]:
                    # When we're the maker, we give up our token and receive USDC
                    # USDC has 6 decimal places
                    usdc_amount = float(trade["takerAmountFilled"]) / 1_000_000
                    token_amount = float(trade["makerAmountFilled"])
                    trades.append({
                        "usdc_volume": usdc_amount,  # USDC we receive
                        "token_volume": token_amount,  # Token we give
                        "timestamp": trade["timestamp"],
                        "role": "maker",
                        "side": "sell",  # We're selling our token
                        "implied_price": usdc_amount / token_amount if token_amount > 0 else 0  # USDC per token
                    })
            # Process taker orders (our token is the taker)
            if "takerOrders" in data["data"]:
                for trade in data["data"]["takerOrders"]:
                    # When we're the taker, we give up USDC and receive our token
                    # USDC has 6 decimal places
                    usdc_amount = float(trade["takerAmountFilled"]) / 1_000_000
                    token_amount = float(trade["makerAmountFilled"])
                    trades.append({
                        "usdc_volume": usdc_amount,  # USDC we give
                        "token_volume": token_amount,  # Token we receive
                        "timestamp": trade["timestamp"],
                        "role": "taker",
                        "side": "buy",  # We're buying our token
                        "implied_price": usdc_amount / token_amount if token_amount > 0 else 0  # USDC per token
                    })
            return trades
        else:
            st.error(f"Unexpected response format: {data}")
            return []
    except Exception as e:
        st.error(f"Error fetching trade data: {str(e)}")
        return []

@st.cache_data(ttl=60)
def fetch_historical_trades(token_id, hours=24):
    """Fetch historical trades for analysis"""
    endpoint = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/prod/gn"
    now = int(datetime.utcnow().timestamp())
    start_time = now - (hours * 3600)

    # Format token_id as a string without scientific notation
    token_id_str = str(int(token_id))

    # Query for both maker and taker sides
    query = f"""
    {{
      makerOrders: ordersMatchedEvents(
        where: {{
          makerAssetID: "{token_id_str}",
          timestamp_gt: {start_time}
        }},
        first: 1000,
        orderBy: timestamp,
        orderDirection: desc
      ) {{
        makerAmountFilled
        takerAmountFilled
        timestamp
        makerAssetID
        takerAssetID
      }}
      takerOrders: ordersMatchedEvents(
        where: {{
          takerAssetID: "{token_id_str}",
          timestamp_gt: {start_time}
        }},
        first: 1000,
        orderBy: timestamp,
        orderDirection: desc
      ) {{
        makerAmountFilled
        takerAmountFilled
        timestamp
        makerAssetID
        takerAssetID
      }}
    }}
    """

    try:
        response = requests.post(endpoint, json={"query": query})
        response.raise_for_status()
        data = response.json()
        
        if "data" in data:
            trades = []
            # Process maker orders (our token is the maker)
            if "makerOrders" in data["data"]:
                for trade in data["data"]["makerOrders"]:
                    # When we're the maker, we give up our token and receive USDC
                    # USDC has 6 decimal places
                    usdc_amount = float(trade["takerAmountFilled"]) / 1_000_000
                    token_amount = float(trade["makerAmountFilled"])
                    trades.append({
                        "usdc_volume": usdc_amount,  # USDC we receive
                        "token_volume": token_amount,  # Token we give
                        "timestamp": trade["timestamp"],
                        "role": "maker",
                        "side": "sell",  # We're selling our token
                        "implied_price": usdc_amount / token_amount if token_amount > 0 else 0  # USDC per token
                    })
            # Process taker orders (our token is the taker)
            if "takerOrders" in data["data"]:
                for trade in data["data"]["takerOrders"]:
                    # When we're the taker, we give up USDC and receive our token
                    # USDC has 6 decimal places
                    usdc_amount = float(trade["takerAmountFilled"]) / 1_000_000
                    token_amount = float(trade["makerAmountFilled"])
                    trades.append({
                        "usdc_volume": usdc_amount,  # USDC we give
                        "token_volume": token_amount,  # Token we receive
                        "timestamp": trade["timestamp"],
                        "role": "taker",
                        "side": "buy",  # We're buying our token
                        "implied_price": usdc_amount / token_amount if token_amount > 0 else 0  # USDC per token
                    })
            return trades
        else:
            st.error(f"Unexpected response format: {data}")
            return []
    except Exception as e:
        st.error(f"Error fetching historical trade data: {str(e)}")
        return []

def format_volume(volume):
    """Format volume numbers in a human-readable way"""
    if volume >= 1_000_000:
        return f"${volume/1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"${volume/1_000:.1f}K"
    else:
        return f"${volume:.2f}"

def calculate_trade_metrics(trades):
    """Calculate various trade metrics"""
    if not trades:
        return {
            'total_volume': 0,
            'usdc_volume': 0,
            'token_volume': 0,
            'trade_count': 0,
            'avg_trade_size': 0,
            'largest_trade': 0,
            'volume_by_hour': pd.DataFrame(columns=['hour', 'usdc_volume']),
            'buy_volume': 0,
            'sell_volume': 0,
            'maker_volume': 0,
            'taker_volume': 0,
            'avg_price': 0
        }
    
    # Convert trades to DataFrame
    df = pd.DataFrame(trades)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Calculate metrics
    usdc_volume = df['usdc_volume'].sum()
    token_volume = df['token_volume'].sum()
    total_volume = usdc_volume + token_volume
    trade_count = len(df)
    avg_trade_size = usdc_volume / trade_count if trade_count > 0 else 0
    largest_trade = df['usdc_volume'].max()
    
    # Calculate buy/sell volumes
    buy_volume = df[df['side'] == 'buy']['usdc_volume'].sum()
    sell_volume = df[df['side'] == 'sell']['usdc_volume'].sum()
    
    # Calculate maker/taker volumes
    maker_volume = df[df['role'] == 'maker']['usdc_volume'].sum()
    taker_volume = df[df['role'] == 'taker']['usdc_volume'].sum()
    
    # Calculate average price
    avg_price = df['implied_price'].mean()
    
    # Calculate hourly volume
    df['hour'] = df['timestamp'].dt.floor('H')
    volume_by_hour = df.groupby('hour')['usdc_volume'].sum().reset_index()
    
    return {
        'total_volume': total_volume,
        'usdc_volume': usdc_volume,
        'token_volume': token_volume,
        'trade_count': trade_count,
        'avg_trade_size': avg_trade_size,
        'largest_trade': largest_trade,
        'volume_by_hour': volume_by_hour,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'maker_volume': maker_volume,
        'taker_volume': taker_volume,
        'avg_price': avg_price
    }

# ---- DuckDB Setup ---- #
parquet_path = "liquidity_snapshots3.parquet"
con = duckdb.connect()
con.execute(f"CREATE OR REPLACE VIEW liquidity AS SELECT * FROM read_parquet('{parquet_path}')")

def query_df(sql: str) -> pd.DataFrame:
    return con.execute(sql).df()

# Cache the latest snapshot
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_latest_snapshot():
    return query_df("""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY timestamp DESC) AS rn
            FROM liquidity
        )
        WHERE rn = 1
    """)

# Cache market list
@st.cache_data(ttl=300)
def get_market_list():
    return query_df("""
        SELECT DISTINCT market_id, question
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY token_id ORDER BY timestamp DESC) AS rn
            FROM liquidity
        )
        WHERE rn = 1
        ORDER BY question
    """)

# Cache outcome list for selected market
@st.cache_data(ttl=300)
def get_outcome_list(market_id):
    return query_df(f"""
        SELECT DISTINCT outcome_name, token_id
        FROM liquidity
        WHERE market_id = '{market_id}'
        ORDER BY outcome_name
    """)

# Cache filtered data
@st.cache_data(ttl=300)
def get_filtered_data(market_id, outcome, cutoff):
    return query_df(f"""
        SELECT * FROM liquidity
        WHERE market_id = '{market_id}'
          AND outcome_name = '{outcome}'
          AND timestamp >= TIMESTAMP '{cutoff}'
        ORDER BY timestamp
    """)

# ---- Sidebar ---- #
now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M") + " UTC"
latest_df = get_latest_snapshot()
latest_ts = latest_df['timestamp'].max().strftime("%Y-%m-%d %H:%M") + " UTC"

col1, col2 = st.sidebar.columns([3, 1])
with col1:
    st.markdown(f"🕒 **Current UTC Time:**<br>`{now_utc}`", unsafe_allow_html=True)
    st.markdown(f"📸 **Latest Snapshot:**<br>`{latest_ts}`", unsafe_allow_html=True)
with col2:
    st.markdown("###")  # Spacer for alignment
    if st.button("🔄", help="Reload latest snapshot"):
        get_latest_snapshot.clear()
        st.rerun()

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
    
    # Apply filters in-memory
    filtered = latest_df.copy()
    filtered = filtered[
        (filtered['spread_platform'].between(spread_platform_range[0], spread_platform_range[1])) &
        (filtered['bid_ask_spread'].between(bid_ask_spread_range[0], bid_ask_spread_range[1])) &
        (filtered['volume24hr'] >= volume_range[0])
    ]
    
    if volume_range[1] < 1_000_000:
        filtered = filtered[filtered['volume24hr'] <= volume_range[1]]

    if active_only:
        cutoff = datetime.utcnow() - timedelta(hours=1)
        filtered = filtered[filtered['timestamp'] >= cutoff]

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
        filtered = filtered[
            filtered['outcome_name'].str.lower().str.contains(keyword_lower) |
            filtered['question'].str.lower().str.contains(keyword_lower)
        ]
    
    default_cols = [
        "market_id", "volume24hr", "outcome_name", "spread_platform", "bid_ask_spread",
        "bid_slope_5", "ask_slope_5", "order_book_imbalance", "rewards_min_size", "slug"
    ]
    selected_cols = st.multiselect("Columns to Display", filtered.columns.tolist(), default=default_cols)
    
    st.dataframe(
        filtered[selected_cols].reset_index(drop=True),
        use_container_width=True,
        height=400
    )

elif page == "Liquidity Analyzer":
    st.markdown("### 🔍 Liquidity Analyzer")
    
    market_list = get_market_list()

    # Add market ID lookup
    lookup_method = st.radio("Lookup Method", ["Select Market", "Enter Market ID"], horizontal=True)
    
    if lookup_method == "Select Market":
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_market = st.selectbox("Select Market", options=market_list["question"].tolist())
            selected_market_id = market_list[market_list["question"] == selected_market]["market_id"].values[0]
        with col2:
            time_window = st.radio("Time Range", ["24h", "3d", "7d", "30d", "All"], index=0, horizontal=True)
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            market_id_input = st.text_input("Enter Market ID", placeholder="e.g., 0x1234...")
            if market_id_input:
                # Validate market ID exists
                market_exists = market_list[market_list["market_id"] == market_id_input]
                if not market_exists.empty:
                    selected_market_id = market_id_input
                    selected_market = market_exists["question"].values[0]
                    st.success(f"Found market: {selected_market}")
                else:
                    st.error("Market ID not found. Please check the ID and try again.")
                    selected_market_id = None
                    selected_market = None
            else:
                selected_market_id = None
                selected_market = None
        with col2:
            time_window = st.radio("Time Range", ["24h", "3d", "7d", "30d", "All"], index=0, horizontal=True)

    if selected_market_id:
        outcome_list = get_outcome_list(selected_market_id)
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

        df_filtered = get_filtered_data(selected_market_id, selected_outcome, time_cutoff)

        if df_filtered.empty:
            st.warning("No data available for selected market and outcome.")
        else:
            latest_row = df_filtered.iloc[-1]
            active_flag = "🟢 Active" if (now - latest_row["timestamp"]).total_seconds() < 3 * 3600 else "🔴 Inactive"
            snapshot_time = latest_row["timestamp"].strftime("%Y-%m-%d %H:%M")
            reward_min = latest_row.get("rewards_min_size", "N/A")
            reward_max = latest_row.get("rewards_max_spread", "N/A")
            volume = f"{latest_row.get('volume24hr', 0.0):,.2f}"

            # Calculate liquidity risk score
            liquidity_risk = calculate_liquidity_risk_score(latest_row)

            # Enhanced Header Summary
            st.markdown(f"""
            #### **{selected_outcome}** in *{selected_market}*
            | ⏱️ Last Snapshot | 💰 Reward Range | 📊 24h Volume | Status |
            |------------------|------------------|----------------|--------|
            | `{snapshot_time} UTC` | `{reward_min} – {reward_max}` | `{volume} USDC` | {active_flag} |
            """)

            # Risk Metrics Panel
            with st.expander("⚠️ Risk Analysis", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Liquidity Risk Score",
                        f"{liquidity_risk:.2%}",
                        help="Composite score (0-100%) measuring market liquidity risk. Higher scores indicate higher risk."
                    )
                with col2:
                    if 'order_book_imbalance' in latest_row:
                        st.metric(
                            "Order Book Imbalance",
                            f"{latest_row['order_book_imbalance']:.2%}",
                            help="Measures the imbalance between bid and ask volumes. Positive values indicate more bids, negative values indicate more asks."
                        )
                with col3:
                    if 'price_skew' in latest_row:
                        st.metric(
                            "Price Skew",
                            f"{latest_row['price_skew']:.4f}",
                            help="Measures asymmetry in the order book. Positive values indicate higher ask prices, negative values indicate higher bid prices."
                        )

            # Depth Analysis Panel
            with st.expander("📊 Depth Analysis", expanded=True):
                if all(m in latest_row for m in ['depth_bid_1pct', 'depth_ask_1pct', 'depth_bid_2pct', 'depth_ask_2pct']):
                    depth_data = {
                        'Level': ['1%', '2%'],
                        'Bid Depth': [latest_row['depth_bid_1pct'], latest_row['depth_bid_2pct']],
                        'Ask Depth': [latest_row['depth_ask_1pct'], latest_row['depth_ask_2pct']]
                    }
                    depth_df = pd.DataFrame(depth_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=depth_df['Level'],
                        y=depth_df['Bid Depth'],
                        name='Bid Depth',
                        marker_color='green'
                    ))
                    fig.add_trace(go.Bar(
                        x=depth_df['Level'],
                        y=depth_df['Ask Depth'],
                        name='Ask Depth',
                        marker_color='red'
                    ))
                    
                    fig.update_layout(
                        title='Order Book Depth Profile',
                        barmode='group',
                        height=400,
                        template='simple_white',
                        xaxis_title='Price Level',
                        yaxis_title='Depth (USDC)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Depth metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        total_depth = sum(depth_df['Bid Depth']) + sum(depth_df['Ask Depth'])
                        st.metric(
                            "Total Depth (2%)",
                            f"{total_depth:,.0f} USDC",
                            help="Total liquidity available within 2% of mid price"
                        )
                    with col2:
                        if 'liquidity_density_2pct' in latest_row:
                            st.metric(
                                "Liquidity Density",
                                f"{latest_row['liquidity_density_2pct']:,.0f}",
                                help="Amount of liquidity per basis point of spread within 2% of mid price"
                            )

            # Price & Spread Metrics
            with st.expander("📈 Price & Spread Metrics", expanded=True):
                price_metrics = [m for m in ["mid_price", "spread_platform", "bid_ask_spread"] if m in df_filtered.columns]
                fig1 = create_time_series_plot(df_filtered, price_metrics, "Price & Spread Metrics")
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)

            # Liquidity & Depth Metrics
            with st.expander("📊 Liquidity & Depth Metrics", expanded=True):
                liquidity_metrics = [m for m in ["top_of_book_depth_bid", "top_of_book_depth_ask", "order_book_imbalance"] if m in df_filtered.columns]
                fig2 = create_time_series_plot(df_filtered, liquidity_metrics, "Liquidity & Depth Metrics")
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)

            # Price Impact Analysis
            with st.expander("💹 Price Impact Analysis", expanded=True):
                if all(m in latest_row for m in ['price_impact_buy_100', 'price_impact_sell_100']):
                    impact_data = {
                        'Side': ['Buy', 'Sell'],
                        'Impact': [latest_row['price_impact_buy_100'], latest_row['price_impact_sell_100']]
                    }
                    impact_df = pd.DataFrame(impact_data)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=impact_df['Side'],
                        y=impact_df['Impact'],
                        marker_color=['red', 'green']
                    ))
                    
                    fig.update_layout(
                        title='Price Impact for 100 USDC Trade',
                        height=400,
                        template='simple_white',
                        yaxis_title='Price Impact'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Impact metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        impact_ratio = latest_row['price_impact_buy_100'] / latest_row['price_impact_sell_100'] if latest_row['price_impact_sell_100'] != 0 else float('inf')
                        st.metric(
                            "Impact Ratio (Buy/Sell)",
                            f"{impact_ratio:.2f}",
                            help="Ratio of buy impact to sell impact. Values > 1 indicate higher buy impact, < 1 indicate higher sell impact."
                        )
                    with col2:
                        if 'levels_used_100' in latest_row:
                            st.metric(
                                "Levels Used (100 USDC)",
                                f"{latest_row['levels_used_100']}",
                                help="Number of price levels needed to execute a 100 USDC trade"
                            )

            # Trade Analysis Panel
            with st.expander("💹 Trade Analysis", expanded=True):
                st.markdown("#### Recent Trade Activity")
                
                # Get token_id for the selected outcome
                outcome_data = get_outcome_list(selected_market_id)
                selected_outcome_data = outcome_data[outcome_data['outcome_name'] == selected_outcome]
                if selected_outcome_data.empty:
                    st.error("Could not find token ID for selected outcome")
                    st.stop()
                
                token_id = selected_outcome_data['token_id'].iloc[0]
                
                # Fetch trade data
                trades_1h = fetch_trade_volume_1h(token_id)
                trades_24h = fetch_historical_trades(token_id, hours=24)
                
                # Calculate metrics
                metrics_1h = calculate_trade_metrics(trades_1h)
                metrics_24h = calculate_trade_metrics(trades_24h)
                
                # Display metrics with better formatting
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "1h USDC Volume",
                        format_volume(metrics_1h['usdc_volume']),
                        help="Total USDC trading volume in the last hour"
                    )
                with col2:
                    st.metric(
                        "1h Trade Count",
                        f"{metrics_1h['trade_count']:,}",
                        help="Number of trades in the last hour"
                    )
                with col3:
                    st.metric(
                        "Avg Trade Size",
                        format_volume(metrics_1h['avg_trade_size']),
                        help="Average size of trades in the last hour"
                    )
                with col4:
                    st.metric(
                        "Largest Trade",
                        format_volume(metrics_1h['largest_trade']),
                        help="Largest single trade in the last hour"
                    )
                
                # Add buy/sell and maker/taker metrics
                st.markdown("#### Trade Flow Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Buy Volume",
                        format_volume(metrics_1h['buy_volume']),
                        help="Volume of buy trades in the last hour"
                    )
                with col2:
                    st.metric(
                        "Sell Volume",
                        format_volume(metrics_1h['sell_volume']),
                        help="Volume of sell trades in the last hour"
                    )
                with col3:
                    st.metric(
                        "Maker Volume",
                        format_volume(metrics_1h['maker_volume']),
                        help="Volume where our token was the maker"
                    )
                with col4:
                    st.metric(
                        "Taker Volume",
                        format_volume(metrics_1h['taker_volume']),
                        help="Volume where our token was the taker"
                    )

                # 24-Hour Trade Analysis
                st.markdown("#### 24-Hour Trade Analysis")
                
                if trades_24h:
                    # Convert trades to DataFrame for analysis
                    df_24h = pd.DataFrame(trades_24h)
                    df_24h['timestamp'] = pd.to_datetime(df_24h['timestamp'], unit='s')
                    df_24h['hour'] = df_24h['timestamp'].dt.floor('H')
                    
                    # 1. Volume and Price Over Time
                    st.markdown("##### Volume and Price Trends")
                    hourly_data = df_24h.groupby('hour').agg({
                        'usdc_volume': 'sum',
                        'implied_price': 'mean',
                        'token_volume': 'sum'
                    }).reset_index()
                    
                    fig = go.Figure()
                    # Add volume bars
                    fig.add_trace(go.Bar(
                        x=hourly_data['hour'],
                        y=hourly_data['usdc_volume'],
                        name='Volume (USDC)',
                        yaxis='y1'
                    ))
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=hourly_data['hour'],
                        y=hourly_data['implied_price'],
                        name='Price (USDC)',
                        yaxis='y2',
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title='Hourly Volume and Price',
                        xaxis_title='Hour',
                        yaxis_title='Volume (USDC)',
                        yaxis2=dict(
                            title='Price (USDC)',
                            overlaying='y',
                            side='right'
                        ),
                        template='simple_white',
                        height=400,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Trade Size Distribution
                    st.markdown("##### Trade Size Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Trade size histogram
                        fig = px.histogram(
                            df_24h,
                            x='usdc_volume',
                            nbins=30,
                            title='Distribution of Trade Sizes',
                            labels={'usdc_volume': 'Trade Size (USDC)', 'count': 'Number of Trades'}
                        )
                        fig.update_layout(
                            xaxis=dict(tickformat='$,.0f'),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cumulative volume by trade size
                        df_sorted = df_24h.sort_values('usdc_volume')
                        df_sorted['cumulative_volume'] = df_sorted['usdc_volume'].cumsum()
                        df_sorted['cumulative_pct'] = df_sorted['cumulative_volume'] / df_sorted['usdc_volume'].sum() * 100
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df_sorted['usdc_volume'],
                            y=df_sorted['cumulative_pct'],
                            mode='lines',
                            name='Cumulative Volume'
                        ))
                        fig.update_layout(
                            title='Cumulative Volume by Trade Size',
                            xaxis_title='Trade Size (USDC)',
                            yaxis_title='Cumulative Volume (%)',
                            xaxis=dict(tickformat='$,.0f'),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. Trade Flow Analysis
                    st.markdown("##### Trade Flow Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Buy vs Sell Volume Over Time
                        hourly_side = df_24h.groupby(['hour', 'side'])['usdc_volume'].sum().reset_index()
                        fig = px.bar(
                            hourly_side,
                            x='hour',
                            y='usdc_volume',
                            color='side',
                            title='Buy vs Sell Volume Over Time',
                            labels={'usdc_volume': 'Volume (USDC)', 'side': 'Trade Side'}
                        )
                        fig.update_layout(
                            yaxis=dict(tickformat='$,.0f'),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Maker vs Taker Volume Over Time
                        hourly_role = df_24h.groupby(['hour', 'role'])['usdc_volume'].sum().reset_index()
                        fig = px.bar(
                            hourly_role,
                            x='hour',
                            y='usdc_volume',
                            color='role',
                            title='Maker vs Taker Volume Over Time',
                            labels={'usdc_volume': 'Volume (USDC)', 'role': 'Trade Role'}
                        )
                        fig.update_layout(
                            yaxis=dict(tickformat='$,.0f'),
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 4. Summary Statistics
                    st.markdown("##### 24-Hour Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Volume",
                            format_volume(metrics_24h['usdc_volume']),
                            help="Total trading volume in the last 24 hours"
                        )
                    with col2:
                        st.metric(
                            "Total Trades",
                            f"{metrics_24h['trade_count']:,}",
                            help="Total number of trades in the last 24 hours"
                        )
                    with col3:
                        st.metric(
                            "Avg Price",
                            f"${metrics_24h['avg_price']:.4f}",
                            help="Average price in the last 24 hours"
                        )
                    with col4:
                        st.metric(
                            "Price Range",
                            f"${df_24h['implied_price'].min():.4f} - ${df_24h['implied_price'].max():.4f}",
                            help="Price range in the last 24 hours"
                        )
                else:
                    st.warning("No trade data available for the last 24 hours")

            # Raw Data
            with st.expander("🧾 Raw Data", expanded=False):
                st.dataframe(df_filtered.tail(100), use_container_width=True, height=400)

            