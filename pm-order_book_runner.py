import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from py_clob_client.client import ClobClient, BookParams
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def compute_order_book_metrics(order_book, token_id, market_meta):
    bids = sorted(order_book.bids, key=lambda x: -float(x.price))
    asks = sorted(order_book.asks, key=lambda x: float(x.price))

    if not bids or not asks:
        return None

    best_bid = float(bids[0].price)
    best_ask = float(asks[0].price)
    top_bid_size = float(bids[0].size)
    top_ask_size = float(asks[0].size)
    mid_price = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else None
    normalized_spread = spread / mid_price if mid_price > 0 else None

    def cum_depth(book, side, pct):
        bound = mid_price * (1 + pct) if side == 'ask' else mid_price * (1 - pct)
        return sum(
            float(level.size) for level in book
            if (side == 'ask' and float(level.price) <= bound)
            or (side == 'bid' and float(level.price) >= bound)
        )

    def simulate_slippage(book, size=100):
        remaining = size
        cost = 0
        levels_used = 0
        for level in book:
            price = float(level.price)
            size = float(level.size)
            take = min(remaining, size)
            cost += take * price
            remaining -= take
            levels_used += 1
            if remaining <= 0:
                break
        if remaining > 0 or mid_price == 0:
            return None, None
        return (cost / size) - mid_price, levels_used

    def slope(book, levels=5):
        if len(book) < levels:
            return None
        prices = [float(level.price) for level in book[:levels]]
        return abs(prices[0] - prices[-1]) / levels

    def calculate_liquidity_flow(book, side):
        """Calculate how liquidity is distributed across price levels"""
        if not book:
            return None, None, None
        
        prices = [float(level.price) for level in book]
        sizes = [float(level.size) for level in book]
        
        # Calculate weighted average price
        wap = sum(p * s for p, s in zip(prices, sizes)) / sum(sizes) if sum(sizes) > 0 else None
        
        # Calculate price levels with significant liquidity (top 80%)
        total_size = sum(sizes)
        cum_size = 0
        significant_levels = 0
        for size in sorted(sizes, reverse=True):
            cum_size += size
            significant_levels += 1
            if cum_size >= 0.8 * total_size:
                break
        
        # Calculate liquidity concentration
        concentration = sum(sizes[:3]) / total_size if total_size > 0 else None
        
        return wap, significant_levels, concentration

    bid_volume = sum(float(b.size) for b in bids)
    ask_volume = sum(float(a.size) for a in asks)
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-6)
    
    # Calculate depth at different levels
    depth_within_1pct = cum_depth(bids, 'bid', 0.01) + cum_depth(asks, 'ask', 0.01)
    depth_within_2pct = cum_depth(bids, 'bid', 0.02) + cum_depth(asks, 'ask', 0.02)
    depth_within_5pct = cum_depth(bids, 'bid', 0.05) + cum_depth(asks, 'ask', 0.05)
    
    # Calculate liquidity flow metrics
    bid_wap, bid_levels, bid_concentration = calculate_liquidity_flow(bids, 'bid')
    ask_wap, ask_levels, ask_concentration = calculate_liquidity_flow(asks, 'ask')
    
    # Calculate price impact at different sizes
    impact_100, levels_100 = simulate_slippage(asks, 100) or (None, None)
    impact_1000, levels_1000 = simulate_slippage(asks, 1000) or (None, None)
    impact_10000, levels_10000 = simulate_slippage(asks, 10000) or (None, None)
    
    # Calculate liquidity density at different levels
    liquidity_density_1pct = depth_within_1pct / spread if spread > 0 else None
    liquidity_density_2pct = depth_within_2pct / spread if spread > 0 else None
    liquidity_density_5pct = depth_within_5pct / spread if spread > 0 else None

    return {
        "timestamp": datetime.utcnow(),
        "token_id": token_id,
        **market_meta,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "bid_ask_spread": spread,
        "spread_bps": spread_bps,
        "normalized_spread": normalized_spread,
        "top_of_book_depth_bid": top_bid_size,
        "top_of_book_depth_ask": top_ask_size,
        "depth_bid_1pct": cum_depth(bids, 'bid', 0.01),
        "depth_ask_1pct": cum_depth(asks, 'ask', 0.01),
        "depth_bid_2pct": cum_depth(bids, 'bid', 0.02),
        "depth_ask_2pct": cum_depth(asks, 'ask', 0.02),
        "depth_bid_5pct": cum_depth(bids, 'bid', 0.05),
        "depth_ask_5pct": cum_depth(asks, 'ask', 0.05),
        "order_book_imbalance": imbalance,
        "price_impact_buy_100": impact_100,
        "price_impact_buy_1000": impact_1000,
        "price_impact_buy_10000": impact_10000,
        "levels_used_100": levels_100,
        "levels_used_1000": levels_1000,
        "levels_used_10000": levels_10000,
        "level_count_bid": len(bids),
        "level_count_ask": len(asks),
        "price_skew": (best_bid + best_ask - 2 * mid_price),
        "bid_slope_5": slope(bids),
        "ask_slope_5": slope(asks),
        "liquidity_density_1pct": liquidity_density_1pct,
        "liquidity_density_2pct": liquidity_density_2pct,
        "liquidity_density_5pct": liquidity_density_5pct,
        "bid_wap": bid_wap,
        "ask_wap": ask_wap,
        "bid_liquidity_levels": bid_levels,
        "ask_liquidity_levels": ask_levels,
        "bid_concentration": bid_concentration,
        "ask_concentration": ask_concentration,
        "effective_spread": None,
        "realized_spread": None,
        "adverse_selection_cost": None,
        "liquidity_resiliency": None,
    }

def get_all_clob_enabled_token_ids():
    base_url = "https://gamma-api.polymarket.com/markets"
    now_iso = datetime.utcnow().isoformat() + "Z"
    offset = 0
    limit = 500
    all_assets = []

    while True:
        params = {
            "enableOrderBook": "true",
            "archived": "false",
            "closed": "false",
            "active": "true",
            "liquidity_num_min": "1.0",
            "end_date_min": now_iso,
            "limit": limit,
            "offset": offset
        }

        resp = requests.get(base_url, params=params)
        markets = resp.json()
        if not markets:
            break

        for m in markets:
            try:
                token_ids = json.loads(m.get("clobTokenIds", "[]"))
                outcomes = json.loads(m.get("outcomes", "[]"))
                for i, token_id in enumerate(token_ids):
                    all_assets.append({
                        "token_id": token_id,
                        "market_meta": {
                            "market_id": m["id"],
                            "question": m.get("question", ""),
                            "slug": m.get("slug", ""),
                            "category": m.get("category", ""),
                            "end_date": m.get("endDate", ""),
                            "market_type": m.get("marketType", ""),
                            "closed": m.get("closed", False),
                            "archived": m.get("archived", False),
                            "active": m.get("active", True),
                            "volumeNum": m.get("volumeNum", None),
                            "volume24hr": m.get("volume24hr", None),
                            "liquidityNum": m.get("liquidityNum", None),
                            "outcome_name": outcomes[i] if i < len(outcomes) else f"Outcome {i}",
                            "rewards_min_size": m.get("rewardsMinSize", None),
                            "rewards_max_spread": m.get("rewardsMaxSpread", None),
                            "spread_platform": m.get("spread", None),
                            "fpmm_live": m.get("fpmmLive", False),
                        }
                    })
            except Exception as e:
                print(f"[ERROR] Parsing market {m.get('id', '?')} → {e}")

        offset += limit

    return all_assets

def append_snapshot_to_parquet(df: pd.DataFrame, file_path: str):
    if os.path.exists(file_path):
        logger.info(f"Reading existing parquet file: {file_path}")
        df_existing = pd.read_parquet(file_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_parquet(file_path, index=False)
        logger.info(f"Appended {len(df)} rows to {file_path}. Total rows: {len(df_combined)}")
    else:
        df.to_parquet(file_path, index=False)
        logger.info(f"Created new parquet file {file_path} with {len(df)} rows")

def compute_metrics_for_assets(assets, client, save_path=None):
    results = []
    token_id_map = {a['token_id']: a['market_meta'] for a in assets}
    book_params = [BookParams(token_id=token_id) for token_id in token_id_map.keys()]
    
    logger.info(f"Processing {len(book_params)} order books in batches of 100")
    total_batches = (len(book_params) + 99) // 100  # Ceiling division

    order_books = {}
    for batch_num, batch in enumerate(chunked(book_params, 100), 1):
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} order books)")
        try:
            response = client.get_order_books(params=batch)
            if isinstance(response, dict):
                order_books.update(response)
                logger.info(f"Batch {batch_num}: Received {len(response)} order books (dict response)")
            elif isinstance(response, list):
                mapped = {bp.token_id: ob for bp, ob in zip(batch, response)}
                order_books.update(mapped)
                logger.info(f"Batch {batch_num}: Received {len(response)} order books (list response)")
            else:
                logger.warning(f"Batch {batch_num}: Unexpected response type: {type(response)}")
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {str(e)}")

    logger.info(f"Successfully retrieved {len(order_books)} order books out of {len(book_params)} requested")
    
    processed_count = 0
    error_count = 0
    for token_id, ob in order_books.items():
        meta = token_id_map.get(token_id)
        if not meta or ob is None:
            logger.warning(f"Skipping token {token_id}: Missing metadata or empty order book")
            continue
        try:
            metrics = compute_order_book_metrics(ob, token_id, meta)
            if metrics:
                results.append(metrics)
                processed_count += 1
                if processed_count % 50 == 0:  # Log progress every 50 tokens
                    logger.info(f"Processed {processed_count} tokens successfully")
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing token {token_id}: {str(e)}")

    logger.info(f"Completed processing: {processed_count} successful, {error_count} errors")
    
    df = pd.DataFrame(results)
    if save_path and not df.empty:
        append_snapshot_to_parquet(df, file_path=save_path)
    return df

def snapshot_loop(snapshot_file="liquidity_snapshots3.parquet"):
    client = ClobClient(host='https://clob.polymarket.com/')
    while True:
        start = datetime.utcnow()
        logger.info(f"Starting new snapshot cycle at {start:%Y-%m-%d %H:%M:%S} UTC")

        try:
            logger.info("Fetching CLOB enabled token IDs")
            assets = get_all_clob_enabled_token_ids()
            logger.info(f"Retrieved {len(assets)} assets")
            
            df = compute_metrics_for_assets(assets, client=client, save_path=snapshot_file)
            logger.info(f"Successfully saved {len(df)} rows to snapshot")
        except Exception as e:
            logger.error(f"Error during snapshot cycle: {str(e)}")

        duration = (datetime.utcnow() - start).total_seconds()
        logger.info(f"Snapshot cycle completed in {duration:.2f} seconds")

        # Wait until 10 min has passed since start
        wait_time = max(0, 600 - duration)
        logger.info(f"Waiting {wait_time:.2f} seconds until next snapshot cycle")
        time.sleep(wait_time)

if __name__ == "__main__":
    logger.info("Starting order book snapshot service")
    snapshot_loop("liquidity_snapshots3.parquet")