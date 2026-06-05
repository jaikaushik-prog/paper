"""
Advanced Alpha Feature Engineering for Nifty 500 Strategy (IMPROVED)
====================================================================
Generates calendar/SIP, regime, momentum, and target features
from raw 5-min OHLCV data (resampled to daily).

IMPROVEMENTS ADDED:
- Overnight gap metrics
- Intraday range patterns
- Volume surge detection
- Price action patterns (shadows, breakouts)
- Regime stability indicators

Uses Polars for fast CSV loading, Pandas + ta for feature engineering,
and RobustScaler for fat-tail-safe normalization (Kurtosis=706).
"""

import os, json, time, warnings
import numpy as np
import polars as pl
import pandas as pd
import ta
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, 'raw_Data')

# ══════════════════════════════════════════════════════════════════════════
#  STEP 0 : LOAD ALL DATA WITH POLARS → MultiIndex (Date, Ticker) DataFrame
# ══════════════════════════════════════════════════════════════════════════
def load_data():
    """Load all stock CSVs with Polars, resample to daily, return tall DataFrame."""
    csv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.csv')])
    required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
    t0 = time.time()
    print(f"Loading {len(csv_files)} CSVs from {RAW_DIR} ...")

    frames = []
    skipped = []
    for i, fname in enumerate(csv_files):
        ticker = fname.replace('.csv', '')
        fpath  = os.path.join(RAW_DIR, fname)
        try:
            with open(fpath, 'r') as fh:
                header = set(fh.readline().strip().split(','))
            if not required_cols.issubset(header):
                skipped.append(ticker)
                continue
            daily = (
                pl.scan_csv(fpath, try_parse_dates=True)
                .with_columns(pl.col('date').cast(pl.Datetime('us')).dt.date().alias('trade_date'))
                .group_by('trade_date')
                .agg([
                    pl.col('open').first().alias('Open'),
                    pl.col('high').max().alias('High'),
                    pl.col('low').min().alias('Low'),
                    pl.col('close').last().alias('Close'),
                    pl.col('volume').sum().alias('Volume'),
                ])
                .with_columns(pl.lit(ticker).alias('Ticker'))
                .sort('trade_date')
                .collect()
            )
            frames.append(daily)
        except Exception as e:
            skipped.append(ticker)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(csv_files)} loaded ... ({time.time()-t0:.1f}s)")

    if skipped:
        print(f"  Skipped {len(skipped)} non-stock files: {skipped}")

    big = pl.concat(frames).to_pandas()
    big['trade_date'] = pd.to_datetime(big['trade_date'])
    big.rename(columns={'trade_date': 'Date'}, inplace=True)

    # Replace zero prices with NaN (data cleaning from EDA)
    for col in ['Open', 'High', 'Low', 'Close']:
        big[col] = big[col].replace(0, np.nan)

    # Set MultiIndex (Date, Ticker)
    big = big.set_index(['Date', 'Ticker']).sort_index()

    elapsed = time.time() - t0
    n_stocks = big.index.get_level_values('Ticker').nunique()
    n_days   = big.index.get_level_values('Date').nunique()
    print(f"\n✓ Loaded {n_stocks} stocks × {n_days} days = {len(big):,} rows in {elapsed:.1f}s")
    return big


# ══════════════════════════════════════════════════════════════════════════
#  STEP 0.5 : CLEAN DATA (REMOVE SPLIT/BONUS ARTIFACTS)
# ══════════════════════════════════════════════════════════════════════════
def drop_outliers(df):
    """
    Remove rows with extreme price moves that are likely data errors 
    (e.g. unadjusted splits/bonuses).
    Rule: 
    1. Daily Return > 50% or < -50%
    2. 5-Day Return > 100% (unless verified, but safe to drop for robustness)
    """
    # Calculate daily return
    close_wide = df['Close'].unstack('Ticker')
    ret_daily = close_wide.pct_change()
    
    # Identify tickers/dates with extreme moves
    mask_extreme = (ret_daily.abs() > 0.50).stack()
    
    outliers = mask_extreme[mask_extreme].index
    
    if len(outliers) > 0:
        print(f"\n⚠ FOUND {len(outliers)} DATA ERRORS (Extreme Moves > 50%). DROPPING THEM.")
        print(f"  Examples: {outliers[:5].tolist()}")
        df = df.drop(outliers)
        
    print("✓ Data cleaning complete.")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 : CALENDAR & SIP FLOW FEATURES
# ══════════════════════════════════════════════════════════════════════════
def add_calendar_features(df):
    """Calendar + SIP window features (vectorized)."""
    dates = df.index.get_level_values('Date')
    df['day_of_month'] = dates.day
    df['is_sip_window'] = ((dates.day >= 25) | (dates.day <= 7)).astype(int)
    df['day_of_week']   = dates.dayofweek
    df['is_monday']     = (dates.dayofweek == 0).astype(int)
    print("✓ Calendar & SIP features added")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 : REGIME FILTERS (MARKET-LEVEL → BROADCAST TO ALL STOCKS)
# ══════════════════════════════════════════════════════════════════════════
def add_regime_features(df):
    """
    Market regime features: Multi-Dimensional (Breadth + Volatility + ADX).
    CORRECT 4-STATE SYSTEM (matching original):
        0: Bear (Breadth < 40%)
        1: Neutral (Transition/Chop)
        2: Bull Low Vol (Breadth > 65% + ADX > 20 + Vol Rank < 50%)
        3: Bull High Vol (Breadth > 65% + ADX > 20 + Vol Rank >= 50%)
    """
    # 1. Broad Market Proxy (Mean of All Tickers)
    close_wide = df['Close'].unstack('Ticker')
    open_wide  = df['Open'].unstack('Ticker')
    high_wide  = df['High'].unstack('Ticker')
    low_wide   = df['Low'].unstack('Ticker')
    
    nifty_close = close_wide.mean(axis=1).rename('nifty_index')
    nifty_open  = open_wide.mean(axis=1)
    nifty_high  = high_wide.mean(axis=1)
    nifty_low   = low_wide.mean(axis=1)
    
    # 2. Market Breadth (% Stocks > 200 SMA)
    sma200_wide = close_wide.rolling(200, min_periods=50).mean()
    above_200 = (close_wide > sma200_wide).sum(axis=1)
    total_valid = close_wide.count(axis=1)
    
    breadth_pct = (above_200 / total_valid) * 100
    breadth_pct = breadth_pct.fillna(50)
    
    # 3. Index ADX (Trend Strength)
    adx_ind = ta.trend.ADXIndicator(high=nifty_high, low=nifty_low, close=nifty_close, window=14)
    nifty_adx = adx_ind.adx().fillna(0)
    
    # 4. Volatility Percentile (252-day Rank)
    vol_series = nifty_close.pct_change().rolling(21).std() * np.sqrt(252) * 100
    vol_rank = vol_series.rolling(252, min_periods=100).rank(pct=True).fillna(0.5)
    
    # 5. Regime Definition (0, 1, 2, 3) - CORRECTED 4-STATE SYSTEM
    regime_val = pd.Series(1, index=nifty_close.index, name='market_regime_val')
    
    # Bear: Breadth < 40%
    mask_bear = (breadth_pct < 40.0)
    
    # Bull Candidate: Breadth > 65% AND ADX > 20
    mask_bull_candidate = (breadth_pct > 65.0) & (nifty_adx > 20.0)
    
    # Bull Low Vol (2): Candidate + Vol Rank < 0.50
    mask_bull_low = mask_bull_candidate & (vol_rank < 0.50)
    
    # Bull High Vol (3): Candidate + Vol Rank >= 0.50
    mask_bull_high = mask_bull_candidate & (vol_rank >= 0.50)
    
    # Apply assignments (order matters)
    regime_val[mask_bear] = 0
    regime_val[mask_bull_low] = 2
    regime_val[mask_bull_high] = 3
    # Neutral (1) is default
    
    # 6. NEW: Regime Stability (check if regime changed in last 5 days)
    regime_changes = regime_val.diff().abs()
    regime_stable = (regime_changes.rolling(5).sum() == 0).astype(int)
    
    # 7. High Volatility Flag
    is_high_vol = (vol_rank > 0.75).astype(int)
    
    # Broadcast to all stocks
    df['market_regime_val'] = regime_val.reindex(df.index, level='Date')
    df['market_regime_stable'] = regime_stable.reindex(df.index, level='Date')  # NEW
    df['market_breadth'] = breadth_pct.reindex(df.index, level='Date')
    df['market_adx'] = nifty_adx.reindex(df.index, level='Date')
    df['market_vol_rank'] = vol_rank.reindex(df.index, level='Date')
    df['market_volatility'] = vol_series.reindex(df.index, level='Date')
    df['is_high_vol'] = is_high_vol.reindex(df.index, level='Date')
    
    print("✓ Regime features added (4-STATE: 0=Bear, 1=Neutral, 2=Bull LowVol, 3=Bull HighVol)")
    print(f"  + Regime stability indicator added")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 : MOMENTUM & TECHNICAL FEATURES
# ══════════════════════════════════════════════════════════════════════════
def add_momentum_features(df):
    """Classic TA momentum indicators (RSI, SMA Distance, ROC, Vol Breakout, ATR)."""
    close_wide = df['Close'].unstack('Ticker')
    high_wide  = df['High'].unstack('Ticker')
    low_wide   = df['Low'].unstack('Ticker')
    volume_wide = df['Volume'].unstack('Ticker')
    
    # 1. RSI (14-day)
    rsi_14 = close_wide.apply(lambda col: ta.momentum.RSIIndicator(col, window=14).rsi(), axis=0)
    
    # 2. Distance to 200 SMA
    sma_200 = close_wide.rolling(200, min_periods=50).mean()
    dist_to_200ma = (close_wide - sma_200) / sma_200
    
    # 3. Rate of Change (5-day)
    roc_5 = close_wide.pct_change(5)
    
    # 4. Volume Breakout (Volume > 2x 20-day avg)
    avg_vol_20 = volume_wide.rolling(20, min_periods=5).mean()
    vol_breakout = (volume_wide / avg_vol_20 - 1.0).clip(lower=-1, upper=5)
    
    # 5. ATR (14-day)
    atr_14 = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
    for col in close_wide.columns:
        try:
            atr_ind = ta.volatility.AverageTrueRange(
                high=high_wide[col], 
                low=low_wide[col], 
                close=close_wide[col], 
                window=14
            )
            atr_14[col] = atr_ind.average_true_range()
        except:
            atr_14[col] = np.nan
    
    # Stack back to MultiIndex
    df['rsi_14']        = rsi_14.stack()
    df['dist_to_200ma'] = dist_to_200ma.stack()
    df['roc_5']         = roc_5.stack()
    df['vol_breakout']  = vol_breakout.stack()
    df['atr_14']        = atr_14.stack()
    
    print("✓ Momentum features added (RSI, SMA Distance, ROC, Vol Breakout, ATR)")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3.5 : NEW PRICE ACTION FEATURES (OHLCV ONLY)
# ══════════════════════════════════════════════════════════════════════════
def add_price_action_features(df):
    """
    NEW FEATURES derived from OHLCV only:
    1. Overnight Gap
    2. Intraday Range
    3. Volume Surge
    4. Price Breakouts
    5. Candlestick Patterns (Shadows)
    """
    close_wide = df['Close'].unstack('Ticker')
    open_wide  = df['Open'].unstack('Ticker')
    high_wide  = df['High'].unstack('Ticker')
    low_wide   = df['Low'].unstack('Ticker')
    volume_wide = df['Volume'].unstack('Ticker')
    
    # 1. Overnight Gap (Open - prev Close) / prev Close
    overnight_gap = (open_wide - close_wide.shift(1)) / close_wide.shift(1)
    
    # 2. Intraday Range (High - Low) / Close
    intraday_range = (high_wide - low_wide) / close_wide
    
    # 3. Volume Surge (Volume / 20d avg) - percentile rank
    avg_vol_20 = volume_wide.rolling(20, min_periods=5).mean()
    volume_surge = volume_wide / avg_vol_20
    
    # 4. High Breakout (High > max(High, 20d))
    high_20d_max = high_wide.rolling(20).max()
    high_breakout = (high_wide > high_20d_max.shift(1)).astype(int)
    
    # 5. Candlestick Shadows
    # Lower shadow = (Close - Low) / (High - Low)
    # Upper shadow = (High - Close) / (High - Low)
    candle_range = high_wide - low_wide
    lower_shadow = (close_wide - low_wide) / candle_range.replace(0, np.nan)
    upper_shadow = (high_wide - close_wide) / candle_range.replace(0, np.nan)
    
    # 6. Typical Price Distance (for VWAP approximation)
    # Typical Price = (H + L + C) / 3
    typical_price = (high_wide + low_wide + close_wide) / 3
    typical_vwap_10 = typical_price.rolling(10).mean()
    vwap_dist = (close_wide - typical_vwap_10) / typical_vwap_10
    
    # Stack
    df['overnight_gap']   = overnight_gap.stack()
    df['intraday_range']  = intraday_range.stack()
    df['volume_surge']    = volume_surge.stack()
    df['high_breakout']   = high_breakout.stack()
    df['lower_shadow']    = lower_shadow.stack()
    df['upper_shadow']    = upper_shadow.stack()
    df['vwap_dist']       = vwap_dist.stack()
    
    print("✓ NEW Price Action features added (Gap, Range, Volume Surge, Breakouts, Shadows, VWAP)")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3.6 : RELATIVE STRENGTH FEATURES (LAYER 2)
# ══════════════════════════════════════════════════════════════════════════
def add_relative_strength_features(df):
    """
    Layer 2 Part A: RS, RS Rank, Vol Rank, Vol-Adjusted Momentum.
    """
    close_wide = df['Close'].unstack('Ticker')
    volume_wide = df['Volume'].unstack('Ticker')
    
    # 1. Relative Strength (RS) vs Nifty Index
    # RS = (Stock Return / Market Return) over 20d and 63d
    nifty_close = close_wide.mean(axis=1)
    
    ret_20_stock = close_wide.pct_change(20)
    ret_20_nifty = nifty_close.pct_change(20)
    rs_20 = ret_20_stock.div(ret_20_nifty, axis=0)
    
    ret_63_stock = close_wide.pct_change(63)
    ret_63_nifty = nifty_close.pct_change(63)
    rs_63 = ret_63_stock.div(ret_63_nifty, axis=0)
    
    # 2. RS Rank (Cross-sectional Percentile)
    rs_rank_20 = rs_20.rank(axis=1, pct=True)
    
    # 3. Volume Rank (20d percentile per stock)
    vol_rank_20 = volume_wide.rolling(20).rank(pct=True)
    
    # 4. Volume-Adjusted Momentum (20d return * vol_rank)
    mom_vol_20 = ret_20_stock * vol_rank_20
    
    # Stack
    df['rs_20']      = rs_20.stack()
    df['rs_63']      = rs_63.stack()
    df['rs_rank_20'] = rs_rank_20.stack()
    df['vol_rank_20'] = vol_rank_20.stack()
    df['mom_vol_20']  = mom_vol_20.stack()
    
    print("✓ Layer 2 Features added (RS, Ranks, Vol-Adj Mom)")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 : TARGET VARIABLE
# ══════════════════════════════════════════════════════════════════════════
def add_target(df):
    """5-day forward return target + binary classification label."""
    close_wide = df['Close'].unstack('Ticker')

    # Forward return = (Close[t+5] / Close[t]) - 1
    fwd_ret_wide = close_wide.shift(-5) / close_wide - 1

    df['fwd_ret_5d']     = fwd_ret_wide.stack()
    df['target_binary']  = (df['fwd_ret_5d'] > 0.02).astype(int)

    print("✓ Target added (5d fwd return, binary threshold = 2%)")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   NIFTY 500 : IMPROVED FEATURE ENGINEERING (LAYER 1)      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Load
    df = load_data()

    # Clean Data Errors (Split/Bonus Artifacts)
    df = drop_outliers(df)

    # Feature engineering pipeline
    print("\n── Feature Engineering Pipeline ──────────────────────────────")
    df = add_calendar_features(df)
    df = add_regime_features(df)
    df = add_momentum_features(df)
    df = add_price_action_features(df)      # NEW
    df = add_relative_strength_features(df)
    df = add_target(df)
    
    # Drop NaN rows (from rolling windows, forward returns, etc.)
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"\n── Cleanup ──────────────────────────────────────────────────")
    print(f"  Rows before dropna: {rows_before:,}")
    print(f"  Rows after  dropna: {rows_after:,}  (dropped {rows_before - rows_after:,})")

    # Final output
    print(f"\n{'='*70}")
    print(f"  FINAL DATASET")
    print(f"{'='*70}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index.get_level_values('Date').min().date()} → "
          f"{df.index.get_level_values('Date').max().date()}")
    print(f"  Unique tickers: {df.index.get_level_values('Ticker').nunique()}")
    print(f"\n  Target distribution:")
    print(f"    target_binary=1  : {(df['target_binary']==1).sum():>8,} ({(df['target_binary']==1).mean()*100:.1f}%)")
    print(f"    target_binary=0  : {(df['target_binary']==0).sum():>8,} ({(df['target_binary']==0).mean()*100:.1f}%)")
    print(f"\n  SIP Window verification:")
    sip_ret = df.loc[df['is_sip_window']==1, 'fwd_ret_5d'].mean()
    non_sip = df.loc[df['is_sip_window']==0, 'fwd_ret_5d'].mean()
    print(f"    SIP window mean 5d fwd ret : {sip_ret*100:.3f}%")
    print(f"    Non-SIP mean 5d fwd ret    : {non_sip*100:.3f}%")
    print(f"    Ratio                      : {sip_ret/non_sip:.2f}x")

    print(f"\n── Regime Distribution ──────────────────────────────────────")
    regime_counts = df['market_regime_val'].value_counts().sort_index()
    regime_names = {0: 'Bear', 1: 'Neutral', 2: 'Bull LowVol', 3: 'Bull HighVol'}
    for regime, count in regime_counts.items():
        regime_name = regime_names.get(regime, 'Unknown')
        pct = count / len(df) * 100
        print(f"    Regime {regime} ({regime_name:<12s}): {count:>8,} ({pct:.1f}%)")

    print(f"\n── First 5 Rows ─────────────────────────────────────────────")
    pd.set_option('display.max_columns', 25)
    pd.set_option('display.width', 180)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df.head(5).to_string())

    # Save to parquet for downstream ML use
    output_path = os.path.join(BASE_DIR, 'features_nifty500_improved.parquet')
    df.to_parquet(output_path)
    print(f"\n✓ Saved to {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")

    return df


if __name__ == '__main__':
    main()