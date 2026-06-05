import pandas as pd
import numpy as np

def create_adaptive_volume_bars(df, bars_per_day=50):
    """
    df: 5-min OHLCV data. Must have 'date', 'open', 'high', 'low', 'close', 'volume'.
    bars_per_day: Target number of bars per day (e.g., 50).
    """
    # Ensure DataFrame has a datetime index or 'date' column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df.set_index('date').sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    else:
        # Fallback or error
        return pd.DataFrame()

    # 1. Calculate Dynamic Threshold (The "target")
    # Resample to daily to get daily volume
    # Use 'D' for calendar day resampling
    daily_vol = df['volume'].resample('D').sum()
    
    # Calculate rolling 20-day mean, SHIFTED by 1 to avoid lookahead bias
    # We want today's target to be based on the PAST 20 days.
    rolling_adv = daily_vol.rolling(window=20).mean().shift(1)
    
    # Reindex rolling_adv to match the 5-min index (forward fill)
    # This assigns the daily target to every 5-min bar of that day
    # We forward fill the daily value.
    target_vol_series = rolling_adv.reindex(df.index, method='ffill')
    
    # Calculate target per bar
    df['target_vol'] = target_vol_series / bars_per_day
    
    # Drop the startup period where Rolling ADV is NaN (first 20 days)
    df_clean = df.dropna(subset=['target_vol'])
    
    if df_clean.empty:
        return pd.DataFrame()

    # Optimized Bucket Fill
    # We can iterate, but pure python loops on 10 years of 5-min data (approx 75*250*10 = 187k rows) might be slow?
    # 200k iterations is fine in Python (~1 second).
    
    volume_bars = []
    
    current_vol = 0
    current_high = -np.inf
    current_low = np.inf
    current_open = None
    
    # Iterate as tuples for speed
    for row in df_clean.itertuples():
        if current_open is None:
            current_open = row.open
            
        current_high = max(current_high, row.high)
        current_low = min(current_low, row.low)
        current_vol += row.volume
        
        target = row.target_vol
        
        # Safety check for zero target
        if target <= 0: 
            target = 1e9
            
        if current_vol >= target:
            # Bucket full, close the bar
            volume_bars.append({
                'timestamp': row.Index,
                'open': current_open,
                'high': current_high,
                'low': current_low,
                'close': row.close,
                'volume': current_vol
            })
            
            # Reset
            current_vol = 0
            current_open = None
            current_high = -np.inf
            current_low = np.inf
            
    if not volume_bars:
        return pd.DataFrame()
        
    return pd.DataFrame(volume_bars)
