"""
S&P 500 Push-Response Library
Mathematical core for computing push-response z-scores using daily data.

Adapted from push_response_lib.py for daily frequency data.
"""

import pandas as pd
import numpy as np

def prepare_push_response_data(df, lag_l, window=60):
    """
    Compute Push and Response Z-scores for a stock's daily data.
    
    Parameters:
    -----------
    df : DataFrame with 'close' column
    lag_l : int, number of trading days for the lag
    window : int, rolling window for standardization (default: 60 days ~ 3 months)
    
    Returns:
    --------
    DataFrame with 'z_push' and 'z_resp' columns
    """
    df = df.copy()
    
    # 1. Calculate Raw Push and Response (Log Returns)
    # Push: Return from t-L to t
    df['push_raw'] = np.log(df['close'] / df['close'].shift(lag_l))
    
    # Response: Return from t to t+L
    df['resp_raw'] = np.log(df['close'].shift(-lag_l) / df['close'])
    
    # Drop NaNs from shifting
    df = df.dropna()
    
    if df.empty or len(df) < window:
        return pd.DataFrame()
    
    # 2. Rolling Z-Score Standardization
    push_mean = df['push_raw'].rolling(window, min_periods=window//2).mean()
    push_std = df['push_raw'].rolling(window, min_periods=window//2).std()
    
    resp_mean = df['resp_raw'].rolling(window, min_periods=window//2).mean()
    resp_std = df['resp_raw'].rolling(window, min_periods=window//2).std()
    
    # Avoid division by zero
    push_std = push_std.replace(0, np.nan)
    resp_std = resp_std.replace(0, np.nan)
    
    # Z-scores
    df['z_push'] = (df['push_raw'] - push_mean) / push_std
    df['z_resp'] = (df['resp_raw'] - resp_mean) / resp_std
    
    return df[['z_push', 'z_resp']].dropna()

def get_surface_for_lag(panel_data, bin_width=0.25):
    """
    Compute average response per push bin for surface visualization.
    
    Parameters:
    -----------
    panel_data : DataFrame with 'z_push' and 'z_resp' columns
    bin_width : float, width of z-score bins (default: 0.25 for daily data)
    
    Returns:
    --------
    Series with bin centers as index and mean response as values
    """
    if panel_data.empty:
        return None
    
    # Create bins from -4 to +4
    bins = np.arange(-4.0, 4.0 + bin_width, bin_width)
    centers = (bins[:-1] + bins[1:]) / 2
    
    # Bin the push values
    panel_data = panel_data.copy()
    panel_data['bin'] = pd.cut(panel_data['z_push'], bins=bins, labels=centers)
    
    # Calculate mean response per bin
    surface = panel_data.groupby('bin', observed=True)['z_resp'].mean()
    
    return surface

def get_surface_stats(panel_data, bin_width=0.25):
    """
    Compute detailed statistics per bin for analysis.
    
    Returns:
    --------
    DataFrame with count, mean, std, sem for each bin
    """
    if panel_data.empty:
        return None
    
    bins = np.arange(-4.0, 4.0 + bin_width, bin_width)
    centers = (bins[:-1] + bins[1:]) / 2
    
    panel_data = panel_data.copy()
    panel_data['bin'] = pd.cut(panel_data['z_push'], bins=bins, labels=centers)
    
    stats = panel_data.groupby('bin', observed=True)['z_resp'].agg([
        'count', 'mean', 'std'
    ])
    stats['sem'] = stats['std'] / np.sqrt(stats['count'])
    stats['t_stat'] = stats['mean'] / stats['sem']
    
    return stats
