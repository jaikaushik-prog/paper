import pandas as pd
import numpy as np

def prepare_push_response_data(df, lag_l):
    """
    df: DataFrame with 'close' column, indexed by datetime (5min data)
    lag_l: Integer, number of 5-min periods for the lag
    """
    # Create a copy to avoid SettingWithCopy warnings on the original df
    df = df.copy()
    
    # 1. Calculate Raw Push and Response (Log Returns)
    # Push: Return from t-L to t
    # p_t(L) = ln(Close_t) - ln(Close_{t-L})
    df['push_raw'] = np.log(df['close'] / df['close'].shift(lag_l))
    
    # Response: Return from t to t+L
    # r_t(L) = ln(Close_{t+L}) - ln(Close_t)
    df['resp_raw'] = np.log(df['close'].shift(-lag_l) / df['close'])
    
    # Drop NaNs created by shifting
    df = df.dropna()
    
    if df.empty:
        return df
    
    # 2. Standardization (Z-Score)
    # Use a rolling window of 1 month (approx 1500 5-min bars).
    window = 1500 
    
    # We need rolling mean and std of the Push and Response series
    # standardizing by LOCAL volatility (rolling window) is crucial as per paper/plan
    push_mean = df['push_raw'].rolling(window).mean()
    push_std = df['push_raw'].rolling(window).std()
    
    resp_mean = df['resp_raw'].rolling(window).mean()
    resp_std = df['resp_raw'].rolling(window).std()
    
    # Z-scores
    df['z_push'] = (df['push_raw'] - push_mean) / push_std
    df['z_resp'] = (df['resp_raw'] - resp_mean) / resp_std
    
    return df.dropna()

def get_surface_for_lag(panel_data):
    """
    panel_data: Combined DF of all stocks in a specific group for a SPECIFIC LAG.
                Must contain 'z_push' and 'z_resp'.
    """
    if panel_data.empty:
        return None

    # 3. Binning by Push Magnitude
    # The paper uses bins of width 0.025 from -4 to 4. 
    # We use width 0.1 as per user request/plan for robustness.
    # We want bins centered? Or edges? pd.cut uses edges.
    bins = np.arange(-4.0, 4.05, 0.1) 
    centers = (bins[:-1] + bins[1:]) / 2
    
    # pd.cut returns a categorical index
    # Use centers as labels so the groupby index is the float center
    panel_data['bin'] = pd.cut(panel_data['z_push'], bins=bins, labels=centers)
    
    # 4. Calculate Expected Response per Bin
    # E[z_resp | z_push]
    # We groupby bin and take mean of z_resp
    surface = panel_data.groupby('bin', observed=True)['z_resp'].mean()
    
    return surface
