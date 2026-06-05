"""
Intraday Liquidity Regimes - Phase 1: Data Construction
========================================================
Constructs normalized intraday profiles and feature vectors for regime identification.

Key outputs:
- Per-stock-year intraday profiles (volume/volatility shares)
- Feature matrix for clustering (U-shape, entropy, concentration metrics)
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import time
from scipy.stats import skew, entropy as scipy_entropy
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
OUTPUT_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

# Trading session: 9:15 AM to 3:30 PM IST = 75 bars of 5-min each
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
BARS_PER_DAY = 75

# Feature extraction windows
OPEN_WINDOW = 6   # First 30 min (bars 0-5)
CLOSE_WINDOW = 6  # Last 30 min (bars 69-74)
MIDDAY_START = 18  # 10:45 AM
MIDDAY_END = 54    # 2:00 PM


def load_stock_data(symbol):
    """Load and preprocess stock 5-min data."""
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['trading_date'] = df['date'].dt.date
        df['time'] = df['date'].dt.time
        
        # Filter market hours
        df = df[(df['time'] >= MARKET_OPEN) & (df['time'] <= MARKET_CLOSE)]
        
        # Add bar index within day
        df['bar_idx'] = df.groupby('trading_date').cumcount()
        
        # Compute returns
        df['return'] = df.groupby('trading_date')['close'].pct_change()
        df['abs_return'] = df['return'].abs()
        
        return df
    except Exception as e:
        return None


def normalize_intraday_profile(day_df):
    """
    Normalize a single day's data to get intraday shares.
    
    Returns dict with:
        - vol_share: volume share per bar
        - volatility_share: |return| share per bar
        - illiq: Amihud illiquidity per bar
    """
    if len(day_df) < BARS_PER_DAY * 0.8:  # Skip incomplete days
        return None
    
    # Limit to first 75 bars
    day_df = day_df.head(BARS_PER_DAY)
    
    # Volume share
    total_vol = day_df['volume'].sum()
    if total_vol == 0:
        return None
    vol_share = day_df['volume'].values / total_vol
    
    # Volatility share
    total_abs_ret = day_df['abs_return'].sum()
    if total_abs_ret == 0:
        volatility_share = np.zeros(len(day_df))
    else:
        volatility_share = day_df['abs_return'].values / total_abs_ret
    
    # Amihud illiquidity (handle zeros)
    with np.errstate(divide='ignore', invalid='ignore'):
        illiq = np.where(day_df['volume'].values > 0,
                         day_df['abs_return'].values / day_df['volume'].values,
                         0)
    
    return {
        'vol_share': vol_share,
        'volatility_share': volatility_share,
        'illiq': illiq,
        'bar_idx': day_df['bar_idx'].values
    }


def compute_features_from_profile(profiles):
    """
    Compute regime fingerprint features from a list of daily profiles.
    
    Features:
        - open_intensity: Mean volume share in first 30 min
        - close_intensity: Mean volume share in last 30 min
        - midday_flatness: Std dev of midday volume shares
        - u_shape_strength: (open + close) - midday
        - volume_entropy: Entropy of volume distribution
        - peak_concentration: Max bar volume share
        - skewness: Skewness of volume distribution
    """
    if not profiles:
        return None
    
    # Average profile across days
    n_bars = BARS_PER_DAY
    vol_profiles = np.array([p['vol_share'][:n_bars] for p in profiles if len(p['vol_share']) >= n_bars])
    
    if len(vol_profiles) < 10:  # Need enough days
        return None
    
    avg_vol_profile = vol_profiles.mean(axis=0)
    
    # Normalize to ensure it sums to 1
    avg_vol_profile = avg_vol_profile / avg_vol_profile.sum()
    
    # Feature extraction
    open_intensity = avg_vol_profile[:OPEN_WINDOW].sum()
    close_intensity = avg_vol_profile[-CLOSE_WINDOW:].sum()
    midday_profile = avg_vol_profile[MIDDAY_START:MIDDAY_END]
    midday_mean = midday_profile.mean()
    midday_flatness = midday_profile.std()
    
    # U-shape strength
    u_shape_strength = (open_intensity + close_intensity) / 2 - midday_mean * (MIDDAY_END - MIDDAY_START)
    
    # Entropy (higher = more uniform distribution)
    # Add small epsilon to avoid log(0)
    vol_entropy = scipy_entropy(avg_vol_profile + 1e-10)
    
    # Peak concentration
    peak_concentration = avg_vol_profile.max()
    
    # Skewness (positive = right-skewed, more at close)
    bar_indices = np.arange(n_bars)
    mean_bar = np.sum(bar_indices * avg_vol_profile)
    vol_skewness = skew(avg_vol_profile)
    
    return {
        'open_intensity': open_intensity,
        'close_intensity': close_intensity,
        'midday_flatness': midday_flatness,
        'u_shape_strength': u_shape_strength,
        'volume_entropy': vol_entropy,
        'peak_concentration': peak_concentration,
        'skewness': vol_skewness,
        'n_days': len(vol_profiles)
    }


def process_stock_year(args):
    """Process a single stock-year combination."""
    symbol, year = args
    
    df = load_stock_data(symbol)
    if df is None:
        return None
    
    # Filter to year
    df['year'] = df['date'].dt.year
    year_df = df[df['year'] == year]
    
    if year_df.empty:
        return None
    
    # Get daily profiles
    profiles = []
    for trading_date, day_df in year_df.groupby('trading_date'):
        profile = normalize_intraday_profile(day_df)
        if profile is not None:
            profiles.append(profile)
    
    # Compute features
    features = compute_features_from_profile(profiles)
    if features is None:
        return None
    
    features['symbol'] = symbol
    features['year'] = year
    
    return features


def build_feature_matrix(years=None, max_workers=8):
    """
    Build the complete feature matrix for all stocks and years.
    """
    if years is None:
        years = list(range(2015, 2026))  # 2015-2025
    
    # Get all stock symbols
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols = [os.path.basename(f).replace('.csv', '') for f in stock_files]
    
    print(f"Processing {len(symbols)} stocks across {len(years)} years...")
    
    # Create all stock-year combinations
    tasks = [(sym, year) for sym in symbols for year in years]
    
    print(f"Total tasks: {len(tasks)}")
    
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_stock_year, task): task for task in tasks}
        
        count = 0
        for future in as_completed(futures):
            count += 1
            if count % 500 == 0:
                print(f"Processed {count}/{len(tasks)} stock-years...")
            
            result = future.result()
            if result is not None:
                results.append(result)
    
    print(f"Completed! Got {len(results)} valid stock-year features.")
    
    # Create DataFrame
    feature_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['symbol', 'year', 'open_intensity', 'close_intensity', 
            'midday_flatness', 'u_shape_strength', 'volume_entropy',
            'peak_concentration', 'skewness', 'n_days']
    feature_df = feature_df[cols]
    
    return feature_df


def compute_average_profiles(symbols=None, year=None, max_workers=8):
    """
    Compute average intraday profiles for visualization.
    Returns bar-by-bar average volume shares.
    """
    if symbols is None:
        stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        symbols = [os.path.basename(f).replace('.csv', '') for f in stock_files][:50]
    
    all_profiles = []
    
    for sym in symbols:
        df = load_stock_data(sym)
        if df is None:
            continue
        
        if year:
            df = df[df['date'].dt.year == year]
        
        for trading_date, day_df in df.groupby('trading_date'):
            profile = normalize_intraday_profile(day_df)
            if profile is not None and len(profile['vol_share']) >= BARS_PER_DAY:
                all_profiles.append(profile['vol_share'][:BARS_PER_DAY])
    
    if not all_profiles:
        return None
    
    avg_profile = np.mean(all_profiles, axis=0)
    return avg_profile / avg_profile.sum()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Construct intraday profiles')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--years', type=str, default='2015-2025', help='Year range')
    args = parser.parse_args()
    
    if args.test:
        print("Running quick test...")
        # Test with 5 stocks, 2 years
        test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        test_years = [2023, 2024]
        
        for sym in test_symbols:
            for year in test_years:
                result = process_stock_year((sym, year))
                if result:
                    print(f"{sym} {year}: U-shape={result['u_shape_strength']:.4f}, "
                          f"Open={result['open_intensity']:.3f}, Close={result['close_intensity']:.3f}")
        print("\nTest completed successfully!")
    else:
        # Parse year range
        start_year, end_year = map(int, args.years.split('-'))
        years = list(range(start_year, end_year + 1))
        
        # Build full matrix
        feature_df = build_feature_matrix(years=years)
        
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "intraday_features.csv")
        feature_df.to_csv(output_path, index=False)
        print(f"\nSaved feature matrix to {output_path}")
        
        # Summary stats
        print("\nFeature Summary:")
        print(feature_df.describe())
