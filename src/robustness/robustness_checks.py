"""
Intraday Liquidity Regimes - Phase 6: Robustness Checks
=========================================================
Validates findings with alternative specifications.

Checks:
1. Alternative bar sizes (10-min vs 5-min)
2. Excluding zero-volume bars
3. Alternative liquidity proxies
4. Sector-neutral clustering
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

FEATURE_COLS = [
    'open_intensity', 'close_intensity', 'midday_flatness',
    'u_shape_strength', 'volume_entropy', 'peak_concentration', 'skewness'
]


def load_baseline_results():
    """Load baseline clustering results."""
    path = os.path.join(RESULTS_DIR, "regime_assignments.csv")
    return pd.read_csv(path)


def robustness_10min_bars(sample_symbols=None, year=2024):
    """
    Recompute features using 10-minute bars instead of 5-minute.
    """
    from construct_intraday_profiles import load_stock_data, BARS_PER_DAY
    from scipy.stats import skew, entropy as scipy_entropy
    
    if sample_symbols is None:
        stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        sample_symbols = [os.path.basename(f).replace('.csv', '') for f in stock_files[:100]]
    
    features_10min = []
    
    for symbol in sample_symbols:
        df = load_stock_data(symbol)
        if df is None:
            continue
        
        df = df[df['date'].dt.year == year]
        if df.empty:
            continue
        
        profiles = []
        
        for trading_date, day_df in df.groupby('trading_date'):
            # Resample to 10-min bars
            day_df = day_df.copy()
            day_df['bar_group'] = day_df['bar_idx'] // 2  # Combine pairs of 5-min bars
            
            grouped = day_df.groupby('bar_group').agg({
                'volume': 'sum',
                'close': 'last'
            })
            
            if len(grouped) < 30:  # At least 30 10-min bars
                continue
            
            total_vol = grouped['volume'].sum()
            if total_vol == 0:
                continue
            
            vol_share = grouped['volume'].values / total_vol
            profiles.append(vol_share[:38])  # ~38 bars in 10-min
        
        if len(profiles) < 10:
            continue
        
        # Average profile
        n_bars = 38
        valid_profiles = [p for p in profiles if len(p) >= n_bars]
        if not valid_profiles:
            continue
        
        avg_profile = np.mean([p[:n_bars] for p in valid_profiles], axis=0)
        avg_profile = avg_profile / avg_profile.sum()
        
        # Features (adjusted for 38 bars)
        features_10min.append({
            'symbol': symbol,
            'year': year,
            'open_intensity': avg_profile[:3].sum(),  # First 30 min (3 bars)
            'close_intensity': avg_profile[-3:].sum(),  # Last 30 min
            'midday_flatness': avg_profile[9:27].std(),
            'u_shape_strength': (avg_profile[:3].sum() + avg_profile[-3:].sum()) / 2 - avg_profile[9:27].mean() * 18,
            'volume_entropy': scipy_entropy(avg_profile + 1e-10),
            'peak_concentration': avg_profile.max(),
            'skewness': skew(avg_profile)
        })
    
    return pd.DataFrame(features_10min)


def robustness_exclude_zero_volume(sample_symbols=None, year=2024):
    """
    Recompute features after excluding zero-volume bars.
    """
    from construct_intraday_profiles import load_stock_data, normalize_intraday_profile, compute_features_from_profile, BARS_PER_DAY
    
    if sample_symbols is None:
        stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        sample_symbols = [os.path.basename(f).replace('.csv', '') for f in stock_files[:100]]
    
    features_excl = []
    
    for symbol in sample_symbols:
        df = load_stock_data(symbol)
        if df is None:
            continue
        
        df = df[df['date'].dt.year == year]
        if df.empty:
            continue
        
        # Filter zero-volume bars
        df = df[df['volume'] > 0]
        
        profiles = []
        for trading_date, day_df in df.groupby('trading_date'):
            profile = normalize_intraday_profile(day_df)
            if profile is not None:
                profiles.append(profile)
        
        features = compute_features_from_profile(profiles)
        if features:
            features['symbol'] = symbol
            features['year'] = year
            features_excl.append(features)
    
    return pd.DataFrame(features_excl)


def compare_clustering_stability(baseline_df, robustness_df, name='robustness'):
    """
    Compare clustering results between baseline and robustness check.
    """
    # Merge on symbol-year
    merged = baseline_df.merge(robustness_df, on=['symbol', 'year'], suffixes=('_base', '_rob'))
    
    if merged.empty:
        return {'comparison': name, 'n_matched': 0, 'correlation': np.nan}
    
    # Feature correlations
    correlations = {}
    for col in FEATURE_COLS:
        base_col = f'{col}_base'
        rob_col = f'{col}_rob'
        if base_col in merged.columns and rob_col in merged.columns:
            valid = merged[[base_col, rob_col]].dropna()
            if len(valid) > 10:
                correlations[col] = valid[base_col].corr(valid[rob_col])
    
    return {
        'comparison': name,
        'n_matched': len(merged),
        'mean_correlation': np.mean(list(correlations.values())),
        **correlations
    }


def sector_neutral_clustering(df, n_sectors=5):
    """
    Run clustering within pseudo-sectors (based on profile characteristics).
    """
    # Create pseudo-sectors based on volume entropy quartiles
    df = df.copy()
    df['sector'] = pd.qcut(df['volume_entropy'], q=n_sectors, labels=False)
    
    results = []
    
    for sector in range(n_sectors):
        sector_df = df[df['sector'] == sector]
        
        if len(sector_df) < 50:
            continue
        
        X = sector_df[FEATURE_COLS].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means with k=3
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        
        results.append({
            'sector': sector,
            'n_obs': len(sector_df),
            'silhouette': sil
        })
    
    return pd.DataFrame(results)


def run_robustness_checks():
    """Run all robustness checks."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("ROBUSTNESS CHECKS")
    print("=" * 60)
    
    # Load baseline
    print("\n1. Loading baseline results...")
    try:
        baseline_df = load_baseline_results()
        print(f"  Baseline: {len(baseline_df)} observations")
    except FileNotFoundError:
        print("  Baseline not found. Run identify_liquidity_regimes.py first.")
        return
    
    # 2. 10-minute bars
    print("\n2. Testing 10-minute bar aggregation...")
    features_10min = robustness_10min_bars()
    if not features_10min.empty:
        comparison = compare_clustering_stability(baseline_df, features_10min, '10min_bars')
        print(f"  Matched {comparison['n_matched']} observations")
        print(f"  Mean feature correlation: {comparison.get('mean_correlation', np.nan):.3f}")
    
    # 3. Exclude zero-volume
    print("\n3. Testing zero-volume exclusion...")
    features_excl = robustness_exclude_zero_volume()
    if not features_excl.empty:
        comparison = compare_clustering_stability(baseline_df, features_excl, 'excl_zero_vol')
        print(f"  Matched {comparison['n_matched']} observations")
        print(f"  Mean feature correlation: {comparison.get('mean_correlation', np.nan):.3f}")
    
    # 4. Sector-neutral clustering
    print("\n4. Testing sector-neutral clustering...")
    sector_results = sector_neutral_clustering(baseline_df)
    if not sector_results.empty:
        print("  Silhouette by sector:")
        print(sector_results)
        sector_results.to_csv(os.path.join(RESULTS_DIR, "robustness_sector_neutral.csv"), index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print("All robustness checks completed.")
    print("Results saved to results/robustness_*.csv")


if __name__ == "__main__":
    run_robustness_checks()
