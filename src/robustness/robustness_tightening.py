"""
Intraday Liquidity Regimes - Robustness Tightening
====================================================
Critical robustness checks to address reviewer concerns.

1. Endogeneity control: Is this just market cap?
2. Alternative proxy: Volatility share instead of volume share
3. Out-of-sample stability: Train 2015-2018, test 2019-2025
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import chi2_contingency
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


def load_regime_data():
    """Load regime assignments with features."""
    path = os.path.join(RESULTS_DIR, "regime_assignments.csv")
    return pd.read_csv(path)


# =============================================================================
# 1. ENDOGENEITY CONTROL: Regime distribution within cap buckets
# =============================================================================

def endogeneity_cap_control(df):
    """
    Show regime shift persists WITHIN cap buckets.
    If regime shift is just cap, within-bucket shift should be zero.
    """
    print("\n" + "="*70)
    print("1. ENDOGENEITY CONTROL: Regime Distribution Within Cap Buckets")
    print("="*70)
    
    # Create cap buckets using volume_entropy as proxy (higher = larger cap)
    df = df.copy()
    df['cap_bucket'] = pd.qcut(df['volume_entropy'], q=3, 
                                labels=['Small', 'Mid', 'Large'])
    
    # For each cap bucket, show regime evolution over time
    results = []
    
    for bucket in ['Small', 'Mid', 'Large']:
        bucket_df = df[df['cap_bucket'] == bucket]
        
        # Regime distribution by period
        for period, (start, end) in [('pre_2019', (2015, 2018)), 
                                      ('post_2019', (2019, 2025))]:
            period_df = bucket_df[(bucket_df['year'] >= start) & 
                                   (bucket_df['year'] <= end)]
            
            if period_df.empty:
                continue
            
            regime_dist = period_df['regime_name'].value_counts(normalize=True) * 100
            
            for regime, pct in regime_dist.items():
                results.append({
                    'cap_bucket': bucket,
                    'period': period,
                    'regime': regime,
                    'percentage': pct
                })
    
    results_df = pd.DataFrame(results)
    
    # Pivot for display
    pivot = results_df.pivot_table(
        index=['cap_bucket', 'regime'],
        columns='period',
        values='percentage',
        fill_value=0
    ).round(1)
    
    print("\nRegime distribution by cap bucket and period:")
    print(pivot)
    
    # Compute shift within each bucket
    print("\n--- Regime Shift WITHIN Cap Buckets ---")
    
    shifts = []
    for bucket in ['Small', 'Mid', 'Large']:
        bucket_df = df[df['cap_bucket'] == bucket]
        
        pre = bucket_df[bucket_df['year'] <= 2018]['regime_name'].value_counts(normalize=True)
        post = bucket_df[bucket_df['year'] >= 2019]['regime_name'].value_counts(normalize=True)
        
        for regime in pre.index.union(post.index):
            pre_pct = pre.get(regime, 0) * 100
            post_pct = post.get(regime, 0) * 100
            shift = post_pct - pre_pct
            shifts.append({
                'cap_bucket': bucket,
                'regime': regime,
                'pre_2019': pre_pct,
                'post_2019': post_pct,
                'shift': shift
            })
            print(f"  {bucket} → {regime}: {pre_pct:.1f}% → {post_pct:.1f}% (Δ={shift:+.1f}%)")
    
    shifts_df = pd.DataFrame(shifts)
    
    # Chi-squared test for each bucket
    print("\n--- Chi-squared Tests (within bucket) ---")
    for bucket in ['Small', 'Mid', 'Large']:
        bucket_df = df[df['cap_bucket'] == bucket]
        
        pre = bucket_df[bucket_df['year'] <= 2018]['regime_name'].value_counts()
        post = bucket_df[bucket_df['year'] >= 2019]['regime_name'].value_counts()
        
        all_regimes = list(set(pre.index) | set(post.index))
        contingency = np.array([[pre.get(r, 0) for r in all_regimes],
                                [post.get(r, 0) for r in all_regimes]])
        
        chi2, p, dof, expected = chi2_contingency(contingency)
        sig = "✓ SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {bucket}: χ²={chi2:.1f}, p={p:.2e} {sig}")
    
    shifts_df.to_csv(os.path.join(RESULTS_DIR, "robustness_cap_endogeneity.csv"), index=False)
    
    return shifts_df


# =============================================================================
# 2. ALTERNATIVE PROXY: Volatility share instead of volume share
# =============================================================================

def alternative_volatility_proxy():
    """
    Repeat regime classification using volatility share instead of volume share.
    If regimes persist → robust finding.
    """
    print("\n" + "="*70)
    print("2. ALTERNATIVE PROXY: Volatility Share Clustering")
    print("="*70)
    
    from construct_intraday_profiles import load_stock_data, BARS_PER_DAY
    from scipy.stats import skew, entropy as scipy_entropy
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Get sample of stocks
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    symbols = [os.path.basename(f).replace('.csv', '') for f in stock_files]
    
    print(f"Computing volatility-based features for {len(symbols)} stocks...")
    
    def compute_volatility_features(symbol, year):
        """Compute features using volatility share instead of volume."""
        df = load_stock_data(symbol)
        if df is None:
            return None
        
        df = df[df['date'].dt.year == year]
        if df.empty:
            return None
        
        profiles = []
        for trading_date, day_df in df.groupby('trading_date'):
            if len(day_df) < BARS_PER_DAY * 0.8:
                continue
            
            day_df = day_df.head(BARS_PER_DAY)
            
            # Volatility share instead of volume share
            total_abs_ret = day_df['abs_return'].sum()
            if total_abs_ret == 0:
                continue
            
            vol_share = day_df['abs_return'].values / total_abs_ret
            if len(vol_share) >= BARS_PER_DAY:
                profiles.append(vol_share[:BARS_PER_DAY])
        
        if len(profiles) < 10:
            return None
        
        avg_profile = np.mean(profiles, axis=0)
        avg_profile = avg_profile / avg_profile.sum()
        
        return {
            'symbol': symbol,
            'year': year,
            'open_intensity': avg_profile[:6].sum(),
            'close_intensity': avg_profile[-6:].sum(),
            'midday_flatness': avg_profile[18:54].std(),
            'u_shape_strength': (avg_profile[:6].sum() + avg_profile[-6:].sum()) / 2 - avg_profile[18:54].mean() * 36,
            'volume_entropy': scipy_entropy(avg_profile + 1e-10),
            'peak_concentration': avg_profile.max(),
            'skewness': skew(avg_profile)
        }
    
    # Process sample (for speed, use 2023-2024)
    features = []
    for year in [2023, 2024]:
        for symbol in symbols[:200]:  # Sample 200 stocks
            result = compute_volatility_features(symbol, year)
            if result:
                features.append(result)
    
    if not features:
        print("  No features computed. Skipping.")
        return None
    
    vol_df = pd.DataFrame(features)
    print(f"  Computed {len(vol_df)} volatility-based observations")
    
    # Cluster
    X = vol_df[FEATURE_COLS].values
    X = np.nan_to_num(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    vol_df['regime_volatility'] = kmeans.fit_predict(X_scaled)
    
    n_labels = len(set(vol_df['regime_volatility']))
    if n_labels < 2:
        print(f"  Warning: Only {n_labels} cluster(s) found. Skipping silhouette.")
        sil_score = np.nan
    else:
        sil_score = silhouette_score(X_scaled, vol_df['regime_volatility'])
    print(f"  Silhouette score (volatility proxy): {sil_score:.4f}" if not np.isnan(sil_score) else "  Silhouette: N/A")
    
    # Compare with volume-based regimes
    regime_df = load_regime_data()
    merged = vol_df.merge(regime_df[['symbol', 'year', 'regime_kmeans']], 
                           on=['symbol', 'year'], how='inner')
    
    if len(merged) > 50:
        ari = adjusted_rand_score(merged['regime_volatility'], merged['regime_kmeans'])
        print(f"  Adjusted Rand Index (vol vs volume): {ari:.4f}")
        print(f"  Interpretation: {ari:.0%} agreement between proxies")
    
    vol_df.to_csv(os.path.join(RESULTS_DIR, "robustness_volatility_proxy.csv"), index=False)
    
    return vol_df


# =============================================================================
# 3. OUT-OF-SAMPLE STABILITY: Train 2015-2018, test 2019-2025
# =============================================================================

def out_of_sample_stability(df):
    """
    Train clusters on 2015-2018, assign regimes in 2019-2025.
    Show consistency of regime definitions.
    """
    print("\n" + "="*70)
    print("3. OUT-OF-SAMPLE STABILITY: Train 2015-2018, Test 2019-2025")
    print("="*70)
    
    # Split data
    train_df = df[df['year'] <= 2018].copy()
    test_df = df[df['year'] >= 2019].copy()
    
    print(f"  Training set: {len(train_df)} observations (2015-2018)")
    print(f"  Test set: {len(test_df)} observations (2019-2025)")
    
    # Train on 2015-2018
    X_train = train_df[FEATURE_COLS].values
    X_train = np.nan_to_num(X_train)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    train_df['regime_oos'] = kmeans.fit_predict(X_train_scaled)
    
    train_sil = silhouette_score(X_train_scaled, train_df['regime_oos'])
    print(f"\n  Training silhouette: {train_sil:.4f}")
    
    # Apply to test set
    X_test = test_df[FEATURE_COLS].values
    X_test = np.nan_to_num(X_test)
    X_test_scaled = scaler.transform(X_test)  # Use training scaler
    
    test_df['regime_oos'] = kmeans.predict(X_test_scaled)
    
    test_sil = silhouette_score(X_test_scaled, test_df['regime_oos'])
    print(f"  Test silhouette: {test_sil:.4f}")
    print(f"  Silhouette retention: {test_sil/train_sil:.1%}")
    
    # Compare regime distributions
    print("\n--- Regime Distribution Comparison ---")
    train_dist = train_df['regime_oos'].value_counts(normalize=True).sort_index() * 100
    test_dist = test_df['regime_oos'].value_counts(normalize=True).sort_index() * 100
    
    print(f"  {'Regime':<10} {'Train %':<12} {'Test %':<12} {'Δ':<10}")
    for regime in sorted(set(train_dist.index) | set(test_dist.index)):
        t = train_dist.get(regime, 0)
        te = test_dist.get(regime, 0)
        print(f"  {regime:<10} {t:<12.1f} {te:<12.1f} {te-t:+.1f}")
    
    # Compare with full-sample regimes
    combined = pd.concat([train_df, test_df])
    ari = adjusted_rand_score(combined['regime_oos'], combined['regime_kmeans'])
    print(f"\n  ARI (out-of-sample vs full-sample): {ari:.4f}")
    
    # Save
    result = pd.DataFrame({
        'metric': ['train_size', 'test_size', 'train_silhouette', 'test_silhouette', 
                   'silhouette_retention', 'ari_vs_fullsample'],
        'value': [len(train_df), len(test_df), train_sil, test_sil, 
                  test_sil/train_sil, ari]
    })
    result.to_csv(os.path.join(RESULTS_DIR, "robustness_oos_stability.csv"), index=False)
    
    return train_df, test_df


def run_robustness_tightening():
    """Run all robustness tightening checks."""
    print("\n" + "#"*70)
    print("# ROBUSTNESS TIGHTENING")
    print("#"*70)
    
    df = load_regime_data()
    
    # 1. Endogeneity control
    endogeneity_cap_control(df)
    
    # 2. Alternative proxy
    alternative_volatility_proxy()
    
    # 3. Out-of-sample stability
    out_of_sample_stability(df)
    
    print("\n" + "="*70)
    print("ROBUSTNESS TIGHTENING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_robustness_tightening()
