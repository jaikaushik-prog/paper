"""
Directional FDI: Improved Feedback Dominance Index
====================================================

Improvements over standard FDI:
1. FDI_up: Feedback during up-moves
2. FDI_down: Feedback during down-moves  
3. FDI_asymmetry: Ratio of down/up feedback (>1 = crashes are feedback-driven)

Then compare predictive power vs standard FDI.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================

def load_intraday_data(data_dir='raw_Data'):
    """Load all intraday stock data."""
    print("📦 Loading intraday data...")
    
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    stock_data = {}
    for f in all_files:
        ticker = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, parse_dates=['date'])
            if len(df) > 1000:  # Minimum data requirement
                stock_data[ticker] = df
        except:
            continue
    
    print(f"   ✅ Loaded {len(stock_data)} stocks")
    return stock_data


def load_standard_fdi():
    """Load standard FDI output for comparison."""
    fdi = pd.read_csv('fdi_output.csv', parse_dates=['date'], index_col='date')
    return fdi


# =============================================================================
# DIRECTIONAL FDI COMPUTATION
# =============================================================================

def compute_directional_fdi(stock_data, window=20):
    """
    Compute Directional FDI:
    - FDI_up: Volatility/Illiquidity on up days
    - FDI_down: Volatility/Illiquidity on down days
    - FDI_asymmetry: FDI_down / FDI_up
    """
    print("\n📊 Computing Directional FDI...")
    
    # Aggregate to daily
    daily_data = []
    
    for ticker, df in stock_data.items():
        df = df.sort_values('date')
        
        # Daily aggregation
        daily = df.groupby(df['date'].dt.date).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Compute returns
        daily['return'] = daily['close'].pct_change()
        
        # Compute intraday volatility (Parkinson)
        daily['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(daily['high'] / daily['low']) ** 2)
        )
        
        # Compute Amihud illiquidity
        daily['amihud'] = np.abs(daily['return']) / (daily['volume'] * daily['close'] / 1e7 + 1e-10)
        
        # FDI = Volatility / Illiquidity
        daily['fdi'] = daily['parkinson_vol'] / (daily['amihud'] + 1e-10)
        
        # Direction flag
        daily['is_up'] = daily['return'] > 0
        daily['is_down'] = daily['return'] < 0
        
        daily['ticker'] = ticker
        daily_data.append(daily)
    
    all_daily = pd.concat(daily_data, ignore_index=True)
    
    # Aggregate across all stocks by date
    market_daily = all_daily.groupby('date').agg({
        'return': 'mean',
        'parkinson_vol': 'mean',
        'amihud': 'mean',
        'fdi': 'mean',
        'volume': 'sum'
    }).reset_index()
    market_daily = market_daily.set_index('date').sort_index()
    
    # Standard FDI (rolling)
    market_daily['FDI_standard'] = market_daily['fdi'].rolling(window).mean()
    market_daily['FDI_zscore'] = (
        (market_daily['FDI_standard'] - market_daily['FDI_standard'].rolling(252).mean()) / 
        market_daily['FDI_standard'].rolling(252).std()
    )
    
    # Directional FDI
    # Separate up and down days
    up_mask = market_daily['return'] > 0
    down_mask = market_daily['return'] < 0
    
    # Create masked series
    fdi_up_series = market_daily['fdi'].where(up_mask, np.nan)
    fdi_down_series = market_daily['fdi'].where(down_mask, np.nan)
    
    # Rolling mean with forward fill for missing
    market_daily['FDI_up'] = fdi_up_series.rolling(window, min_periods=5).mean().ffill()
    market_daily['FDI_down'] = fdi_down_series.rolling(window, min_periods=5).mean().ffill()
    
    # Asymmetry ratio
    market_daily['FDI_asymmetry'] = market_daily['FDI_down'] / (market_daily['FDI_up'] + 1e-10)
    
    # Z-scores for directional
    market_daily['FDI_up_zscore'] = (
        (market_daily['FDI_up'] - market_daily['FDI_up'].rolling(252).mean()) / 
        market_daily['FDI_up'].rolling(252).std()
    )
    market_daily['FDI_down_zscore'] = (
        (market_daily['FDI_down'] - market_daily['FDI_down'].rolling(252).mean()) / 
        market_daily['FDI_down'].rolling(252).std()
    )
    market_daily['FDI_asymmetry_zscore'] = (
        (market_daily['FDI_asymmetry'] - market_daily['FDI_asymmetry'].rolling(252).mean()) / 
        market_daily['FDI_asymmetry'].rolling(252).std()
    )
    
    print(f"   ✅ Computed directional FDI for {len(market_daily)} days")
    
    return market_daily


# =============================================================================
# COMPARISON: DIRECTIONAL VS STANDARD FDI
# =============================================================================

def compare_fdi_variants(directional_fdi, window=10, drawdown_threshold=0.03):
    """
    Compare predictive power of:
    - Standard FDI
    - FDI_up
    - FDI_down
    - FDI_asymmetry
    
    Metric: Hit rate for predicting drawdowns
    """
    print("\n" + "=" * 70)
    print("  COMPARING FDI VARIANTS: Crash Prediction Power")
    print("=" * 70)
    
    df = directional_fdi.copy()
    
    # Forward maximum drawdown
    df['fwd_return'] = df['return'].rolling(window).sum().shift(-window)
    df['fwd_min_return'] = df['return'].rolling(window).min().shift(-window)
    df['had_drawdown'] = df['fwd_min_return'] < -drawdown_threshold
    
    # Test each variant
    variants = [
        ('FDI_zscore', 'Standard FDI'),
        ('FDI_up_zscore', 'FDI (Up Days Only)'),
        ('FDI_down_zscore', 'FDI (Down Days Only)'),
        ('FDI_asymmetry_zscore', 'FDI Asymmetry (Down/Up)')
    ]
    
    results = []
    
    for col, name in variants:
        if col not in df.columns:
            continue
            
        df_clean = df[[col, 'had_drawdown', 'fwd_return']].dropna()
        
        if len(df_clean) < 100:
            continue
        
        # Test different thresholds
        for threshold in [1.0, 1.5, 2.0]:
            high_fdi = df_clean[df_clean[col] > threshold]
            
            if len(high_fdi) < 10:
                continue
            
            hit_rate = high_fdi['had_drawdown'].mean()
            avg_return = high_fdi['fwd_return'].mean()
            n_events = len(high_fdi)
            
            results.append({
                'variant': name,
                'threshold': threshold,
                'hit_rate': hit_rate,
                'avg_return': avg_return,
                'n_events': n_events
            })
    
    results_df = pd.DataFrame(results)
    
    # Display
    print(f"\n   Forward window: {window} days")
    print(f"   Drawdown threshold: {drawdown_threshold:.1%}")
    print(f"\n   {'Variant':<25} {'Thresh':>8} {'Hit Rate':>10} {'Avg Ret':>10} {'Events':>8}")
    print("   " + "-" * 65)
    
    for _, row in results_df.iterrows():
        print(f"   {row['variant']:<25} {row['threshold']:>8.1f} {row['hit_rate']:>9.1%} {row['avg_return']:>9.2%} {row['n_events']:>8}")
    
    # Find best
    if len(results_df) > 0:
        best = results_df.loc[results_df['hit_rate'].idxmax()]
        print(f"\n   ⭐ BEST: {best['variant']} @ {best['threshold']} → {best['hit_rate']:.1%} hit rate")
    
    return results_df


def statistical_comparison(directional_fdi):
    """
    Statistical tests: Does directional FDI outperform standard?
    """
    print("\n" + "=" * 70)
    print("  STATISTICAL COMPARISON")
    print("=" * 70)
    
    df = directional_fdi.copy()
    
    # Forward returns
    df['fwd_5d'] = df['return'].rolling(5).sum().shift(-5)
    df['fwd_10d'] = df['return'].rolling(10).sum().shift(-10)
    
    # Correlations with forward returns
    print("\n   📊 Correlation with Forward Returns:")
    print(f"\n   {'Metric':<25} {'5d Fwd':>10} {'10d Fwd':>10}")
    print("   " + "-" * 45)
    
    metrics = ['FDI_zscore', 'FDI_up_zscore', 'FDI_down_zscore', 'FDI_asymmetry_zscore']
    correlations = {}
    
    for m in metrics:
        if m in df.columns:
            corr_5d = df[m].corr(df['fwd_5d'])
            corr_10d = df[m].corr(df['fwd_10d'])
            correlations[m] = {'5d': corr_5d, '10d': corr_10d}
            print(f"   {m:<25} {corr_5d:>+10.3f} {corr_10d:>+10.3f}")
    
    # T-test: Is FDI_down correlation significantly different from FDI_standard?
    print("\n   📊 Testing if FDI_down beats Standard FDI:")
    
    df_clean = df[['FDI_zscore', 'FDI_down_zscore', 'fwd_10d']].dropna()
    
    # Bootstrap test
    n_bootstrap = 1000
    diff_corrs = []
    
    for _ in range(n_bootstrap):
        sample = df_clean.sample(frac=0.8, replace=True)
        corr_std = sample['FDI_zscore'].corr(sample['fwd_10d'])
        corr_down = sample['FDI_down_zscore'].corr(sample['fwd_10d'])
        diff_corrs.append(corr_down - corr_std)
    
    diff_mean = np.mean(diff_corrs)
    diff_std = np.std(diff_corrs)
    p_value = 2 * min(
        np.mean(np.array(diff_corrs) > 0),
        np.mean(np.array(diff_corrs) < 0)
    )
    
    print(f"      Correlation difference (Down - Standard): {diff_mean:+.4f} ± {diff_std:.4f}")
    print(f"      p-value (bootstrap): {p_value:.4f}")
    
    if diff_mean < 0 and p_value < 0.10:
        print(f"      ✅ FDI_down has STRONGER negative correlation (better predictor)")
    elif diff_mean > 0 and p_value < 0.10:
        print(f"      ❌ Standard FDI is better")
    else:
        print(f"      🟡 No significant difference")
    
    return correlations


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_directional_fdi(directional_fdi):
    """Create visualization of directional FDI."""
    print("\n📊 Creating visualizations...")
    
    df = directional_fdi.dropna(subset=['FDI_zscore', 'FDI_up_zscore', 'FDI_down_zscore'])
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # 1. Standard vs Directional FDI
    ax1 = axes[0]
    ax1.plot(df.index, df['FDI_zscore'], 'b-', alpha=0.7, label='Standard FDI', linewidth=1)
    ax1.plot(df.index, df['FDI_down_zscore'], 'r-', alpha=0.7, label='FDI (Down Days)', linewidth=1)
    ax1.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Z-Score')
    ax1.set_title('Standard FDI vs Directional FDI (Down Days)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. FDI Asymmetry
    ax2 = axes[1]
    colors = ['red' if x > 1 else 'green' for x in df['FDI_asymmetry']]
    ax2.fill_between(df.index, 1, df['FDI_asymmetry'], 
                     where=df['FDI_asymmetry'] > 1, color='red', alpha=0.5, label='More Down Feedback')
    ax2.fill_between(df.index, 1, df['FDI_asymmetry'], 
                     where=df['FDI_asymmetry'] < 1, color='green', alpha=0.5, label='More Up Feedback')
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Asymmetry (Down/Up)')
    ax2.set_title('FDI Asymmetry: Crash Feedback vs Rally Feedback', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative returns
    ax3 = axes[2]
    cumret = (1 + df['return']).cumprod()
    ax3.plot(df.index, cumret, 'k-', linewidth=1)
    
    # Highlight high FDI_down periods
    high_down_fdi = df['FDI_down_zscore'] > 1.5
    for i in range(len(df)):
        if high_down_fdi.iloc[i]:
            ax3.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                       alpha=0.3, color='red')
    
    ax3.set_ylabel('Cumulative Return')
    ax3.set_xlabel('Date')
    ax3.set_title('Market Returns (Red = High FDI_down)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/directional_fdi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/directional_fdi.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  DIRECTIONAL FDI: IMPROVED FEEDBACK DOMINANCE INDEX")
    print("=" * 70)
    
    # Load data
    stock_data = load_intraday_data()
    
    if len(stock_data) == 0:
        print("❌ No stock data found")
        return
    
    # Compute directional FDI
    directional_fdi = compute_directional_fdi(stock_data)
    
    # Save output
    directional_fdi.to_csv('directional_fdi_output.csv')
    print(f"\n   ✅ Saved: directional_fdi_output.csv")
    
    # Compare variants
    comparison = compare_fdi_variants(directional_fdi)
    
    # Statistical tests
    correlations = statistical_comparison(directional_fdi)
    
    # Visualize
    visualize_directional_fdi(directional_fdi)
    
    # Summary
    print("\n" + "=" * 70)
    print("  DIRECTIONAL FDI SUMMARY")
    print("=" * 70)
    
    print("\n   Key Metrics Computed:")
    print("   • FDI_up: Feedback on up-days only")
    print("   • FDI_down: Feedback on down-days only")
    print("   • FDI_asymmetry: Down/Up ratio (>1 = crashes are feedback-driven)")
    
    if len(comparison) > 0:
        best = comparison.loc[comparison['hit_rate'].idxmax()]
        standard_best = comparison[comparison['variant'] == 'Standard FDI']['hit_rate'].max()
        down_best = comparison[comparison['variant'] == 'FDI (Down Days Only)']['hit_rate'].max()
        
        print(f"\n   Results:")
        print(f"   • Standard FDI best hit rate: {standard_best:.1%}")
        print(f"   • FDI_down best hit rate: {down_best:.1%}")
        
        if down_best > standard_best:
            improvement = (down_best - standard_best) / standard_best * 100
            print(f"   ✅ FDI_down OUTPERFORMS by {improvement:.1f}%")
            return True, directional_fdi
        else:
            print(f"   🟡 No significant improvement")
            return False, directional_fdi
    
    return False, directional_fdi


if __name__ == "__main__":
    outperforms, fdi_data = main()
