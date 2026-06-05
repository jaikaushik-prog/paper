"""
Intraday Liquidity Regimes - Phase 5: Stress & Event Analysis
===============================================================
Analyzes how liquidity regimes behave during market stress.

Analyses:
- Define stress days (top 5% market volatility)
- Compare regime behavior: stress vs normal
- Test regime fragility during stress

Hypotheses tested:
- Does regime switching accelerate during stress?
- Does liquidity concentrate more at open/close?
- Are small-cap regimes more fragile?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"

# Stress threshold
STRESS_PERCENTILE = 95  # Top 5% volatility days


def compute_market_volatility():
    """
    Compute market-wide daily volatility using large-cap stocks.
    Uses realized volatility of top 10 liquid stocks.
    """
    # Use major indices/stocks as proxy
    benchmark_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
                        'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC']
    
    daily_volatility = {}
    
    for symbol in benchmark_stocks:
        try:
            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df['trading_date'] = df['date'].dt.date
            
            # Compute 5-min returns
            df['return'] = df.groupby('trading_date')['close'].pct_change()
            
            # Daily realized volatility
            daily_vol = df.groupby('trading_date')['return'].std() * np.sqrt(75)  # Annualize within day
            
            for date, vol in daily_vol.items():
                if date not in daily_volatility:
                    daily_volatility[date] = []
                daily_volatility[date].append(vol)
        except Exception as e:
            print(f"  Skipped {symbol}: {e}")
    
    # Average across stocks
    market_vol = {date: np.mean(vols) for date, vols in daily_volatility.items() 
                  if len(vols) >= 5}  # Require at least 5 stocks
    
    vol_df = pd.DataFrame(list(market_vol.items()), columns=['date', 'volatility'])
    vol_df['date'] = pd.to_datetime(vol_df['date'])
    
    return vol_df


def classify_stress_days(vol_df, percentile=STRESS_PERCENTILE):
    """
    Classify days as STRESS (top percentile) or NORMAL.
    """
    threshold = vol_df['volatility'].quantile(percentile / 100)
    vol_df['is_stress'] = vol_df['volatility'] >= threshold
    vol_df['regime'] = np.where(vol_df['is_stress'], 'STRESS', 'NORMAL')
    
    print(f"Stress threshold (p{percentile}): {threshold:.4f}")
    print(f"Stress days: {vol_df['is_stress'].sum()} / {len(vol_df)} ({100*vol_df['is_stress'].mean():.1f}%)")
    
    return vol_df, threshold


def analyze_profile_differences(stress_vol_df):
    """
    Compare intraday profiles between stress and normal days.
    """
    from construct_intraday_profiles import load_stock_data, normalize_intraday_profile, BARS_PER_DAY
    
    # Sample stocks
    sample_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    stress_dates = set(stress_vol_df[stress_vol_df['is_stress']]['date'].dt.date)
    normal_dates = set(stress_vol_df[~stress_vol_df['is_stress']]['date'].dt.date)
    
    stress_profiles = []
    normal_profiles = []
    
    for symbol in sample_stocks:
        df = load_stock_data(symbol)
        if df is None:
            continue
        
        for trading_date, day_df in df.groupby('trading_date'):
            profile = normalize_intraday_profile(day_df)
            if profile is None or len(profile['vol_share']) < BARS_PER_DAY:
                continue
            
            if trading_date in stress_dates:
                stress_profiles.append(profile['vol_share'][:BARS_PER_DAY])
            elif trading_date in normal_dates:
                normal_profiles.append(profile['vol_share'][:BARS_PER_DAY])
    
    # Average profiles
    avg_stress = np.mean(stress_profiles, axis=0) if stress_profiles else None
    avg_normal = np.mean(normal_profiles, axis=0) if normal_profiles else None
    
    return avg_stress, avg_normal


def compute_concentration_metrics(avg_stress, avg_normal):
    """
    Compare concentration metrics between stress and normal.
    """
    if avg_stress is None or avg_normal is None:
        return None
    
    # Normalize
    avg_stress = avg_stress / avg_stress.sum()
    avg_normal = avg_normal / avg_normal.sum()
    
    metrics = {
        'stress_open_intensity': avg_stress[:6].sum(),
        'normal_open_intensity': avg_normal[:6].sum(),
        'stress_close_intensity': avg_stress[-6:].sum(),
        'normal_close_intensity': avg_normal[-6:].sum(),
        'stress_peak': avg_stress.max(),
        'normal_peak': avg_normal.max()
    }
    
    # Compute differences
    metrics['open_diff'] = metrics['stress_open_intensity'] - metrics['normal_open_intensity']
    metrics['close_diff'] = metrics['stress_close_intensity'] - metrics['normal_close_intensity']
    metrics['concentration_increase'] = (
        (metrics['stress_open_intensity'] + metrics['stress_close_intensity']) /
        (metrics['normal_open_intensity'] + metrics['normal_close_intensity']) - 1
    ) * 100
    
    return metrics


def test_regime_fragility(regime_df, stress_vol_df):
    """
    Test if regime assignments are more volatile during stress periods.
    """
    # This requires daily regime assignments, which we don't have
    # Instead, test at yearly level: do years with more stress days have different regimes?
    
    stress_vol_df['year'] = stress_vol_df['date'].dt.year
    yearly_stress = stress_vol_df.groupby('year')['is_stress'].mean() * 100
    
    # Merge with regime data
    regime_summary = regime_df.groupby('year').agg({
        'u_shape_strength': 'mean',
        'open_intensity': 'mean',
        'close_intensity': 'mean'
    })
    
    merged = regime_summary.join(yearly_stress.rename('stress_pct'))
    
    # Correlations
    correlations = {
        'u_shape_vs_stress': merged['u_shape_strength'].corr(merged['stress_pct']),
        'open_vs_stress': merged['open_intensity'].corr(merged['stress_pct']),
        'close_vs_stress': merged['close_intensity'].corr(merged['stress_pct'])
    }
    
    return merged, correlations


def plot_stress_comparison(avg_stress, avg_normal, save_path=None):
    """Plot stress vs normal day profiles."""
    from datetime import datetime, timedelta
    
    # Create time labels for x-axis
    base = datetime(2024, 1, 1, 9, 15)
    times = [base + timedelta(minutes=5*i) for i in range(75)]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if avg_stress is not None:
        stress_norm = avg_stress / avg_stress.sum() * 100
        ax.plot(times, stress_norm, label='STRESS Days (Top 5% Volatility)', 
                color='#e74c3c', linewidth=2.5)
    
    if avg_normal is not None:
        normal_norm = avg_normal / avg_normal.sum() * 100
        ax.plot(times, normal_norm, label='NORMAL Days', 
                color='#3498db', linewidth=2.5)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Volume Share (%)', fontsize=12)
    ax.set_title('Intraday Volume Profile: Stress vs Normal Days', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def run_stress_analysis():
    """Run complete stress analysis."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("STRESS DAY ANALYSIS")
    print("=" * 60)
    
    # 1. Compute market volatility
    print("\n1. Computing market volatility...")
    vol_df = compute_market_volatility()
    print(f"  Computed volatility for {len(vol_df)} trading days")
    
    # 2. Classify stress days
    print("\n2. Classifying stress days...")
    vol_df, threshold = classify_stress_days(vol_df)
    vol_df.to_csv(os.path.join(RESULTS_DIR, "market_stress_days.csv"), index=False)
    
    # 3. Compare profiles
    print("\n3. Comparing intraday profiles (stress vs normal)...")
    avg_stress, avg_normal = analyze_profile_differences(vol_df)
    
    # 4. Concentration metrics
    print("\n4. Computing concentration metrics...")
    metrics = compute_concentration_metrics(avg_stress, avg_normal)
    if metrics:
        print("\nConcentration Comparison:")
        print(f"  Open intensity: STRESS={metrics['stress_open_intensity']:.3f}, NORMAL={metrics['normal_open_intensity']:.3f}")
        print(f"  Close intensity: STRESS={metrics['stress_close_intensity']:.3f}, NORMAL={metrics['normal_close_intensity']:.3f}")
        print(f"  → Concentration increase during stress: {metrics['concentration_increase']:.1f}%")
        
        pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "stress_concentration_metrics.csv"), index=False)
    
    # 5. Plot comparison
    print("\n5. Generating stress comparison plot...")
    plot_stress_comparison(avg_stress, avg_normal, os.path.join(PLOTS_DIR, "stress_vs_normal.png"))
    
    # 6. Regime fragility test
    print("\n6. Testing regime fragility...")
    try:
        regime_df = pd.read_csv(os.path.join(RESULTS_DIR, "regime_assignments.csv"))
        merged, correlations = test_regime_fragility(regime_df, vol_df)
        print("\nCorrelations with yearly stress percentage:")
        for k, v in correlations.items():
            print(f"  {k}: {v:.3f}")
    except FileNotFoundError:
        print("  Regime assignments not found. Run clustering first.")
    
    print("\nStress analysis complete!")


if __name__ == "__main__":
    run_stress_analysis()
