"""
Intraday Liquidity Regimes - Phase 2b: Regime Visualization
=============================================================
Generates paper-ready visualizations of liquidity regimes.

Outputs:
- Average intraday curves per regime
- Cluster scatter plots (PCA)
- Regime distribution charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import os
from datetime import time, datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

FEATURE_COLS = [
    'open_intensity', 'close_intensity', 'midday_flatness',
    'u_shape_strength', 'volume_entropy', 'peak_concentration', 'skewness'
]

# Create time labels using datetime arithmetic
from datetime import datetime as dt_datetime
_base = dt_datetime(2024, 1, 1, 9, 15)
BAR_TIMES = [(_base + timedelta(minutes=5*i)).time() for i in range(75)]


def load_regime_assignments():
    """Load regime assignments."""
    path = os.path.join(RESULTS_DIR, "regime_assignments.csv")
    return pd.read_csv(path)


def get_regime_average_profiles(regime_df):
    """
    For each regime, compute the average intraday profile.
    Returns dict of regime_name -> avg_profile array (75 bars)
    """
    from construct_intraday_profiles import load_stock_data, normalize_intraday_profile, BARS_PER_DAY
    
    regime_profiles = {}
    
    for regime_name in regime_df['regime_name'].unique():
        if pd.isna(regime_name):
            continue
            
        regime_stocks = regime_df[regime_df['regime_name'] == regime_name]
        
        # Sample up to 20 stocks per regime for efficiency
        sample = regime_stocks.sample(min(20, len(regime_stocks)), random_state=42)
        
        all_profiles = []
        
        for _, row in sample.iterrows():
            symbol = row['symbol']
            year = row['year']
            
            df = load_stock_data(symbol)
            if df is None:
                continue
            
            df = df[df['date'].dt.year == year]
            
            for trading_date, day_df in df.groupby('trading_date'):
                profile = normalize_intraday_profile(day_df)
                if profile is not None and len(profile['vol_share']) >= BARS_PER_DAY:
                    all_profiles.append(profile['vol_share'][:BARS_PER_DAY])
        
        if all_profiles:
            avg_profile = np.mean(all_profiles, axis=0)
            regime_profiles[regime_name] = avg_profile / avg_profile.sum()
    
    return regime_profiles


def plot_regime_curves(regime_profiles, save_path=None):
    """Plot average intraday curves per regime."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # X-axis as time of day
    times = [datetime.combine(datetime.today(), t) for t in BAR_TIMES]
    
    for i, (regime_name, profile) in enumerate(regime_profiles.items()):
        ax.plot(times, profile * 100, label=regime_name, 
                color=COLORS[i % len(COLORS)], linewidth=2.5, alpha=0.85)
    
    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Volume Share (%)', fontsize=12)
    ax.set_title('Average Intraday Volume Profiles by Liquidity Regime', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotations for open/close
    ax.axvline(x=times[0], color='gray', linestyle='--', alpha=0.5, label='Market Open')
    ax.axvline(x=times[-1], color='gray', linestyle='--', alpha=0.5, label='Market Close')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_cluster_scatter(regime_df, save_path=None):
    """Plot PCA scatter of clusters."""
    # Prepare features
    X = regime_df[FEATURE_COLS].values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    regimes = regime_df['regime_name'].unique()
    
    for i, regime in enumerate(regimes):
        if pd.isna(regime):
            continue
        mask = regime_df['regime_name'] == regime
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=COLORS[i % len(COLORS)], label=regime,
                   alpha=0.6, s=30)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('Liquidity Regime Clusters (PCA Projection)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_regime_distribution(regime_df, save_path=None):
    """Plot regime distribution over years."""
    # Aggregate by year and regime
    year_regime = regime_df.groupby(['year', 'regime_name']).size().unstack(fill_value=0)
    
    # Convert to percentages
    year_regime_pct = year_regime.div(year_regime.sum(axis=1), axis=0) * 100
    
    # Stacked area chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    year_regime_pct.plot(kind='area', stacked=True, ax=ax, 
                         color=COLORS[:len(year_regime_pct.columns)],
                         alpha=0.8)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Regime Share (%)', fontsize=12)
    ax.set_title('Evolution of Intraday Liquidity Regimes (2015-2025)', fontsize=14, fontweight='bold')
    ax.legend(title='Regime', loc='upper left', fontsize=10)
    ax.set_xlim(year_regime_pct.index.min(), year_regime_pct.index.max())
    ax.set_ylim(0, 100)
    
    # Add COVID marker
    ax.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(2020.1, 95, 'COVID', color='red', fontsize=10, fontweight='bold')
    
    # Add 2019 algo adoption marker
    ax.axvline(x=2019, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(2019.1, 90, 'Algo Adoption', color='orange', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()
    
    return year_regime_pct


def plot_cap_tier_breakdown(regime_df, save_path=None):
    """Plot regime breakdown by market cap tier (proxy: open_intensity as inverse of cap)."""
    # Create cap tier proxy based on volume entropy (higher = larger cap typically)
    regime_df = regime_df.copy()
    regime_df['cap_tier'] = pd.qcut(regime_df['volume_entropy'], q=3, 
                                     labels=['Small Cap', 'Mid Cap', 'Large Cap'])
    
    # Cross-tabulation
    cap_regime = pd.crosstab(regime_df['cap_tier'], regime_df['regime_name'], normalize='index') * 100
    
    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    cap_regime.plot(kind='bar', ax=ax, color=COLORS[:len(cap_regime.columns)], 
                    edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Market Cap Tier', fontsize=12)
    ax.set_ylabel('Regime Share (%)', fontsize=12)
    ax.set_title('Liquidity Regime Distribution by Market Cap Tier', fontsize=14, fontweight='bold')
    ax.legend(title='Regime', loc='upper right', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def generate_all_visualizations():
    """Generate all regime visualizations."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    print("Loading regime assignments...")
    regime_df = load_regime_assignments()
    
    print(f"Loaded {len(regime_df)} observations across {regime_df['regime_name'].nunique()} regimes")
    
    # 1. Cluster scatter
    print("\n1. Generating cluster scatter plot...")
    plot_cluster_scatter(regime_df, os.path.join(PLOTS_DIR, 'regime_clusters_pca.png'))
    
    # 2. Regime distribution over time
    print("\n2. Generating regime evolution plot...")
    year_regime = plot_regime_distribution(regime_df, os.path.join(PLOTS_DIR, 'regime_evolution.png'))
    
    # 3. Cap tier breakdown
    print("\n3. Generating cap-tier breakdown...")
    plot_cap_tier_breakdown(regime_df, os.path.join(PLOTS_DIR, 'regime_by_cap_tier.png'))
    
    # 4. Average intraday curves (slower - samples profiles)
    print("\n4. Generating average regime curves (this may take a few minutes)...")
    try:
        regime_profiles = get_regime_average_profiles(regime_df)
        plot_regime_curves(regime_profiles, os.path.join(PLOTS_DIR, 'regime_curves.png'))
    except Exception as e:
        print(f"  Skipped regime curves: {e}")
    
    print("\nAll visualizations generated!")
    
    return year_regime


if __name__ == "__main__":
    generate_all_visualizations()
