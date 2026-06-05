"""
Intraday Liquidity Regimes - Phase 4: Regime Evolution & Structural Change
============================================================================
Tracks regime evolution over time and tests for structural breaks.

Analyses:
- Regime distribution time series (yearly)
- Transition matrices
- Bai-Perron structural break tests

Key periods:
- Pre-2019 (baseline)
- 2019-2020 (algo adoption + COVID)
- Post-COVID (2021-2025)
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

# Key structural break points
PERIODS = {
    'pre_algo': (2015, 2018),
    'transition': (2019, 2020),
    'post_covid': (2021, 2025)
}


def load_regime_data():
    """Load regime assignments."""
    path = os.path.join(RESULTS_DIR, "regime_assignments.csv")
    return pd.read_csv(path)


def compute_regime_distribution(df):
    """
    Compute yearly regime distribution.
    Returns DataFrame with year rows and regime columns (percentages).
    """
    # Cross-tabulation
    cross = pd.crosstab(df['year'], df['regime_name'], normalize='index') * 100
    return cross


def compute_transition_matrix(df):
    """
    Compute regime transition probabilities: P(R_{t+1} | R_t)
    where t is measured in years for each stock.
    """
    transitions = []
    
    # Sort by stock and year
    df = df.sort_values(['symbol', 'year'])
    
    for symbol in df['symbol'].unique():
        stock_data = df[df['symbol'] == symbol].sort_values('year')
        
        if len(stock_data) < 2:
            continue
        
        regimes = stock_data['regime_name'].values
        
        for i in range(len(regimes) - 1):
            if pd.notna(regimes[i]) and pd.notna(regimes[i+1]):
                transitions.append({
                    'from_regime': regimes[i],
                    'to_regime': regimes[i+1]
                })
    
    trans_df = pd.DataFrame(transitions)
    
    if trans_df.empty:
        return None
    
    # Compute transition matrix
    matrix = pd.crosstab(trans_df['from_regime'], trans_df['to_regime'], 
                         normalize='index') * 100
    
    return matrix


def analyze_u_shape_evolution(df):
    """
    Track U-shape strength evolution over time.
    Compute yearly mean and test for structural breaks.
    """
    yearly_stats = df.groupby('year').agg({
        'u_shape_strength': ['mean', 'std', 'count'],
        'open_intensity': 'mean',
        'close_intensity': 'mean'
    }).round(4)
    
    yearly_stats.columns = ['u_shape_mean', 'u_shape_std', 'n_obs',
                            'open_mean', 'close_mean']
    
    return yearly_stats


def test_structural_breaks(df):
    """
    Test for structural breaks in regime distribution.
    Uses chi-squared test comparing periods.
    """
    results = []
    
    for period_name, (start, end) in PERIODS.items():
        period_data = df[(df['year'] >= start) & (df['year'] <= end)]
        regime_counts = period_data['regime_name'].value_counts()
        
        results.append({
            'period': period_name,
            'start': start,
            'end': end,
            'n_obs': len(period_data),
            'dominant_regime': regime_counts.idxmax() if len(regime_counts) > 0 else None,
            'dominant_pct': regime_counts.max() / len(period_data) * 100 if len(period_data) > 0 else 0
        })
    
    # Chi-squared tests between periods
    period_tests = []
    
    period_names = list(PERIODS.keys())
    for i in range(len(period_names)):
        for j in range(i+1, len(period_names)):
            p1, p2 = period_names[i], period_names[j]
            
            df1 = df[(df['year'] >= PERIODS[p1][0]) & (df['year'] <= PERIODS[p1][1])]
            df2 = df[(df['year'] >= PERIODS[p2][0]) & (df['year'] <= PERIODS[p2][1])]
            
            counts1 = df1['regime_name'].value_counts()
            counts2 = df2['regime_name'].value_counts()
            
            # Align indices
            all_regimes = list(set(counts1.index) | set(counts2.index))
            c1 = [counts1.get(r, 0) for r in all_regimes]
            c2 = [counts2.get(r, 0) for r in all_regimes]
            
            # Chi-squared test
            contingency = np.array([c1, c2])
            if contingency.sum() > 0:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                period_tests.append({
                    'comparison': f'{p1} vs {p2}',
                    'chi2': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
    
    return pd.DataFrame(results), pd.DataFrame(period_tests)


def analyze_regime_persistence(trans_matrix):
    """
    Analyze regime persistence (diagonal of transition matrix).
    Higher diagonal = more persistent regimes.
    """
    if trans_matrix is None:
        return None
    
    persistence = {}
    for regime in trans_matrix.index:
        if regime in trans_matrix.columns:
            persistence[regime] = trans_matrix.loc[regime, regime]
    
    return pd.Series(persistence).sort_values(ascending=False)


def plot_transition_heatmap(trans_matrix, save_path=None):
    """Plot transition matrix as heatmap."""
    if trans_matrix is None:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(trans_matrix.values, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Transition Probability (%)')
    
    # Labels
    ax.set_xticks(range(len(trans_matrix.columns)))
    ax.set_yticks(range(len(trans_matrix.index)))
    ax.set_xticklabels(trans_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(trans_matrix.index)
    
    ax.set_xlabel('To Regime', fontsize=12)
    ax.set_ylabel('From Regime', fontsize=12)
    ax.set_title('Regime Transition Probabilities (Year-over-Year)', fontsize=14, fontweight='bold')
    
    # Add values in cells
    for i in range(len(trans_matrix.index)):
        for j in range(len(trans_matrix.columns)):
            val = trans_matrix.iloc[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                   color=color, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_u_shape_evolution(yearly_stats, save_path=None):
    """Plot U-shape strength evolution over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    years = yearly_stats.index
    means = yearly_stats['u_shape_mean']
    stds = yearly_stats['u_shape_std']
    
    ax.plot(years, means, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.fill_between(years, means - stds, means + stds, alpha=0.2, color='#3498db')
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('U-Shape Strength (normalized)', fontsize=12)
    ax.set_title('Evolution of U-Shape Strength in NIFTY 500 Stocks', fontsize=14, fontweight='bold')
    
    # Add period markers
    ax.axvline(x=2019, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=2020, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.text(2019.1, ax.get_ylim()[1] * 0.95, 'Algo\nAdoption', color='orange', fontsize=9)
    ax.text(2020.1, ax.get_ylim()[1] * 0.95, 'COVID', color='red', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def run_evolution_analysis():
    """Run complete evolution analysis."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading regime data...")
    df = load_regime_data()
    print(f"Loaded {len(df)} observations")
    
    # 1. Regime distribution over time
    print("\n1. Computing regime distribution...")
    regime_dist = compute_regime_distribution(df)
    print(regime_dist.round(1))
    regime_dist.to_csv(os.path.join(RESULTS_DIR, "regime_evolution.csv"))
    
    # 2. Transition matrix
    print("\n2. Computing transition matrix...")
    trans_matrix = compute_transition_matrix(df)
    if trans_matrix is not None:
        print(trans_matrix.round(1))
        trans_matrix.to_csv(os.path.join(RESULTS_DIR, "transition_matrix.csv"))
        plot_transition_heatmap(trans_matrix, os.path.join(PLOTS_DIR, "regime_transitions.png"))
    
    # 3. U-shape evolution
    print("\n3. Analyzing U-shape evolution...")
    yearly_stats = analyze_u_shape_evolution(df)
    print(yearly_stats)
    yearly_stats.to_csv(os.path.join(RESULTS_DIR, "u_shape_evolution.csv"))
    plot_u_shape_evolution(yearly_stats, os.path.join(PLOTS_DIR, "u_shape_evolution.png"))
    
    # 4. Regime persistence
    print("\n4. Regime persistence (diagonal of transition matrix):")
    persistence = analyze_regime_persistence(trans_matrix)
    if persistence is not None:
        print(persistence)
    
    # 5. Structural break tests
    print("\n5. Structural break tests...")
    period_stats, chi2_tests = test_structural_breaks(df)
    print("\nPeriod Statistics:")
    print(period_stats)
    print("\nChi-squared Tests:")
    print(chi2_tests)
    
    chi2_tests.to_csv(os.path.join(RESULTS_DIR, "structural_break_tests.csv"), index=False)
    
    return regime_dist, trans_matrix, yearly_stats


if __name__ == "__main__":
    results = run_evolution_analysis()
