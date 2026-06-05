"""
Intraday Liquidity Regimes - Phase 3: Cross-Sectional Analysis
===============================================================
Analyzes determinants of regime membership using regression.

Model:
    P(Regime_k) = f(log(MarketCap), Volatility, Turnover, Sector)

Uses multinomial logit with year fixed effects.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try importing statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import MNLogit
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"


def load_regime_data():
    """Load regime assignments with features."""
    path = os.path.join(RESULTS_DIR, "regime_assignments.csv")
    return pd.read_csv(path)


def construct_explanatory_variables(df):
    """
    Construct explanatory variables for regime membership.
    Uses available features as proxies:
    - volume_entropy ~ log(market cap) proxy (larger caps have flatter profiles)
    - u_shape_strength ~ volatility proxy
    - peak_concentration ~ turnover concentration
    """
    df = df.copy()
    
    # Create regime dummy (numeric)
    regime_mapping = {name: i for i, name in enumerate(df['regime_name'].unique())}
    df['regime_code'] = df['regime_name'].map(regime_mapping)
    
    # Log-transform relevant features
    df['log_entropy'] = np.log1p(df['volume_entropy'])
    
    # Year dummies for fixed effects
    df['year_fe'] = df['year'].astype(str)
    
    return df, regime_mapping


def run_multinomial_logit(df):
    """
    Run multinomial logit regression.
    """
    if not STATSMODELS_AVAILABLE:
        print("Statsmodels not available. Using simplified analysis.")
        return None
    
    # Prepare features
    feature_cols = ['open_intensity', 'close_intensity', 'u_shape_strength', 'volume_entropy']
    
    X = df[feature_cols].copy()
    X = sm.add_constant(X)
    
    y = df['regime_code']
    
    # Handle missing
    valid = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid]
    y = y[valid]
    
    try:
        model = MNLogit(y, X)
        result = model.fit(disp=False, maxiter=100)
        return result
    except Exception as e:
        print(f"MNLogit failed: {e}")
        return None


def analyze_regime_determinants_simple(df):
    """
    Simplified analysis: compare feature means across regimes.
    """
    feature_cols = ['open_intensity', 'close_intensity', 'u_shape_strength', 
                    'volume_entropy', 'peak_concentration', 'skewness']
    
    # Group means
    regime_means = df.groupby('regime_name')[feature_cols].mean()
    
    # ANOVA tests for each feature
    anova_results = []
    
    for col in feature_cols:
        groups = [group[col].dropna().values for name, group in df.groupby('regime_name')]
        groups = [g for g in groups if len(g) > 10]
        
        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results.append({
                'feature': col,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    return regime_means, pd.DataFrame(anova_results)


def analyze_cap_tier_regimes(df):
    """
    Create cap-tier proxy and analyze regime distribution.
    """
    # Use volume entropy as cap tier proxy (higher entropy = flatter profile = larger cap)
    df = df.copy()
    df['cap_tier'] = pd.qcut(df['volume_entropy'], q=3, 
                             labels=['Small Cap', 'Mid Cap', 'Large Cap'])
    
    # Cross-tabulation
    cross_tab = pd.crosstab(df['cap_tier'], df['regime_name'], normalize='index') * 100
    
    # Chi-squared test
    contingency = pd.crosstab(df['cap_tier'], df['regime_name'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    return cross_tab, chi2, p_value


def plot_regime_determinants(regime_means, save_path=None):
    """Plot regime characteristic comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    features = regime_means.columns[:6]
    colors = plt.cm.Set2(np.linspace(0, 1, len(regime_means)))
    
    for i, feature in enumerate(features):
        ax = axes[i]
        regime_means[feature].plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        ax.set_title(feature.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Regime Characteristics: Feature Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def run_determinants_analysis():
    """Run complete determinants analysis."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("REGIME DETERMINANTS ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_regime_data()
    print(f"Loaded {len(df)} observations")
    
    # 1. Simple analysis (always runs)
    print("\n1. Computing regime feature means...")
    regime_means, anova_results = analyze_regime_determinants_simple(df)
    
    print("\nMean Features by Regime:")
    print(regime_means.round(4))
    
    print("\nANOVA Results (feature differences across regimes):")
    print(anova_results)
    
    regime_means.to_csv(os.path.join(RESULTS_DIR, "regime_feature_means.csv"))
    anova_results.to_csv(os.path.join(RESULTS_DIR, "regime_anova_tests.csv"), index=False)
    
    # 2. Cap-tier analysis
    print("\n2. Analyzing cap-tier regime distribution...")
    cross_tab, chi2, p_value = analyze_cap_tier_regimes(df)
    
    print("\nRegime Distribution by Cap Tier (%):")
    print(cross_tab.round(1))
    print(f"\nChi-squared test: χ²={chi2:.2f}, p={p_value:.4f}")
    
    cross_tab.to_csv(os.path.join(RESULTS_DIR, "cap_tier_regime_distribution.csv"))
    
    # 3. Multinomial logit (if available)
    if STATSMODELS_AVAILABLE:
        print("\n3. Running multinomial logit regression...")
        df_reg, regime_mapping = construct_explanatory_variables(df)
        result = run_multinomial_logit(df_reg)
        
        if result:
            print("\nMultinomial Logit Summary (first 20 lines):")
            summary_str = str(result.summary())
            print('\n'.join(summary_str.split('\n')[:20]))
            
            # Save full summary
            with open(os.path.join(RESULTS_DIR, "mnlogit_summary.txt"), 'w') as f:
                f.write(summary_str)
    
    # 4. Plot
    print("\n4. Generating determinants plot...")
    plot_regime_determinants(regime_means, os.path.join(PLOTS_DIR, "regime_determinants.png"))
    
    print("\nDeterminants analysis complete!")


if __name__ == "__main__":
    run_determinants_analysis()
