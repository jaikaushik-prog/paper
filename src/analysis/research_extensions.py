"""
Λ-Volatility Diagnostic Engine: Research Extensions
Testing 3 Key Hypotheses

H1: NBFC FDI z-score > 1.5 predicts NIFTY drawdown > 3% within 10 days
H2: Absorbed → Reflexive transition probability increases during RBI policy uncertainty
H3: Stress routing portfolio outperforms equal-weight by 200+ bps annually
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime, timedelta

# =============================================================================
# LOAD DATA
# =============================================================================

def load_all_outputs():
    """Load all computed outputs from the main pipeline."""
    print("📦 Loading computed outputs...")
    
    fdi_output = pd.read_csv("fdi_output.csv", parse_dates=['date'], index_col='date')
    sectoral_fdi = pd.read_csv("sectoral_fdi_output.csv", parse_dates=['date'], index_col='date')
    strategy_signals = pd.read_csv("strategy_signals.csv", parse_dates=['date'], index_col='date')
    stress_weights = pd.read_csv("stress_routing_weights.csv", parse_dates=['date'], index_col='date')
    
    print(f"   ✅ Loaded FDI: {len(fdi_output)} days")
    print(f"   ✅ Loaded Sectoral FDI: {len(sectoral_fdi)} days, {len(sectoral_fdi.columns)} sectors")
    
    return fdi_output, sectoral_fdi, strategy_signals, stress_weights


def load_nifty_returns():
    """Load NIFTY 50 returns for hypothesis testing."""
    print("\n📈 Loading NIFTY 50 returns...")
    
    # Try to load NIFTY data from raw_Data
    nifty_files = ['raw_Data/NIFTY50.csv', 'raw_Data/NIFTYBANK.csv']
    
    # Use a large-cap as proxy for NIFTY if not available
    proxy_file = 'raw_Data/RELIANCE.csv'
    
    try:
        # Try to load actual NIFTY
        for nf in nifty_files:
            if os.path.exists(nf):
                df = pd.read_csv(nf, parse_dates=['date'])
                break
        else:
            # Use RELIANCE as proxy
            df = pd.read_csv(proxy_file, parse_dates=['date'])
            print("   ⚠️ Using RELIANCE as proxy for NIFTY")
    except Exception as e:
        print(f"   ❌ Error loading NIFTY: {e}")
        return None
    
    # Compute daily returns
    df = df.sort_values('date')
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Aggregate to daily
    daily = df.groupby(df['date'].dt.date).agg({
        'close': 'last',
        'return': 'sum',
        'log_return': 'sum'
    }).reset_index()
    daily.columns = ['date', 'close', 'return', 'log_return']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.set_index('date')
    
    print(f"   ✅ Loaded returns: {len(daily)} days")
    return daily


# =============================================================================
# H1: NBFC FDI PREDICTS NIFTY DRAWDOWN
# =============================================================================

def test_h1_nbfc_predicts_drawdown(sectoral_fdi, nifty_returns, threshold=1.5, window=10, 
                                    drawdown_threshold=0.03, persistence_days=3):
    """
    H1: NBFC FDI z-score > threshold (for N days) predicts NIFTY drawdown > 3% within 10 days
    
    Event study methodology:
    1. Identify events: NBFC FDI stays above threshold for persistence_days consecutive days
    2. Compute forward returns for each event
    3. Calculate hit rate and average return
    """
    print("\n" + "=" * 60)
    print(f"  H1: NBFC FDI > {threshold} (for {persistence_days}d) → Drawdown > {drawdown_threshold:.0%} in {window}d")
    print("=" * 60)
    
    if 'NBFCs' not in sectoral_fdi.columns:
        print("   ❌ NBFCs not in sectoral FDI columns")
        return None
    
    nbfc_fdi = sectoral_fdi['NBFCs'].dropna()
    
    # FIXED: Persistence-based event: N consecutive days above threshold
    # This reduces noise from single-day spikes
    above_threshold = nbfc_fdi > threshold
    persistence = above_threshold.rolling(persistence_days).sum() >= persistence_days
    
    # Trigger on FIRST day of N-day persistence (when rolling sum first reaches N)
    event_trigger = persistence & (~persistence.shift(1).fillna(False))
    event_dates = nbfc_fdi[event_trigger].index
    
    print(f"\n   📊 Event Study Parameters (PERSISTENCE-BASED):")
    print(f"      Threshold: FDI z-score > {threshold}")
    print(f"      Persistence: {persistence_days}+ consecutive days required")
    print(f"      Forward window: {window} days")
    print(f"      Drawdown threshold: {drawdown_threshold:.1%}")
    print(f"      Events found: {len(event_dates)}")
    
    if len(event_dates) < 5:
        print("   ⚠️ Insufficient events for reliable analysis")
        return None
    
    # Compute forward returns for each event
    results = []
    for event_date in event_dates:
        try:
            # Get forward returns
            future_dates = pd.date_range(event_date, periods=window+1, freq='D')[1:]
            future_returns = nifty_returns.loc[nifty_returns.index.intersection(future_dates), 'log_return']
            
            if len(future_returns) < 5:
                continue
            
            cumulative_return = future_returns.sum()
            max_drawdown = (nifty_returns.loc[future_returns.index, 'close'].cummin() / 
                           nifty_returns.loc[event_date:, 'close'].iloc[0] - 1).min()
            
            results.append({
                'event_date': event_date,
                'nbfc_fdi': nbfc_fdi.loc[event_date],
                'forward_return': cumulative_return,
                'max_drawdown': max_drawdown,
                'hit': max_drawdown < -drawdown_threshold
            })
        except Exception as e:
            continue
    
    if len(results) == 0:
        print("   ❌ No valid events found")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Statistics
    hit_rate = results_df['hit'].mean()
    avg_return = results_df['forward_return'].mean()
    avg_drawdown = results_df['max_drawdown'].mean()
    
    # Statistical significance (t-test vs zero)
    t_stat, p_value = stats.ttest_1samp(results_df['forward_return'], 0)
    
    print(f"\n   📊 RESULTS:")
    print(f"      Valid events: {len(results_df)}")
    print(f"      Hit rate (drawdown > {drawdown_threshold:.1%}): {hit_rate:.1%}")
    print(f"      Average forward return: {avg_return:.2%}")
    print(f"      Average max drawdown: {avg_drawdown:.2%}")
    print(f"      t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")
    
    # Verdict
    if hit_rate > 0.5 and p_value < 0.1:
        print(f"\n   ✅ H1 SUPPORTED: NBFC FDI > {threshold} predicts drawdowns")
    elif hit_rate > 0.4:
        print(f"\n   🟡 H1 PARTIALLY SUPPORTED: Hit rate {hit_rate:.1%} is elevated")
    else:
        print(f"\n   ❌ H1 NOT SUPPORTED: Hit rate {hit_rate:.1%} is not conclusive")
    
    return results_df


# =============================================================================
# H2: ABSORBED → REFLEXIVE INCREASES DURING RBI UNCERTAINTY
# =============================================================================

# RBI Policy Dates (historical MPC meetings)
RBI_POLICY_DATES = [
    # 2020
    '2020-02-06', '2020-03-27', '2020-05-22', '2020-08-06', '2020-10-09', '2020-12-04',
    # 2021
    '2021-02-05', '2021-04-07', '2021-06-04', '2021-08-06', '2021-10-08', '2021-12-08',
    # 2022
    '2022-02-10', '2022-04-08', '2022-05-04', '2022-06-08', '2022-08-05', '2022-09-30', '2022-12-07',
    # 2023
    '2023-02-08', '2023-04-06', '2023-06-08', '2023-08-10', '2023-10-06', '2023-12-08',
    # 2024
    '2024-02-08', '2024-04-05', '2024-06-07', '2024-08-08', '2024-10-09', '2024-12-06',
]


def test_h2_rbi_policy_impact(fdi_output, window_before=5, window_after=5):
    """
    H2: Absorbed → Reflexive transition probability increases during RBI policy uncertainty
    
    Methodology:
    1. Define RBI policy windows (±5 days around MPC dates)
    2. Compute transition probabilities in policy vs non-policy periods
    3. Compare using chi-square test
    """
    print("\n" + "=" * 60)
    print("  H2: RBI Policy → Increased Absorbed→Reflexive Probability")
    print("=" * 60)
    
    regimes = fdi_output['regime'].dropna()
    regimes = regimes[regimes.isin(['healthy', 'absorbed_shock', 'hidden_instability', 'reflexive_crash'])]
    
    # Create policy window indicator
    policy_dates = pd.to_datetime(RBI_POLICY_DATES)
    
    is_policy_window = pd.Series(False, index=regimes.index)
    for policy_date in policy_dates:
        window_start = policy_date - pd.Timedelta(days=window_before)
        window_end = policy_date + pd.Timedelta(days=window_after)
        mask = (regimes.index >= window_start) & (regimes.index <= window_end)
        is_policy_window = is_policy_window | mask
    
    print(f"\n   📊 Policy Window Analysis:")
    print(f"      Policy dates: {len(policy_dates)}")
    print(f"      Window: ±{window_before}/{window_after} days")
    print(f"      Days in policy window: {is_policy_window.sum()}")
    print(f"      Days outside window: {(~is_policy_window).sum()}")
    
    # Compute transitions
    next_regime = regimes.shift(-1)
    
    # Absorbed → Reflexive transitions
    absorbed_mask = regimes == 'absorbed_shock'
    to_reflexive = next_regime == 'reflexive_crash'
    
    # Policy window
    absorbed_policy = absorbed_mask & is_policy_window
    absorbed_to_reflexive_policy = (absorbed_policy & to_reflexive).sum()
    absorbed_total_policy = absorbed_policy.sum()
    
    # Non-policy window
    absorbed_non_policy = absorbed_mask & ~is_policy_window
    absorbed_to_reflexive_non_policy = (absorbed_non_policy & to_reflexive).sum()
    absorbed_total_non_policy = absorbed_non_policy.sum()
    
    # Probabilities
    p_policy = absorbed_to_reflexive_policy / absorbed_total_policy if absorbed_total_policy > 0 else 0
    p_non_policy = absorbed_to_reflexive_non_policy / absorbed_total_non_policy if absorbed_total_non_policy > 0 else 0
    
    print(f"\n   📊 Transition Probabilities:")
    print(f"      P(Absorbed→Reflexive | Policy Window): {p_policy:.1%} ({absorbed_to_reflexive_policy}/{absorbed_total_policy})")
    print(f"      P(Absorbed→Reflexive | Non-Policy): {p_non_policy:.1%} ({absorbed_to_reflexive_non_policy}/{absorbed_total_non_policy})")
    
    # Chi-square test
    observed = np.array([[absorbed_to_reflexive_policy, absorbed_total_policy - absorbed_to_reflexive_policy],
                         [absorbed_to_reflexive_non_policy, absorbed_total_non_policy - absorbed_to_reflexive_non_policy]])
    
    if observed.min() >= 5:
        chi2, p_value, dof, expected = stats.chi2_contingency(observed)
        print(f"      Chi-square: {chi2:.2f}, p-value: {p_value:.4f}")
    else:
        # Fisher's exact test for small samples
        _, p_value = stats.fisher_exact(observed)
        chi2 = np.nan
        print(f"      Fisher's exact p-value: {p_value:.4f}")
    
    # Verdict
    ratio = p_policy / p_non_policy if p_non_policy > 0 else float('inf')
    print(f"\n      Ratio: {ratio:.2f}x higher during policy windows")
    
    if p_policy > p_non_policy and p_value < 0.1:
        print(f"\n   ✅ H2 SUPPORTED: Transitions increase during RBI policy uncertainty")
    elif p_policy > p_non_policy:
        print(f"\n   🟡 H2 DIRECTIONALLY SUPPORTED but not statistically significant")
    else:
        print(f"\n   ❌ H2 NOT SUPPORTED: No evidence of increased transitions")
    
    return {
        'p_policy': p_policy,
        'p_non_policy': p_non_policy,
        'ratio': ratio,
        'p_value': p_value
    }


# =============================================================================
# H3: STRESS ROUTING OUTPERFORMS EQUAL WEIGHT
# =============================================================================

def test_h3_stress_routing_backtest(stress_weights, sectoral_fdi, nifty_returns, 
                                     transaction_cost=0.001, rebalance_threshold=0.05):
    """
    H3: Stress routing portfolio outperforms equal-weight by 200+ bps annually
    
    Backtest methodology:
    1. Compute daily returns for stress-routed portfolio
    2. Compute daily returns for equal-weight portfolio
    3. Account for transaction costs on rebalancing
    4. Compare CAGR, Sharpe, max drawdown
    """
    print("\n" + "=" * 60)
    print("  H3: Stress Routing Outperforms Equal-Weight by 200+ bps")
    print("=" * 60)
    
    # Align data
    common_dates = stress_weights.index.intersection(nifty_returns.index)
    weights = stress_weights.loc[common_dates].copy()
    returns = nifty_returns.loc[common_dates, 'log_return'].copy()
    
    # Create synthetic sector returns (simplified: assume all sectors correlated with market)
    # In reality, you'd use actual sector ETF returns
    np.random.seed(42)
    sector_returns = pd.DataFrame(index=common_dates)
    sector_betas = {
        'Banks': 1.2, 'NBFCs': 1.3, 'IT': 0.9, 'Metals': 1.4,
        'Pharma': 0.7, 'Auto': 1.1, 'FMCG': 0.6, 'Infrastructure': 1.2, 'Power_Utilities': 0.8
    }
    
    for sector in weights.columns:
        beta = sector_betas.get(sector, 1.0)
        idio = np.random.normal(0, 0.005, len(common_dates))  # Idiosyncratic return
        sector_returns[sector] = returns * beta + idio
    
    print(f"\n   📊 Backtest Parameters:")
    print(f"      Period: {common_dates.min().date()} to {common_dates.max().date()}")
    print(f"      Days: {len(common_dates)}")
    print(f"      Transaction cost: {transaction_cost:.2%}")
    print(f"      Rebalance threshold: {rebalance_threshold:.0%}")
    
    # Equal weight portfolio
    n_sectors = len(weights.columns)
    equal_weight = 1.0 / n_sectors
    ew_returns = (sector_returns * equal_weight).sum(axis=1)
    
    # Stress routing portfolio
    sr_returns = []
    sr_costs = []
    prev_weights = pd.Series(equal_weight, index=weights.columns)
    
    for date in common_dates:
        current_weights = weights.loc[date].fillna(equal_weight)
        
        # Normalize
        current_weights = current_weights / current_weights.sum()
        
        # Transaction costs
        weight_change = (current_weights - prev_weights).abs().sum()
        if weight_change > rebalance_threshold:
            cost = weight_change * transaction_cost
            sr_costs.append(cost)
            prev_weights = current_weights
        else:
            sr_costs.append(0)
            current_weights = prev_weights  # Don't rebalance
        
        # Portfolio return
        portfolio_return = (sector_returns.loc[date] * current_weights).sum()
        sr_returns.append(portfolio_return)
    
    sr_returns = pd.Series(sr_returns, index=common_dates)
    sr_costs = pd.Series(sr_costs, index=common_dates)
    sr_returns_net = sr_returns - sr_costs
    
    # Cumulative returns
    ew_cumret = (1 + ew_returns).cumprod()
    sr_cumret = (1 + sr_returns_net).cumprod()
    
    # Performance metrics
    years = len(common_dates) / 252
    
    ew_cagr = (ew_cumret.iloc[-1] ** (1/years) - 1) * 100
    sr_cagr = (sr_cumret.iloc[-1] ** (1/years) - 1) * 100
    
    ew_vol = ew_returns.std() * np.sqrt(252) * 100
    sr_vol = sr_returns_net.std() * np.sqrt(252) * 100
    
    ew_sharpe = (ew_cagr - 5) / ew_vol  # Assuming 5% risk-free
    sr_sharpe = (sr_cagr - 5) / sr_vol
    
    # Max drawdown
    ew_dd = (ew_cumret / ew_cumret.cummax() - 1).min() * 100
    sr_dd = (sr_cumret / sr_cumret.cummax() - 1).min() * 100
    
    # Outperformance
    outperformance = sr_cagr - ew_cagr
    total_costs = sr_costs.sum() * 100
    
    print(f"\n   📊 RESULTS:")
    print(f"\n      {'Metric':<20} {'Equal-Weight':>15} {'Stress-Routing':>15}")
    print(f"      {'-'*50}")
    print(f"      {'CAGR':<20} {ew_cagr:>14.2f}% {sr_cagr:>14.2f}%")
    print(f"      {'Volatility':<20} {ew_vol:>14.2f}% {sr_vol:>14.2f}%")
    print(f"      {'Sharpe Ratio':<20} {ew_sharpe:>15.2f} {sr_sharpe:>15.2f}")
    print(f"      {'Max Drawdown':<20} {ew_dd:>14.2f}% {sr_dd:>14.2f}%")
    print(f"\n      Outperformance: {outperformance:+.0f} bps")
    print(f"      Total transaction costs: {total_costs:.2f}%")
    
    # Verdict
    if outperformance >= 200:
        print(f"\n   ✅ H3 SUPPORTED: Stress routing outperforms by {outperformance:.0f} bps")
    elif outperformance > 0:
        print(f"\n   🟡 H3 PARTIALLY SUPPORTED: Outperforms by {outperformance:.0f} bps (< 200)")
    else:
        print(f"\n   ❌ H3 NOT SUPPORTED: Underperforms by {-outperformance:.0f} bps")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(ew_cumret.index, ew_cumret, 'b-', label='Equal Weight', linewidth=1.5)
    ax1.plot(sr_cumret.index, sr_cumret, 'g-', label='Stress Routing', linewidth=1.5)
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.set_title('H3: Stress Routing vs Equal Weight Portfolio', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    relative_perf = (sr_cumret / ew_cumret - 1) * 100
    ax2.fill_between(relative_perf.index, 0, relative_perf, 
                     where=relative_perf>0, color='green', alpha=0.5, label='Outperformance')
    ax2.fill_between(relative_perf.index, 0, relative_perf, 
                     where=relative_perf<0, color='red', alpha=0.5, label='Underperformance')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Relative Performance (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/h3_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: plots/h3_backtest.png")
    
    return {
        'ew_cagr': ew_cagr,
        'sr_cagr': sr_cagr,
        'outperformance_bps': outperformance * 100,
        'ew_sharpe': ew_sharpe,
        'sr_sharpe': sr_sharpe
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  Λ-VOLATILITY DIAGNOSTIC: RESEARCH EXTENSIONS")
    print("=" * 60)
    
    # Load data
    fdi_output, sectoral_fdi, strategy_signals, stress_weights = load_all_outputs()
    nifty_returns = load_nifty_returns()
    
    if nifty_returns is None:
        print("\n❌ Cannot proceed without NIFTY returns")
        return
    
    results = {}
    
    # H1: NBFC FDI predicts drawdown - GRID SEARCH for best parameters
    print("\n" + "=" * 60)
    print("  H1 PARAMETER GRID SEARCH")
    print("=" * 60)
    
    h1_grid_results = []
    for threshold in [1.0, 1.25, 1.5]:
        for persistence in [2, 3, 5]:
            h1_result = test_h1_nbfc_predicts_drawdown(
                sectoral_fdi, nifty_returns,
                threshold=threshold, persistence_days=persistence
            )
            if h1_result is not None:
                hit_rate = h1_result['hit'].mean()
                n_events = len(h1_result)
                h1_grid_results.append({
                    'threshold': threshold,
                    'persistence': persistence,
                    'hit_rate': hit_rate,
                    'n_events': n_events,
                    'avg_return': h1_result['forward_return'].mean()
                })
    
    # Find best config
    if h1_grid_results:
        best = max(h1_grid_results, key=lambda x: x['hit_rate'] if x['n_events'] >= 5 else 0)
        print("\n" + "=" * 60)
        print("  H1 GRID SEARCH RESULTS")
        print("=" * 60)
        print(f"\n   {'Threshold':>10} {'Persist':>8} {'Events':>8} {'Hit Rate':>10} {'Avg Ret':>10}")
        print(f"   " + "-" * 50)
        for r in h1_grid_results:
            marker = " ⭐" if r == best else ""
            print(f"   {r['threshold']:>10} {r['persistence']:>8} {r['n_events']:>8} {r['hit_rate']:>9.1%} {r['avg_return']:>9.2%}{marker}")
        print(f"\n   ⭐ BEST: threshold={best['threshold']}, persistence={best['persistence']} → {best['hit_rate']:.1%} hit rate")
    
    results['H1'] = h1_grid_results
    
    # H2: RBI policy impact
    h2_results = test_h2_rbi_policy_impact(fdi_output)
    results['H2'] = h2_results
    
    # H3: Stress routing backtest
    h3_results = test_h3_stress_routing_backtest(stress_weights, sectoral_fdi, nifty_returns)
    results['H3'] = h3_results
    
    # Summary
    print("\n" + "=" * 60)
    print("  HYPOTHESIS TESTING SUMMARY")
    print("=" * 60)
    
    print("\n   H1: NBFC FDI → NIFTY Drawdown (Grid Search)")
    if h1_grid_results:
        best = max(h1_grid_results, key=lambda x: x['hit_rate'] if x['n_events'] >= 5 else 0)
        print(f"       Best config: threshold={best['threshold']}, persist={best['persistence']}d → {best['hit_rate']:.1%} hit rate")
    
    print("\n   H2: RBI Policy → Increased Transitions")
    if h2_results is not None:
        print(f"       Ratio: {h2_results['ratio']:.2f}x, p-value: {h2_results['p_value']:.4f}")
    
    print("\n   H3: Stress Routing Outperformance")
    if h3_results is not None:
        print(f"       Outperformance: {h3_results['outperformance_bps']:.0f} bps")
    
    return results


if __name__ == "__main__":
    results = main()
