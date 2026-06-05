"""
Λ-Volatility Diagnostic Engine: Tier 1 Publication Extensions
==============================================================

1. Granger Causality Network - Does FDI CAUSE market moves?
2. Cross-Market Contagion - Global spillover effects
3. Fama-French Factor Analysis - Is FDI a priced risk factor?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================

def load_fdi_data():
    """Load FDI outputs from main pipeline."""
    print("📦 Loading FDI data...")
    
    fdi = pd.read_csv("fdi_output.csv", parse_dates=['date'], index_col='date')
    sectoral = pd.read_csv("sectoral_fdi_output.csv", parse_dates=['date'], index_col='date')
    
    print(f"   ✅ Market FDI: {len(fdi)} days")
    print(f"   ✅ Sectoral FDI: {len(sectoral)} days, {len(sectoral.columns)} sectors")
    
    return fdi, sectoral


def load_market_data():
    """Load market data for analysis."""
    print("\n📊 Fetching market data...")
    
    tickers = {
        'NIFTY': '^NSEI',
        'NIFTYBANK': '^NSEBANK',
        'SPX': '^GSPC',
        'VIX': '^VIX',
        'INDIAVIX': '^INDIAVIX'
    }
    
    data = {}
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
            if not df.empty:
                # Handle potential MultiIndex from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    close_col = df['Close'].iloc[:, 0] if len(df['Close'].shape) > 1 else df['Close']
                else:
                    close_col = df['Close']
                data[name] = close_col
                print(f"   ✅ {name}: {len(df)} days")
            else:
                print(f"   ⚠️ {name}: No data")
        except Exception as e:
            print(f"   ❌ {name}: {e}")
    
    market_df = pd.DataFrame(data)
    
    # Compute returns
    returns_df = market_df.pct_change().dropna()
    
    return market_df, returns_df


# =============================================================================
# 1. GRANGER CAUSALITY NETWORK
# =============================================================================

def test_granger_causality(fdi, returns_df, max_lag=10):
    """
    Test if FDI Granger-causes market returns (and vice versa).
    
    Granger causality tests whether past values of X help predict Y
    beyond what past values of Y alone can predict.
    """
    print("\n" + "=" * 70)
    print("  1. GRANGER CAUSALITY ANALYSIS")
    print("=" * 70)
    
    # Align data
    common_idx = fdi.index.intersection(returns_df.index)
    fdi_aligned = fdi.loc[common_idx, 'FDI_zscore'].dropna()
    
    results = {}
    
    # Test pairs
    test_pairs = [
        ('FDI_zscore', 'NIFTY', "Does FDI Granger-cause NIFTY returns?"),
        ('NIFTY', 'FDI_zscore', "Do NIFTY returns Granger-cause FDI?"),
        ('FDI_zscore', 'NIFTYBANK', "Does FDI Granger-cause Bank Nifty?"),
    ]
    
    for x_name, y_name, description in test_pairs:
        print(f"\n   📊 {description}")
        
        try:
            if x_name == 'FDI_zscore':
                x_data = fdi_aligned
            else:
                x_data = returns_df.loc[common_idx, x_name].dropna()
            
            if y_name == 'FDI_zscore':
                y_data = fdi_aligned
            else:
                y_data = returns_df.loc[common_idx, y_name].dropna()
            
            # Align
            common = x_data.index.intersection(y_data.index)
            df_test = pd.DataFrame({
                'x': x_data.loc[common],
                'y': y_data.loc[common]
            }).dropna()
            
            if len(df_test) < 100:
                print(f"      ⚠️ Insufficient data: {len(df_test)} points")
                continue
            
            # Check stationarity (ADF test)
            adf_x = adfuller(df_test['x'], autolag='AIC')[1]
            adf_y = adfuller(df_test['y'], autolag='AIC')[1]
            
            # Granger test - note: order is [y, x] to test "x causes y"
            gc_results = grangercausalitytests(df_test[['y', 'x']], maxlag=max_lag, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            for lag in range(1, max_lag + 1):
                p_values[lag] = gc_results[lag][0]['ssr_ftest'][1]
            
            # Find best lag (lowest p-value)
            best_lag = min(p_values, key=p_values.get)
            best_p = p_values[best_lag]
            
            # Verdict
            if best_p < 0.01:
                verdict = "✅ STRONG CAUSALITY"
                sig = "***"
            elif best_p < 0.05:
                verdict = "✅ SIGNIFICANT"
                sig = "**"
            elif best_p < 0.10:
                verdict = "🟡 WEAK"
                sig = "*"
            else:
                verdict = "❌ NO CAUSALITY"
                sig = ""
            
            print(f"      Stationarity: X (p={adf_x:.3f}), Y (p={adf_y:.3f})")
            print(f"      Best lag: {best_lag} days, p-value: {best_p:.4f} {sig}")
            print(f"      Verdict: {verdict}")
            
            results[f"{x_name}_causes_{y_name}"] = {
                'best_lag': best_lag,
                'p_value': best_p,
                'all_p_values': p_values,
                'significant': best_p < 0.05
            }
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    return results


# =============================================================================
# 2. CROSS-MARKET CONTAGION ANALYSIS
# =============================================================================

def test_cross_market_contagion(fdi, sectoral_fdi, market_df, returns_df):
    """
    Test cross-market spillover effects:
    - Does US VIX predict India FDI?
    - Does S&P 500 crash predict India sector stress?
    - Lead-lag relationships between markets
    """
    print("\n" + "=" * 70)
    print("  2. CROSS-MARKET CONTAGION ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # Align all data
    common_idx = fdi.index.intersection(market_df.index)
    
    # 2A: VIX → India FDI correlation at different lags
    print("\n   📊 2A: US VIX → India FDI Lead-Lag Analysis")
    print("   " + "-" * 50)
    
    if 'VIX' in market_df.columns:
        vix = market_df.loc[common_idx, 'VIX'].dropna()
        fdi_z = fdi.loc[common_idx, 'FDI_zscore'].dropna()
        
        common = vix.index.intersection(fdi_z.index)
        vix_aligned = vix.loc[common]
        fdi_aligned = fdi_z.loc[common]
        
        # Test correlations at different lags
        correlations = {}
        for lag in range(-5, 6):  # -5 to +5 days
            if lag < 0:
                # VIX leads FDI
                corr = vix_aligned.iloc[:lag].corr(fdi_aligned.iloc[-lag:])
                label = f"VIX leads by {-lag}d"
            elif lag > 0:
                # FDI leads VIX
                corr = vix_aligned.iloc[lag:].corr(fdi_aligned.iloc[:-lag])
                label = f"FDI leads by {lag}d"
            else:
                corr = vix_aligned.corr(fdi_aligned)
                label = "Same day"
            correlations[lag] = corr
        
        best_lag = max(correlations, key=lambda x: abs(correlations[x]))
        best_corr = correlations[best_lag]
        
        print(f"\n      Lag | Correlation")
        print(f"      ----|------------")
        for lag in range(-5, 6):
            marker = " ⭐" if lag == best_lag else ""
            print(f"      {lag:+3d} | {correlations[lag]:+.3f}{marker}")
        
        if best_lag < 0:
            print(f"\n      ⭐ US VIX leads India FDI by {-best_lag} days (ρ = {best_corr:.3f})")
        elif best_lag > 0:
            print(f"\n      ⭐ India FDI leads US VIX by {best_lag} days (ρ = {best_corr:.3f})")
        else:
            print(f"\n      ⭐ Same-day correlation: ρ = {best_corr:.3f}")
        
        results['vix_fdi_leadlag'] = correlations
        results['vix_fdi_best_lag'] = best_lag
    
    # 2B: S&P 500 crash → India sector stress
    print("\n\n   📊 2B: S&P 500 Crash → India Sector Stress")
    print("   " + "-" * 50)
    
    if 'SPX' in returns_df.columns:
        spx_ret = returns_df.loc[common_idx, 'SPX'].dropna()
        
        # Define S&P crash days: return < -2%
        crash_days = spx_ret[spx_ret < -0.02].index
        non_crash_days = spx_ret[spx_ret >= -0.02].index
        
        print(f"\n      S&P 500 crash days (ret < -2%): {len(crash_days)}")
        
        # Compare sector FDI on crash vs non-crash days (next day)
        sector_stress = {}
        for sector in sectoral_fdi.columns:
            sector_fdi = sectoral_fdi[sector]
            
            # Next-day sector FDI after S&P crash
            crash_next_day = [sector_fdi.loc[d + pd.Timedelta(days=1)] 
                             for d in crash_days 
                             if (d + pd.Timedelta(days=1)) in sector_fdi.index]
            
            non_crash_next_day = [sector_fdi.loc[d + pd.Timedelta(days=1)] 
                                  for d in non_crash_days 
                                  if (d + pd.Timedelta(days=1)) in sector_fdi.index]
            
            if len(crash_next_day) > 5 and len(non_crash_next_day) > 5:
                avg_crash = np.mean(crash_next_day)
                avg_non_crash = np.mean(non_crash_next_day)
                t_stat, p_val = stats.ttest_ind(crash_next_day, non_crash_next_day)
                
                sector_stress[sector] = {
                    'avg_crash': avg_crash,
                    'avg_non_crash': avg_non_crash,
                    'diff': avg_crash - avg_non_crash,
                    'p_value': p_val
                }
        
        # Display results
        print(f"\n      {'Sector':<15} {'After Crash':>12} {'Normal':>10} {'Diff':>8} {'p-val':>8}")
        print(f"      " + "-" * 55)
        for sector, data in sorted(sector_stress.items(), key=lambda x: -x[1]['diff']):
            sig = "**" if data['p_value'] < 0.05 else "*" if data['p_value'] < 0.10 else ""
            print(f"      {sector:<15} {data['avg_crash']:>+12.2f} {data['avg_non_crash']:>+10.2f} {data['diff']:>+8.2f} {data['p_value']:>7.3f}{sig}")
        
        results['spx_crash_sector_stress'] = sector_stress
    
    # 2C: Granger test: VIX → NBFC FDI
    print("\n\n   📊 2C: Granger Causality - US VIX → NBFC FDI")
    print("   " + "-" * 50)
    
    if 'VIX' in market_df.columns and 'NBFCs' in sectoral_fdi.columns:
        vix = market_df['VIX'].pct_change().dropna()  # Use VIX changes
        nbfc = sectoral_fdi['NBFCs'].dropna()
        
        common = vix.index.intersection(nbfc.index)
        df_test = pd.DataFrame({
            'nbfc': nbfc.loc[common],
            'vix': vix.loc[common]
        }).dropna()
        
        if len(df_test) > 100:
            gc = grangercausalitytests(df_test[['nbfc', 'vix']], maxlag=5, verbose=False)
            p_values = {lag: gc[lag][0]['ssr_ftest'][1] for lag in range(1, 6)}
            best_lag = min(p_values, key=p_values.get)
            best_p = p_values[best_lag]
            
            if best_p < 0.05:
                print(f"      ✅ US VIX Granger-causes NBFC FDI (lag={best_lag}d, p={best_p:.4f})")
            else:
                print(f"      ❌ No significant causality (best p={best_p:.4f})")
            
            results['vix_granger_nbfc'] = {'p_value': best_p, 'lag': best_lag}
    
    return results


# =============================================================================
# 3. FAMA-FRENCH FACTOR ANALYSIS
# =============================================================================

def test_fama_french_analysis(fdi, returns_df, sectoral_fdi):
    """
    Test if FDI is a priced risk factor using Fama-French regression.
    
    Model: R_i = α + β_mkt * MKT + β_fdi * FDI + ε
    
    If β_fdi is significant, FDI exposure has explanatory power
    for cross-sectional returns.
    """
    print("\n" + "=" * 70)
    print("  3. FAMA-FRENCH FACTOR ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # Align data
    common_idx = fdi.index.intersection(returns_df.index)
    
    fdi_z = fdi.loc[common_idx, 'FDI_zscore'].dropna()
    nifty_ret = returns_df.loc[common_idx, 'NIFTY'].dropna() if 'NIFTY' in returns_df.columns else None
    
    if nifty_ret is None:
        print("   ❌ NIFTY returns not available")
        return results
    
    common = fdi_z.index.intersection(nifty_ret.index)
    
    # 3A: Time-Series Regression - Does FDI predict future returns?
    print("\n   📊 3A: FDI Predictive Power (Time-Series)")
    print("   " + "-" * 50)
    
    for horizon in [1, 5, 10, 20]:
        # Forward returns
        fwd_ret = nifty_ret.rolling(horizon).sum().shift(-horizon)
        
        df_reg = pd.DataFrame({
            'fdi': fdi_z.loc[common],
            'fwd_ret': fwd_ret.loc[common]
        }).dropna()
        
        if len(df_reg) < 100:
            continue
        
        X = add_constant(df_reg['fdi'])
        y = df_reg['fwd_ret']
        
        model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': horizon})
        
        beta = model.params['fdi']
        t_stat = model.tvalues['fdi']
        p_val = model.pvalues['fdi']
        r2 = model.rsquared
        
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        
        print(f"      Horizon {horizon:2d}d: β_fdi = {beta:+.4f}, t = {t_stat:+.2f}, R² = {r2:.3f} {sig}")
        
        results[f'fdi_predictive_{horizon}d'] = {
            'beta': beta,
            't_stat': t_stat,
            'p_value': p_val,
            'r_squared': r2
        }
    
    # 3B: High vs Low FDI Regime Returns
    print("\n\n   📊 3B: Return Differential by FDI Regime")
    print("   " + "-" * 50)
    
    df_regime = pd.DataFrame({
        'fdi': fdi_z.loc[common],
        'ret': nifty_ret.loc[common]
    }).dropna()
    
    # Quintile analysis
    df_regime['fdi_quintile'] = pd.qcut(df_regime['fdi'], 5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])
    
    quintile_stats = df_regime.groupby('fdi_quintile')['ret'].agg(['mean', 'std', 'count'])
    quintile_stats['mean_ann'] = quintile_stats['mean'] * 252
    quintile_stats['sharpe'] = (quintile_stats['mean'] / quintile_stats['std']) * np.sqrt(252)
    
    print(f"\n      {'FDI Quintile':<15} {'Ann. Return':>12} {'Volatility':>12} {'Sharpe':>8}")
    print(f"      " + "-" * 50)
    for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
        if q in quintile_stats.index:
            row = quintile_stats.loc[q]
            print(f"      {q:<15} {row['mean_ann']*100:>+11.2f}% {row['std']*np.sqrt(252)*100:>11.2f}% {row['sharpe']:>+8.2f}")
    
    # Long Q1 (Low FDI) - Short Q5 (High FDI)
    low_fdi_ret = df_regime[df_regime['fdi_quintile'] == 'Q1 (Low)']['ret'].mean()
    high_fdi_ret = df_regime[df_regime['fdi_quintile'] == 'Q5 (High)']['ret'].mean()
    spread = (low_fdi_ret - high_fdi_ret) * 252 * 100
    
    print(f"\n      📈 Long-Short Spread (Q1 - Q5): {spread:+.0f} bps annually")
    
    results['quintile_analysis'] = quintile_stats.to_dict()
    results['long_short_spread_bps'] = spread
    
    # 3C: FDI Beta by Sector
    print("\n\n   📊 3C: Sector FDI Betas")
    print("   " + "-" * 50)
    
    print(f"\n      {'Sector':<15} {'β_mkt':>8} {'β_fdi':>8} {'t(β_fdi)':>10} {'R²':>8}")
    print(f"      " + "-" * 55)
    
    sector_betas = {}
    for sector in sectoral_fdi.columns:
        sector_fdi_s = sectoral_fdi[sector].dropna()
        common_s = sector_fdi_s.index.intersection(nifty_ret.index)
        
        df_s = pd.DataFrame({
            'mkt': nifty_ret.loc[common_s],
            'fdi': sector_fdi_s.loc[common_s]
        }).dropna()
        
        if len(df_s) < 100:
            continue
        
        # Use 5-day forward returns
        df_s['fwd_ret'] = df_s['mkt'].rolling(5).sum().shift(-5)
        df_s = df_s.dropna()
        
        X = add_constant(df_s[['mkt', 'fdi']])
        y = df_s['fwd_ret']
        
        try:
            model = OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
            
            beta_mkt = model.params['mkt']
            beta_fdi = model.params['fdi']
            t_fdi = model.tvalues['fdi']
            r2 = model.rsquared
            
            sig = "**" if model.pvalues['fdi'] < 0.05 else "*" if model.pvalues['fdi'] < 0.10 else ""
            
            print(f"      {sector:<15} {beta_mkt:>+8.3f} {beta_fdi:>+8.4f} {t_fdi:>+10.2f} {r2:>8.3f}{sig}")
            
            sector_betas[sector] = {
                'beta_mkt': beta_mkt,
                'beta_fdi': beta_fdi,
                't_fdi': t_fdi,
                'r_squared': r2
            }
        except:
            continue
    
    results['sector_betas'] = sector_betas
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_tier1_visualizations(granger_results, contagion_results, ff_results):
    """Create summary visualizations for Tier 1 analysis."""
    print("\n" + "=" * 70)
    print("  CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Granger Causality Summary
    ax1 = axes[0, 0]
    if granger_results:
        labels = list(granger_results.keys())
        p_values = [granger_results[k]['p_value'] for k in labels]
        colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'red' for p in p_values]
        
        bars = ax1.barh(labels, [-np.log10(max(p, 1e-10)) for p in p_values], color=colors)
        ax1.axvline(x=-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
        ax1.set_xlabel('-log10(p-value)')
        ax1.set_title('Granger Causality Tests', fontweight='bold')
    
    # 2. VIX-FDI Lead-Lag
    ax2 = axes[0, 1]
    if 'vix_fdi_leadlag' in contagion_results:
        lags = list(contagion_results['vix_fdi_leadlag'].keys())
        corrs = list(contagion_results['vix_fdi_leadlag'].values())
        colors = ['green' if c > 0 else 'red' for c in corrs]
        ax2.bar(lags, corrs, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Lag (days)')
        ax2.set_ylabel('Correlation')
        ax2.set_title('US VIX ↔ India FDI Lead-Lag', fontweight='bold')
    
    # 3. Long-Short Spread
    ax3 = axes[1, 0]
    if 'quintile_analysis' in ff_results:
        quintiles = ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']
        returns = [ff_results['quintile_analysis']['mean_ann'].get(q, 0) * 100 for q in quintiles]
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, 5))
        ax3.bar(quintiles, returns, color=colors)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_ylabel('Annualized Return (%)')
        ax3.set_title('Returns by FDI Quintile', fontweight='bold')
    
    # 4. Sector FDI Betas
    ax4 = axes[1, 1]
    if 'sector_betas' in ff_results:
        sectors = list(ff_results['sector_betas'].keys())
        betas = [ff_results['sector_betas'][s]['beta_fdi'] for s in sectors]
        t_stats = [ff_results['sector_betas'][s]['t_fdi'] for s in sectors]
        colors = ['green' if abs(t) > 2 else 'gray' for t in t_stats]
        
        ax4.barh(sectors, betas, color=colors)
        ax4.axvline(x=0, color='black', linewidth=0.5)
        ax4.set_xlabel('FDI Beta')
        ax4.set_title('Sector FDI Betas (5d forward)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/tier1_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/tier1_analysis.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Λ-VOLATILITY DIAGNOSTIC: TIER 1 PUBLICATION EXTENSIONS")
    print("=" * 70)
    
    # Load data
    fdi, sectoral_fdi = load_fdi_data()
    market_df, returns_df = load_market_data()
    
    # 1. Granger Causality
    granger_results = test_granger_causality(fdi, returns_df)
    
    # 2. Cross-Market Contagion
    contagion_results = test_cross_market_contagion(fdi, sectoral_fdi, market_df, returns_df)
    
    # 3. Fama-French Factor Analysis
    ff_results = test_fama_french_analysis(fdi, returns_df, sectoral_fdi)
    
    # Visualizations
    create_tier1_visualizations(granger_results, contagion_results, ff_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("  TIER 1 SUMMARY")
    print("=" * 70)
    
    print("\n   1. GRANGER CAUSALITY:")
    for key, val in granger_results.items():
        status = "✅" if val['significant'] else "❌"
        print(f"      {status} {key}: p={val['p_value']:.4f}, lag={val['best_lag']}d")
    
    print("\n   2. CROSS-MARKET CONTAGION:")
    if 'vix_fdi_best_lag' in contagion_results:
        lag = contagion_results['vix_fdi_best_lag']
        if lag < 0:
            print(f"      ✅ US VIX leads India FDI by {-lag} days")
        else:
            print(f"      ✅ India FDI leads US VIX by {lag} days")
    
    print("\n   3. FAMA-FRENCH ANALYSIS:")
    if 'long_short_spread_bps' in ff_results:
        spread = ff_results['long_short_spread_bps']
        status = "✅" if spread > 100 else "🟡" if spread > 0 else "❌"
        print(f"      {status} Long-Short Spread: {spread:+.0f} bps")
    
    return {
        'granger': granger_results,
        'contagion': contagion_results,
        'fama_french': ff_results
    }


if __name__ == "__main__":
    results = main()
