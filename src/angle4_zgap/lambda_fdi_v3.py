"""
Lambda-FDI v3: Advanced Framework with All Improvements
=========================================================

Improvements from v2:
1. SECTOR-WEIGHTED - Uses NBFC/Banks as systemic indicators
2. REGIME-CONDITIONED - Different equations per regime
3. EIGENVALUE STABILITY - Formal stability diagnostic

Master Equation (Regime-Adaptive):
   Healthy:    ρ̇ = 0.3F + 0.2G - 0.5D  (strong dissipation)
   Hidden:     ρ̇ = 0.4F + 0.4G - 0.2D  (rising feedback)
   Reflexive:  ρ̇ = 0.2F + 0.6G - 0.2D  (feedback dominates)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================

def load_intraday_data(data_dir='raw_Data'):
    """Load all intraday stock data with sector classification."""
    print("📦 Loading intraday data...")
    
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    # Sector classification (based on typical NIFTY 500 stocks)
    NBFC_STOCKS = ['BAJFINANCE', 'BAJAJFINSV', 'CHOLAFIN', 'MUTHOOTFIN', 'SHRIRAMFIN', 
                   'M&MFIN', 'LICHSGFIN', 'SBICARD', 'POONAWALLA', 'AAVAS']
    BANK_STOCKS = ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK',
                   'INDUSINDBK', 'BANDHANBNK', 'FEDERALBNK', 'IDFCFIRSTB', 'RBLBANK']
    METAL_STOCKS = ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'COALINDIA',
                    'NMDC', 'SAIL', 'NATIONALUM', 'JINDALSTEL']
    
    stock_data = {'all': {}, 'nbfc': {}, 'bank': {}, 'metal': {}}
    
    for f in all_files:
        ticker = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, parse_dates=['date'])
            if len(df) > 1000:
                stock_data['all'][ticker] = df
                
                if ticker in NBFC_STOCKS:
                    stock_data['nbfc'][ticker] = df
                elif ticker in BANK_STOCKS:
                    stock_data['bank'][ticker] = df
                elif ticker in METAL_STOCKS:
                    stock_data['metal'][ticker] = df
        except:
            continue
    
    print(f"   ✅ Loaded {len(stock_data['all'])} stocks")
    print(f"      NBFCs: {len(stock_data['nbfc'])}, Banks: {len(stock_data['bank'])}, Metals: {len(stock_data['metal'])}")
    
    return stock_data


def aggregate_sector(stock_dict):
    """Aggregate stocks to daily sector data."""
    if len(stock_dict) == 0:
        return None
        
    daily_data = []
    
    for ticker, df in stock_dict.items():
        df = df.sort_values('date')
        daily = df.groupby(df['date'].dt.date).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        }).reset_index()
        daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        daily['date'] = pd.to_datetime(daily['date'])
        daily['return'] = daily['close'].pct_change()
        daily_data.append(daily)
    
    all_daily = pd.concat(daily_data, ignore_index=True)
    
    sector = all_daily.groupby('date').agg({
        'return': 'mean', 'volume': 'sum',
        'open': 'mean', 'high': 'mean', 'low': 'mean', 'close': 'mean'
    }).sort_index()
    
    return sector


# =============================================================================
# BASE FDI COMPUTATION
# =============================================================================

def compute_sector_fdi(sector_data, name='sector'):
    """Compute FDI for a sector."""
    if sector_data is None or len(sector_data) < 100:
        return None
    
    df = sector_data.copy()
    
    # Parkinson volatility
    df['vol'] = np.sqrt((1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2))
    
    # Amihud illiquidity
    df['amihud'] = np.abs(df['return']) / (df['volume'] * df['close'] / 1e7 + 1e-10)
    
    # FDI
    df['fdi'] = df['vol'] / (df['amihud'] + 1e-10)
    df['fdi_smooth'] = df['fdi'].rolling(5).mean()
    df['fdi_zscore'] = (df['fdi_smooth'] - df['fdi_smooth'].rolling(252).mean()) / df['fdi_smooth'].rolling(252).std()
    
    return df[['fdi', 'fdi_smooth', 'fdi_zscore', 'return', 'volume']].copy()


# =============================================================================
# SECTOR-WEIGHTED Λ-FDI
# =============================================================================

def compute_sector_weighted_fdi(stock_data):
    """
    Sector-Weighted FDI: Weight by systemic importance
    
    Weights based on your findings:
    - NBFCs: 40% (proven to lead instability)
    - Banks: 35% (systemic importance)
    - Market: 25% (baseline)
    """
    print("\n📊 Computing Sector-Weighted Λ-FDI...")
    
    # Aggregate each sector
    market = aggregate_sector(stock_data['all'])
    nbfc = aggregate_sector(stock_data['nbfc'])
    bank = aggregate_sector(stock_data['bank'])
    
    # Compute FDI for each
    market_fdi = compute_sector_fdi(market, 'market')
    nbfc_fdi = compute_sector_fdi(nbfc, 'nbfc')
    bank_fdi = compute_sector_fdi(bank, 'bank')
    
    # Combine into single DataFrame
    result = pd.DataFrame(index=market.index)
    result['market_fdi'] = market_fdi['fdi_zscore'] if market_fdi is not None else 0
    result['market_return'] = market['return']
    result['market_volume'] = market['volume']
    
    # Sector FDIs (may have fewer dates)
    if nbfc_fdi is not None:
        result = result.join(nbfc_fdi['fdi_zscore'].rename('nbfc_fdi'), how='left')
    else:
        result['nbfc_fdi'] = 0
        
    if bank_fdi is not None:
        result = result.join(bank_fdi['fdi_zscore'].rename('bank_fdi'), how='left')
    else:
        result['bank_fdi'] = 0
    
    # Fill NaN with market FDI
    result['nbfc_fdi'] = result['nbfc_fdi'].fillna(result['market_fdi'])
    result['bank_fdi'] = result['bank_fdi'].fillna(result['market_fdi'])
    
    # Weighted FDI (based on your research: NBFCs lead)
    w_nbfc, w_bank, w_market = 0.40, 0.35, 0.25
    result['FDI_systemic'] = (
        w_nbfc * result['nbfc_fdi'] + 
        w_bank * result['bank_fdi'] + 
        w_market * result['market_fdi']
    )
    
    print(f"   ✅ Sector-weighted FDI computed")
    print(f"      Weights: NBFC={w_nbfc:.0%}, Bank={w_bank:.0%}, Market={w_market:.0%}")
    print(f"      Systemic FDI mean: {result['FDI_systemic'].mean():.4f}")
    
    return result, market


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(fdi_systemic, market_return):
    """
    Detect market regime based on FDI and returns.
    
    Regimes:
    0 = Healthy:   Low FDI, stable returns
    1 = Hidden:    Rising FDI, returns still positive
    2 = Reflexive: High FDI, negative returns
    """
    print("\n📊 Detecting Regimes...")
    
    regimes = pd.Series(index=fdi_systemic.index, dtype=int)
    
    # Rolling volatility of returns
    ret_vol = market_return.rolling(20).std()
    ret_vol_z = (ret_vol - ret_vol.rolling(252).mean()) / ret_vol.rolling(252).std()
    
    # FDI momentum
    fdi_momentum = fdi_systemic.diff(5)
    
    for i in range(len(fdi_systemic)):
        fdi = fdi_systemic.iloc[i]
        fdi_mom = fdi_momentum.iloc[i] if i >= 5 else 0
        vol_z = ret_vol_z.iloc[i] if not pd.isna(ret_vol_z.iloc[i]) else 0
        
        if fdi > 1.5 and vol_z > 1:
            regimes.iloc[i] = 2  # Reflexive
        elif fdi > 0.5 or fdi_mom > 0.3:
            regimes.iloc[i] = 1  # Hidden
        else:
            regimes.iloc[i] = 0  # Healthy
    
    regime_counts = regimes.value_counts()
    print(f"   ✅ Regimes detected:")
    print(f"      Healthy: {regime_counts.get(0, 0)} days ({regime_counts.get(0, 0)/len(regimes)*100:.1f}%)")
    print(f"      Hidden: {regime_counts.get(1, 0)} days ({regime_counts.get(1, 0)/len(regimes)*100:.1f}%)")
    print(f"      Reflexive: {regime_counts.get(2, 0)} days ({regime_counts.get(2, 0)/len(regimes)*100:.1f}%)")
    
    return regimes


# =============================================================================
# REGIME-CONDITIONED Λ-FDI
# =============================================================================

def compute_regime_conditioned_fdi(data, regimes):
    """
    Regime-Conditioned Λ-FDI: Different equations per regime
    
    Healthy:    ρ̇ = 0.3F + 0.2G - 0.5D  (strong dissipation)
    Hidden:     ρ̇ = 0.4F + 0.4G - 0.2D  (rising feedback)
    Reflexive:  ρ̇ = 0.2F + 0.6G - 0.2D  (feedback dominates)
    """
    print("\n📊 Computing Regime-Conditioned Λ-FDI...")
    
    fdi = data['FDI_systemic'].fillna(0)
    returns = data['market_return'].fillna(0)
    
    # Compute components (all z-scored)
    # F: Regime dynamics (slow)
    F = fdi.rolling(60).mean().fillna(0)
    
    # G: Feedback pressure
    momentum = returns.rolling(20).mean()
    momentum_z = (momentum - momentum.rolling(252).mean()) / (momentum.rolling(252).std() + 1e-10)
    G = (fdi * momentum_z.clip(-2, 2)).rolling(5).mean().fillna(0)
    
    # D: Dissipation (capped)
    D_persist = 0.3 * fdi.shift(1).fillna(0)
    D_shock = 0.5 * (-fdi.diff().rolling(5).mean().fillna(0))
    D_crowd = 0.1 * (fdi.clip(-3, 3) ** 2)
    D = D_persist + D_shock + D_crowd
    
    # Regime-specific weights
    weights = {
        0: {'f': 0.3, 'g': 0.2, 'd': 0.5},  # Healthy: strong dissipation
        1: {'f': 0.4, 'g': 0.4, 'd': 0.2},  # Hidden: rising feedback
        2: {'f': 0.2, 'g': 0.6, 'd': 0.2},  # Reflexive: feedback dominates
    }
    
    # Apply regime-specific equation
    lambda_fdi = pd.Series(index=data.index, dtype=float)
    
    for regime, w in weights.items():
        mask = regimes == regime
        lambda_fdi[mask] = w['f'] * F[mask] + w['g'] * G[mask] - w['d'] * D[mask]
    
    # Z-score
    lambda_fdi_z = (lambda_fdi - lambda_fdi.rolling(252).mean()) / lambda_fdi.rolling(252).std()
    
    data['Lambda_FDI_regime'] = lambda_fdi
    data['Lambda_FDI_regime_zscore'] = lambda_fdi_z
    data['F_component'] = F
    data['G_component'] = G
    data['D_component'] = D
    data['regime'] = regimes
    
    print(f"   ✅ Regime-conditioned Λ-FDI computed")
    
    return data


# =============================================================================
# EIGENVALUE STABILITY DIAGNOSTIC
# =============================================================================

def compute_eigenvalue_stability(data, window=20):
    """
    Eigenvalue Stability Diagnostic
    
    From the master equation ρ̇ = F + G - D, linearize around current state.
    Compute largest eigenvalue to determine local stability.
    
    λ < 0: Stable (stress will decay)
    λ > 0: Unstable (stress will grow)
    λ ≈ 0: Critical (edge of instability)
    """
    print("\n📊 Computing Eigenvalue Stability...")
    
    F = data['F_component'].fillna(0)
    G = data['G_component'].fillna(0)
    D = data['D_component'].fillna(0)
    
    # For a 1D system, "eigenvalue" is the local derivative of the RHS
    # λ = d(F + G - D)/dρ ≈ rate of change
    
    # Approximate: how fast is the system accelerating?
    total = F + G - D
    eigenvalue = total.diff().rolling(window).mean()
    
    # Stability indicator
    # λ < -0.1: Strongly stable
    # -0.1 < λ < 0.1: Marginal
    # λ > 0.1: Unstable
    
    stability = pd.Series(index=data.index, dtype=str)
    stability[eigenvalue < -0.1] = 'stable'
    stability[(eigenvalue >= -0.1) & (eigenvalue <= 0.1)] = 'marginal'
    stability[eigenvalue > 0.1] = 'unstable'
    
    data['eigenvalue'] = eigenvalue
    data['stability'] = stability
    
    stable_pct = (stability == 'stable').mean() * 100
    unstable_pct = (stability == 'unstable').mean() * 100
    
    print(f"   ✅ Eigenvalue stability computed")
    print(f"      Stable: {stable_pct:.1f}%")
    print(f"      Marginal: {100 - stable_pct - unstable_pct:.1f}%")
    print(f"      Unstable: {unstable_pct:.1f}%")
    
    return data


# =============================================================================
# COMPARISON
# =============================================================================

def compare_all_signals(data, window=10, drawdown_thresh=0.03):
    """Compare all FDI variants."""
    print("\n" + "=" * 70)
    print("  COMPARING ALL FDI SIGNALS (v3)")
    print("=" * 70)
    
    df = data.copy()
    df['fwd_return'] = df['market_return'].rolling(window).sum().shift(-window)
    df['fwd_min'] = df['market_return'].rolling(window).min().shift(-window)
    df['had_drawdown'] = df['fwd_min'] < -drawdown_thresh
    
    variants = [
        ('market_fdi', 'Market FDI (base)'),
        ('FDI_systemic', 'Sector-Weighted'),
        ('Lambda_FDI_regime_zscore', 'Λ-FDI Regime-Cond.'),
    ]
    
    results = []
    
    for col, name in variants:
        if col not in df.columns:
            continue
        
        df_clean = df[[col, 'had_drawdown', 'fwd_return']].dropna()
        
        for thresh in [1.0, 1.5, 2.0]:
            high = df_clean[df_clean[col] > thresh]
            if len(high) < 5:
                continue
            
            results.append({
                'variant': name,
                'threshold': thresh,
                'hit_rate': high['had_drawdown'].mean(),
                'avg_return': high['fwd_return'].mean(),
                'n_events': len(high)
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n   {'Signal':<25} {'Thresh':>8} {'Hit Rate':>10} {'Avg Ret':>10} {'Events':>8}")
    print("   " + "-" * 65)
    
    for _, r in results_df.iterrows():
        print(f"   {r['variant']:<25} {r['threshold']:>8.1f} {r['hit_rate']:>9.1%} {r['avg_return']:>9.2%} {r['n_events']:>8}")
    
    # Best per variant
    print("\n   📊 BEST PER VARIANT:")
    for variant in results_df['variant'].unique():
        subset = results_df[results_df['variant'] == variant]
        if len(subset) > 0:
            best = subset.loc[subset['hit_rate'].idxmax()]
            print(f"      {variant}: {best['hit_rate']:.1%} @ thresh={best['threshold']}")
    
    # Correlations
    print("\n   📊 CORRELATION WITH FORWARD RETURNS:")
    df['fwd_5d'] = df['market_return'].rolling(5).sum().shift(-5)
    df['fwd_10d'] = df['market_return'].rolling(10).sum().shift(-10)
    
    print(f"   {'Signal':<25} {'5d':>10} {'10d':>10}")
    print("   " + "-" * 45)
    
    for col, name in variants:
        if col in df.columns:
            c5 = df[col].corr(df['fwd_5d'])
            c10 = df[col].corr(df['fwd_10d'])
            print(f"   {name:<25} {c5:>+10.4f} {c10:>+10.4f}")
    
    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_v3(data):
    """Visualize v3 improvements."""
    print("\n📊 Creating visualizations...")
    
    df = data.dropna(subset=['Lambda_FDI_regime_zscore']).copy()
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    
    # 1. Sector-Weighted vs Market FDI
    ax1 = axes[0]
    ax1.plot(df.index, df['market_fdi'], 'b-', alpha=0.5, label='Market FDI', linewidth=1)
    ax1.plot(df.index, df['FDI_systemic'], 'r-', alpha=0.8, label='Sector-Weighted', linewidth=1.5)
    ax1.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Z-Score')
    ax1.set_title('Market FDI vs Sector-Weighted (NBFC/Bank)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Regime-Conditioned Λ-FDI
    ax2 = axes[1]
    ax2.plot(df.index, df['Lambda_FDI_regime_zscore'], 'purple', linewidth=1.5)
    ax2.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Z-Score')
    ax2.set_title('Λ-FDI (Regime-Conditioned)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Regimes
    ax3 = axes[2]
    colors = {0: 'green', 1: 'orange', 2: 'red'}
    for regime in [0, 1, 2]:
        mask = df['regime'] == regime
        for i in range(len(df)):
            if mask.iloc[i]:
                ax3.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                           alpha=0.3, color=colors[regime])
    ax3.set_ylabel('Regime')
    ax3.set_title('Regime (Green=Healthy, Orange=Hidden, Red=Reflexive)', fontweight='bold')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Healthy', 'Hidden', 'Reflexive'])
    
    # 4. Eigenvalue Stability
    ax4 = axes[3]
    ax4.plot(df.index, df['eigenvalue'], 'k-', linewidth=1)
    ax4.axhline(y=0, color='red', linestyle='-', linewidth=2)
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5)
    ax4.axhline(y=-0.1, color='green', linestyle='--', alpha=0.5)
    ax4.fill_between(df.index, -0.1, 0.1, alpha=0.1, color='yellow')
    ax4.set_ylabel('Eigenvalue')
    ax4.set_title('Eigenvalue Stability (>0 = Unstable, <0 = Stable)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative returns with signals
    ax5 = axes[4]
    cumret = (1 + df['market_return']).cumprod()
    ax5.plot(df.index, cumret, 'k-', linewidth=1)
    
    high_lambda = df['Lambda_FDI_regime_zscore'] > 2
    for i in range(len(df)):
        if high_lambda.iloc[i]:
            ax5.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], alpha=0.4, color='red')
    
    ax5.set_ylabel('Cumulative Return')
    ax5.set_xlabel('Date')
    ax5.set_title('Market Returns (Red = High Λ-FDI)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lambda_fdi_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/lambda_fdi_v3.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Λ-FDI v3: ADVANCED FRAMEWORK")
    print("=" * 70)
    print("\n   Improvements:")
    print("   ✅ Sector-Weighted (NBFC 40% + Bank 35% + Market 25%)")
    print("   ✅ Regime-Conditioned Equations")
    print("   ✅ Eigenvalue Stability Diagnostic")
    
    # Load data with sector classification
    stock_data = load_intraday_data()
    if len(stock_data['all']) == 0:
        print("❌ No stock data found")
        return None
    
    # Compute sector-weighted FDI
    data, market = compute_sector_weighted_fdi(stock_data)
    
    # Detect regimes
    regimes = detect_regime(data['FDI_systemic'], data['market_return'])
    
    # Compute regime-conditioned Λ-FDI
    data = compute_regime_conditioned_fdi(data, regimes)
    
    # Compute eigenvalue stability
    data = compute_eigenvalue_stability(data)
    
    # Save
    data.to_csv('lambda_fdi_v3_output.csv')
    print(f"\n   ✅ Saved: lambda_fdi_v3_output.csv")
    
    # Compare
    results = compare_all_signals(data)
    
    # Visualize
    visualize_v3(data)
    
    # Summary
    print("\n" + "=" * 70)
    print("  Λ-FDI v3 SUMMARY")
    print("=" * 70)
    
    print("\n   COMPLETE FRAMEWORK:")
    print("   ┌────────────────────────────────────────────────────────────────┐")
    print("   │ 1. Sector-Weighted    │ NBFC/Bank leading indicators          │")
    print("   │ 2. Regime-Conditioned │ Adaptive equations per market state   │")
    print("   │ 3. Eigenvalue         │ Formal stability metric               │")
    print("   └────────────────────────────────────────────────────────────────┘")
    
    print("\n   USAGE GUIDE:")
    print("   • Λ-FDI > 2.0 + Unstable eigenvalue → CRISIS → EXIT")
    print("   • Λ-FDI > 1.5 + Hidden regime → CAUTION → REDUCE")
    print("   • Λ-FDI < 1.0 + Stable eigenvalue → HEALTHY → HOLD")
    
    return data


if __name__ == "__main__":
    data = main()
