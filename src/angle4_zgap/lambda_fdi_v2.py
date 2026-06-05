"""
Lambda-FDI v2: Fixed Feedback Dominance Framework
===================================================

Fixes from v1:
1. FIXED DISSIPATION - Properly scaled, capped quadratic term
2. DUAL OUTPUT - Λ-FDI_state (slow) + FDI_impulse (fast)
3. BETTER INTERPRETATION - Stability estimator vs event detector

Master Equation: ρ̇ = F_q(ρ,c) + Σκ_i G(...) - D(ρ; λ, φ)

Key Insight:
- Λ-FDI_state = structural stability estimator (smooth, regime diagnosis)
- FDI_impulse = event detector (sharp, timing triggers)
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
    """Load all intraday stock data."""
    print("📦 Loading intraday data...")
    
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    stock_data = {}
    for f in all_files:
        ticker = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, parse_dates=['date'])
            if len(df) > 1000:
                stock_data[ticker] = df
        except:
            continue
    
    print(f"   ✅ Loaded {len(stock_data)} stocks")
    return stock_data


def aggregate_to_daily(stock_data):
    """Aggregate intraday data to daily."""
    print("\n📊 Aggregating to daily...")
    
    daily_data = []
    
    for ticker, df in stock_data.items():
        df = df.sort_values('date')
        
        daily = df.groupby(df['date'].dt.date).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        daily.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        daily['date'] = pd.to_datetime(daily['date'])
        daily['return'] = daily['close'].pct_change()
        daily['ticker'] = ticker
        daily_data.append(daily)
    
    all_daily = pd.concat(daily_data, ignore_index=True)
    
    market = all_daily.groupby('date').agg({
        'return': 'mean',
        'volume': 'sum',
        'open': 'mean',
        'high': 'mean',
        'low': 'mean',
        'close': 'mean'
    }).sort_index()
    
    print(f"   ✅ {len(market)} trading days")
    return market


# =============================================================================
# BASE FDI (Event Detector)
# =============================================================================

def compute_base_fdi(market, window=20):
    """
    Base FDI: Raw feedback index (sharp, event detector)
    This is FDI_impulse - responds quickly to stress transitions.
    """
    print("\n📊 Computing Base FDI (Impulse Detector)...")
    
    # Parkinson volatility
    market['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(market['high'] / market['low']) ** 2)
    )
    
    # Amihud illiquidity
    market['amihud'] = np.abs(market['return']) / (market['volume'] * market['close'] / 1e7 + 1e-10)
    
    # Raw FDI
    market['FDI_raw'] = market['parkinson_vol'] / (market['amihud'] + 1e-10)
    
    # Smooth slightly (but keep responsive)
    market['FDI_impulse'] = market['FDI_raw'].rolling(5).mean()  # 5-day for responsiveness
    
    # Z-score
    market['FDI_impulse_zscore'] = (
        (market['FDI_impulse'] - market['FDI_impulse'].rolling(252).mean()) /
        market['FDI_impulse'].rolling(252).std()
    )
    
    print(f"   ✅ FDI_impulse computed (5-day smoothing)")
    
    return market


# =============================================================================
# DISSIPATION (FIXED)
# =============================================================================

def compute_dissipation_fixed(market, alpha=0.3, beta=0.5, gamma=0.1):
    """
    FIXED Dissipation: D(ρ) = αλρ + βφρ + γρ²
    
    Key fixes:
    1. Normalize ρ to z-score before any computation
    2. Cap the quadratic term to prevent explosion
    3. Make all terms comparable in scale
    """
    print("\n📊 Computing Dissipation D(ρ) [FIXED]...")
    
    # CRITICAL: Use z-scored FDI, not raw
    fdi_z = market['FDI_impulse_zscore'].fillna(0)
    
    # 1. Persistence drag: autocorrelation effect
    persistence_drag = alpha * fdi_z.shift(1).fillna(0)
    
    # 2. Shock dissipation: rate of change
    fdi_change = fdi_z.diff().rolling(5).mean().fillna(0)
    shock_dissipation = beta * (-fdi_change)
    
    # 3. Crowding: CAPPED quadratic
    # Cap at 3 std to prevent explosion
    fdi_capped = fdi_z.clip(-3, 3)
    crowding = gamma * (fdi_capped ** 2)
    
    # Store components (all now in similar scale)
    market['D_persistence'] = persistence_drag
    market['D_shock'] = shock_dissipation
    market['D_crowding'] = crowding
    market['D_total'] = persistence_drag + shock_dissipation + crowding
    
    print(f"   ✅ Dissipation (FIXED) computed")
    print(f"      Persistence: mean = {persistence_drag.mean():.4f}, std = {persistence_drag.std():.4f}")
    print(f"      Shock: mean = {shock_dissipation.mean():.4f}, std = {shock_dissipation.std():.4f}")
    print(f"      Crowding: mean = {crowding.mean():.4f}, std = {crowding.std():.4f}")
    
    return market


# =============================================================================
# FEEDBACK PRESSURE
# =============================================================================

def compute_feedback_pressure(market, kappa_trend=0.6, kappa_revert=0.4):
    """Feedback pressure: G+ (trend) - G- (revert)"""
    print("\n📊 Computing Feedback Pressure G(x,y)...")
    
    returns = market['return'].fillna(0)
    fdi_z = market['FDI_impulse_zscore'].fillna(0)
    
    # Momentum (z-scored)
    momentum = returns.rolling(20).mean()
    momentum_z = (momentum - momentum.rolling(252).mean()) / (momentum.rolling(252).std() + 1e-10)
    
    # Reversion signal
    short_ma = returns.rolling(5).mean()
    long_ma = returns.rolling(20).mean()
    reversion = -(short_ma - long_ma)
    reversion_z = (reversion - reversion.rolling(252).mean()) / (reversion.rolling(252).std() + 1e-10)
    
    # G+ and G-
    G_plus = (fdi_z * momentum_z.clip(0, None) * kappa_trend).rolling(5).mean()
    G_minus = (fdi_z * reversion_z.clip(0, None) * kappa_revert).rolling(5).mean()
    
    market['G_trend'] = G_plus.fillna(0)
    market['G_revert'] = G_minus.fillna(0)
    market['G_net'] = (G_plus - G_minus).fillna(0)
    
    trend_pct = (G_plus.abs() > G_minus.abs()).mean()
    print(f"   ✅ Feedback pressure computed")
    print(f"      Trend-dominated: {trend_pct:.1%}")
    
    return market


# =============================================================================
# REGIME DYNAMICS
# =============================================================================

def compute_regime_dynamics(market):
    """Regime dynamics F_q(ρ,c)"""
    print("\n📊 Computing Regime Dynamics F_q(ρ,c)...")
    
    fdi_z = market['FDI_impulse_zscore'].fillna(0)
    
    # Linear regime flow (slow-moving average)
    L_q = fdi_z.rolling(60).mean()  # Longer window for structural
    
    # Nonlinear amplification (acceleration when extreme)
    fdi_accel = fdi_z.diff().diff().clip(-1, 1)
    N_q = fdi_z * (1 + 0.2 * fdi_accel)
    
    # Smoothing
    fdi_smooth = pd.Series(gaussian_filter1d(fdi_z.values, sigma=5), index=fdi_z.index)
    
    market['F_linear'] = L_q.fillna(0)
    market['F_nonlinear'] = N_q.fillna(0)
    market['F_regime'] = (0.6 * L_q + 0.4 * N_q).fillna(0)
    
    print(f"   ✅ Regime dynamics computed")
    
    return market


# =============================================================================
# DUAL OUTPUT: Λ-FDI_state + FDI_impulse
# =============================================================================

def compute_dual_fdi(market):
    """
    Create two complementary signals:
    
    1. Λ-FDI_state: Slow, smooth, structural stability estimator
       - For regime diagnosis
       - Won't correlate strongly with returns (by design)
       
    2. FDI_impulse: Fast, sharp, event detector
       - For timing triggers
       - Strong correlation with forward returns
    """
    print("\n📊 Computing Dual FDI Signals...")
    
    # Get components (all z-scored now)
    F = market['F_regime'].fillna(0)
    G = market['G_net'].fillna(0)
    D = market['D_total'].fillna(0)
    
    # ===== Λ-FDI_state: Stability Estimator =====
    # Slower, more structural, emphasizes regime
    # Purpose: "Is the system stable or unstable?"
    
    # Weights: more regime, less feedback/dissipation
    w_f, w_g, w_d = 0.5, 0.3, 0.2
    
    lambda_state = w_f * F + w_g * G - w_d * D
    
    # Extra smoothing for structural signal
    lambda_state_smooth = lambda_state.rolling(20).mean()
    
    market['Lambda_FDI_state'] = lambda_state_smooth
    market['Lambda_FDI_state_zscore'] = (
        (lambda_state_smooth - lambda_state_smooth.rolling(252).mean()) /
        lambda_state_smooth.rolling(252).std()
    )
    
    # ===== FDI_impulse: Already computed =====
    # Fast, responsive (already in market['FDI_impulse_zscore'])
    
    # ===== Combined Signal =====
    # For those who want one number
    market['FDI_combined'] = 0.4 * market['Lambda_FDI_state_zscore'] + 0.6 * market['FDI_impulse_zscore']
    market['FDI_combined'] = market['FDI_combined'].fillna(0)
    
    print(f"   ✅ Dual signals computed:")
    print(f"      Λ-FDI_state (stability): mean={market['Lambda_FDI_state'].mean():.4f}")
    print(f"      FDI_impulse (events): mean={market['FDI_impulse_zscore'].mean():.4f}")
    print(f"      Combined: mean={market['FDI_combined'].mean():.4f}")
    
    return market


# =============================================================================
# COMPARISON
# =============================================================================

def compare_signals(market, window=10, drawdown_thresh=0.03):
    """Compare all FDI variants."""
    print("\n" + "=" * 70)
    print("  COMPARING FDI SIGNALS")
    print("=" * 70)
    
    df = market.copy()
    df['fwd_return'] = df['return'].rolling(window).sum().shift(-window)
    df['fwd_min'] = df['return'].rolling(window).min().shift(-window)
    df['had_drawdown'] = df['fwd_min'] < -drawdown_thresh
    
    variants = [
        ('FDI_impulse_zscore', 'FDI Impulse (fast)'),
        ('Lambda_FDI_state_zscore', 'Λ-FDI State (slow)'),
        ('FDI_combined', 'Combined'),
    ]
    
    results = []
    
    for col, name in variants:
        if col not in df.columns:
            continue
        
        df_clean = df[[col, 'had_drawdown', 'fwd_return']].dropna()
        
        for thresh in [1.0, 1.5, 2.0]:
            high = df_clean[df_clean[col] > thresh]
            if len(high) < 10:
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
    
    # Correlation comparison
    print("\n   📊 CORRELATION WITH FORWARD RETURNS:")
    print(f"   {'Signal':<25} {'5d':>10} {'10d':>10}")
    print("   " + "-" * 45)
    
    df['fwd_5d'] = df['return'].rolling(5).sum().shift(-5)
    df['fwd_10d'] = df['return'].rolling(10).sum().shift(-10)
    
    for col, name in variants:
        if col in df.columns:
            c5 = df[col].corr(df['fwd_5d'])
            c10 = df[col].corr(df['fwd_10d'])
            print(f"   {name:<25} {c5:>+10.4f} {c10:>+10.4f}")
    
    return results_df


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_dual_fdi(market):
    """Visualize dual FDI signals."""
    print("\n📊 Creating visualizations...")
    
    df = market.dropna(subset=['Lambda_FDI_state_zscore', 'FDI_impulse_zscore']).copy()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Dual signals comparison
    ax1 = axes[0]
    ax1.plot(df.index, df['FDI_impulse_zscore'], 'b-', alpha=0.7, label='FDI Impulse (fast)', linewidth=1)
    ax1.plot(df.index, df['Lambda_FDI_state_zscore'], 'r-', alpha=0.9, label='Λ-FDI State (slow)', linewidth=1.5)
    ax1.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Z-Score')
    ax1.set_title('Dual FDI: Impulse (Event Detector) vs State (Stability Estimator)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Dissipation components (now properly scaled)
    ax2 = axes[1]
    ax2.plot(df.index, df['D_persistence'], 'r-', alpha=0.6, label='Persistence', linewidth=1)
    ax2.plot(df.index, df['D_shock'], 'b-', alpha=0.6, label='Shock', linewidth=1)
    ax2.plot(df.index, df['D_crowding'], 'g-', alpha=0.6, label='Crowding', linewidth=1)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Dissipation')
    ax2.set_title('Dissipation Components (Fixed Scaling)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feedback pressure
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df['G_trend'], where=df['G_trend'] > 0,
                     color='green', alpha=0.5, label='G+ (Trend)')
    ax3.fill_between(df.index, 0, -df['G_revert'].abs(), where=df['G_revert'] > 0,
                     color='red', alpha=0.5, label='G- (Revert)')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Feedback G')
    ax3.set_title('Feedback Pressure: Trend vs Reversion', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Market with signals
    ax4 = axes[3]
    cumret = (1 + df['return']).cumprod()
    ax4.plot(df.index, cumret, 'k-', linewidth=1)
    
    # Highlight high impulse (red) and high state (orange)
    high_impulse = df['FDI_impulse_zscore'] > 2
    high_state = df['Lambda_FDI_state_zscore'] > 1.5
    
    for i in range(len(df)):
        if high_impulse.iloc[i]:
            ax4.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], alpha=0.4, color='red')
        elif high_state.iloc[i]:
            ax4.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], alpha=0.2, color='orange')
    
    ax4.set_ylabel('Cumulative Return')
    ax4.set_xlabel('Date')
    ax4.set_title('Market (Red=High Impulse, Orange=High State)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lambda_fdi_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/lambda_fdi_v2.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Λ-FDI v2: FIXED DUAL-OUTPUT FRAMEWORK")
    print("=" * 70)
    print("\n   Key improvements:")
    print("   • Fixed dissipation scaling (no more explosions)")
    print("   • Dual output: FDI_impulse + Λ-FDI_state")
    print("   • Proper interpretation: event detector vs stability estimator")
    
    # Load data
    stock_data = load_intraday_data()
    if len(stock_data) == 0:
        print("❌ No stock data found")
        return None
    
    market = aggregate_to_daily(stock_data)
    
    # Compute signals
    market = compute_base_fdi(market)
    market = compute_dissipation_fixed(market)
    market = compute_feedback_pressure(market)
    market = compute_regime_dynamics(market)
    market = compute_dual_fdi(market)
    
    # Save
    market.to_csv('lambda_fdi_v2_output.csv')
    print(f"\n   ✅ Saved: lambda_fdi_v2_output.csv")
    
    # Compare
    results = compare_signals(market)
    
    # Visualize
    visualize_dual_fdi(market)
    
    # Summary
    print("\n" + "=" * 70)
    print("  Λ-FDI v2 SUMMARY")
    print("=" * 70)
    
    print("\n   DUAL OUTPUT FRAMEWORK:")
    print("   ┌─────────────────────────────────────────────────────────┐")
    print("   │ FDI_impulse     │ Event detector   │ Use for: timing   │")
    print("   │ Λ-FDI_state     │ Stability estim. │ Use for: regime   │")
    print("   │ Combined        │ Ensemble         │ Use for: balanced │")
    print("   └─────────────────────────────────────────────────────────┘")
    
    print("\n   INTERPRETATION:")
    print("   • FDI_impulse > 2.0 → Sharp stress event → REDUCE POSITION")
    print("   • Λ-FDI_state > 1.5 → Structural instability → MONITOR")
    print("   • Both high → CRISIS MODE → EXIT")
    
    return market


if __name__ == "__main__":
    data = main()
