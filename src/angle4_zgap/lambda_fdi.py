"""
Lambda-FDI (Λ-FDI): Advanced Feedback Dominance Framework
==========================================================

Based on the Λ-Vol framework (Quant Research Decoded), this upgrades FDI with:

1. DISSIPATION D(ρ) = αλρ + βφρ + γρ²
   - Persistence drag (how slow stress decays)
   - Shock dissipation (how fast shocks absorbed)
   - Crowding saturation (nonlinear when crowded)

2. FEEDBACK LOOPS G(x,y)
   - G+ = Trend-reinforcing pressure (momentum)
   - G- = Mean-reverting pressure (contrarian)
   - Net Pressure = G+ - G-

3. REGIME DYNAMICS F_q(ρ,c)
   - Baseline regime flow
   - Nonlinear amplification

Master Equation: ρ̇ = F_q(ρ,c) + Σκ_i G(...) - D(ρ; λ, φ)
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
    
    # Market aggregate
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
# CORE FDI COMPONENTS
# =============================================================================

def compute_base_fdi(market, window=20):
    """Compute base FDI (volatility/illiquidity)."""
    
    # Parkinson volatility
    market['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(market['high'] / market['low']) ** 2)
    )
    
    # Amihud illiquidity
    market['amihud'] = np.abs(market['return']) / (market['volume'] * market['close'] / 1e7 + 1e-10)
    
    # Base FDI
    market['FDI_base'] = market['parkinson_vol'] / (market['amihud'] + 1e-10)
    market['FDI_base_smooth'] = market['FDI_base'].rolling(window).mean()
    
    # Z-score
    market['FDI_base_zscore'] = (
        (market['FDI_base_smooth'] - market['FDI_base_smooth'].rolling(252).mean()) /
        market['FDI_base_smooth'].rolling(252).std()
    )
    
    return market


# =============================================================================
# Λ-FDI COMPONENTS
# =============================================================================

def compute_dissipation(market, alpha=0.3, beta=0.5, gamma=0.1):
    """
    Dissipation term: D(ρ) = αλρ + βφρ + γρ²
    
    - αλρ: Persistence drag (slow decay)
    - βφρ: Shock dissipation (fast absorption)
    - γρ²: Crowding saturation (nonlinear)
    """
    print("\n📊 Computing Dissipation D(ρ)...")
    
    fdi = market['FDI_base_smooth'].fillna(0)
    
    # 1. Persistence drag: How much of yesterday's FDI persists today
    # High persistence = slow decay = stress lingers
    fdi_lag = fdi.shift(1).fillna(0)
    persistence_drag = alpha * fdi_lag
    
    # 2. Shock dissipation: Rate of FDI decay
    # Positive = FDI is falling (dissipating)
    # Negative = FDI is rising (stress building)
    fdi_change = fdi.diff().rolling(5).mean().fillna(0)
    shock_dissipation = beta * (-fdi_change)  # Negative change = positive dissipation
    
    # 3. Crowding saturation: Nonlinear when FDI is very high
    # FIXED: Normalize by mean to avoid huge values
    fdi_normalized = fdi / (fdi.rolling(252).mean() + 1e-10)
    crowding = gamma * (fdi_normalized ** 2)
    
    # Total dissipation
    market['D_persistence'] = persistence_drag
    market['D_shock'] = shock_dissipation
    market['D_crowding'] = crowding
    market['D_total'] = persistence_drag + shock_dissipation + crowding
    
    print(f"   ✅ Dissipation terms computed")
    print(f"      Persistence drag (α={alpha}): mean = {persistence_drag.mean():.4f}")
    print(f"      Shock dissipation (β={beta}): mean = {shock_dissipation.mean():.4f}")
    print(f"      Crowding (γ={gamma}): mean = {crowding.mean():.4f}")
    
    return market


def compute_feedback_pressure(market, kappa_trend=0.6, kappa_revert=0.4):
    """
    Feedback loops: G = G+ (trend) - G- (revert)
    
    G+ = Trend-reinforcing pressure (when momentum aligns with FDI)
    G- = Mean-reverting pressure (when contrarian signal strong)
    """
    print("\n📊 Computing Feedback Pressure G(x,y)...")
    
    returns = market['return'].fillna(0)
    fdi = market['FDI_base_smooth'].fillna(0)
    
    # Momentum signal (20-day)
    momentum = returns.rolling(20).mean()
    momentum_strength = momentum / (returns.rolling(20).std() + 1e-10)
    
    # Mean reversion signal (5-day vs 20-day)
    short_ma = returns.rolling(5).mean()
    long_ma = returns.rolling(20).mean()
    reversion_signal = -(short_ma - long_ma)  # Negative when extended, positive when reverting
    
    # Directional masks
    is_up = returns > 0
    is_down = returns < 0
    
    # G+: Trend-reinforcing pressure
    # High when: FDI high + momentum strong + same direction
    G_plus = fdi * momentum_strength.clip(0, None) * kappa_trend
    G_plus = G_plus.rolling(5).mean()
    
    # G-: Mean-reverting pressure
    # High when: FDI high + reversion signal strong
    G_minus = fdi * reversion_signal.clip(0, None) * kappa_revert
    G_minus = G_minus.rolling(5).mean()
    
    # Net pressure
    market['G_trend'] = G_plus.fillna(0)
    market['G_revert'] = G_minus.fillna(0)
    market['G_net'] = (G_plus - G_minus).fillna(0)
    
    # Signed pressure (positive = trend, negative = revert)
    market['G_signed'] = np.where(G_plus > G_minus, G_plus, -G_minus)
    
    print(f"   ✅ Feedback pressure computed")
    print(f"      G+ (trend): mean = {G_plus.mean():.4f}")
    print(f"      G- (revert): mean = {G_minus.mean():.4f}")
    print(f"      Net: {(G_plus > G_minus).mean():.1%} trend-dominated")
    
    return market


def compute_regime_dynamics(market):
    """
    Regime dynamics: F_q(ρ,c) = L_q ρ + N_q(ρ,c) - η_q Δρ
    
    - L_q ρ: Linear regime flow
    - N_q(ρ,c): Nonlinear amplification
    - η_q Δρ: Spatial smoothing
    """
    print("\n📊 Computing Regime Dynamics F_q(ρ,c)...")
    
    fdi = market['FDI_base_smooth'].fillna(0)
    returns = market['return'].fillna(0)
    
    # 1. Linear regime flow: Rolling FDI level
    L_q = fdi.rolling(20).mean()
    
    # 2. Nonlinear amplification: FDI acceleration
    # When FDI is rising AND high, amplify
    fdi_velocity = fdi.diff()
    fdi_acceleration = fdi_velocity.diff()
    N_q = fdi * (1 + fdi_acceleration.clip(-0.5, 0.5))
    
    # 3. Spatial smoothing: Gaussian filter to remove noise
    fdi_smooth = pd.Series(
        gaussian_filter1d(fdi.values, sigma=3),
        index=fdi.index
    )
    eta_smooth = fdi - fdi_smooth
    
    # Combine
    market['F_linear'] = L_q.fillna(0)
    market['F_nonlinear'] = N_q.fillna(0)
    market['F_smoothing'] = eta_smooth.fillna(0)
    market['F_regime'] = (L_q + 0.2 * N_q - 0.1 * eta_smooth).fillna(0)
    
    print(f"   ✅ Regime dynamics computed")
    
    return market


def compute_lambda_fdi(market):
    """
    Master equation: Λ-FDI = F_q(ρ,c) + G_net - D_total
    
    This is the improved FDI that combines:
    - Regime dynamics (F)
    - Feedback pressure (G)
    - Dissipation (D)
    """
    print("\n📊 Computing Λ-FDI (Lambda-FDI)...")
    
    # Normalize components to same scale
    f_norm = (market['F_regime'] - market['F_regime'].mean()) / (market['F_regime'].std() + 1e-10)
    g_norm = (market['G_net'] - market['G_net'].mean()) / (market['G_net'].std() + 1e-10)
    d_norm = (market['D_total'] - market['D_total'].mean()) / (market['D_total'].std() + 1e-10)
    
    # Weights (tunable)
    w_f, w_g, w_d = 0.4, 0.4, 0.2
    
    # Λ-FDI = weighted combination
    market['Lambda_FDI'] = w_f * f_norm + w_g * g_norm - w_d * d_norm
    
    # Z-score for final signal
    market['Lambda_FDI_zscore'] = (
        (market['Lambda_FDI'] - market['Lambda_FDI'].rolling(252).mean()) /
        market['Lambda_FDI'].rolling(252).std()
    )
    
    print(f"   ✅ Λ-FDI computed")
    print(f"      Mean: {market['Lambda_FDI'].mean():.4f}")
    print(f"      Std: {market['Lambda_FDI'].std():.4f}")
    
    return market


def compute_hybrid_fdi(market, alpha=0.6, beta=0.4):
    """
    Hybrid FDI: Ensemble of Lambda-FDI (hit rate) and Base FDI (correlation)
    
    Hybrid = α * Lambda_FDI + β * Base_FDI
    """
    print("\n📊 Computing Hybrid FDI (Ensemble)...")
    
    # Ensure both are z-scored
    lambda_z = market['Lambda_FDI_zscore'].fillna(0)
    base_z = market['FDI_base_zscore'].fillna(0)
    
    # Weighted ensemble
    market['Hybrid_FDI'] = alpha * lambda_z + beta * base_z
    
    # Re-zscore the hybrid
    market['Hybrid_FDI_zscore'] = (
        (market['Hybrid_FDI'] - market['Hybrid_FDI'].rolling(252).mean()) /
        market['Hybrid_FDI'].rolling(252).std()
    )
    
    print(f"   ✅ Hybrid FDI = {alpha:.0%} Λ-FDI + {beta:.0%} Base FDI")
    print(f"      Mean: {market['Hybrid_FDI'].mean():.4f}")
    print(f"      Std: {market['Hybrid_FDI'].std():.4f}")
    
    return market


# =============================================================================
# COMPARISON & EVALUATION
# =============================================================================

def compare_fdi_variants(market, window=10, drawdown_thresh=0.03):
    """Compare Λ-FDI vs Base FDI vs Directional."""
    print("\n" + "=" * 70)
    print("  COMPARING FDI VARIANTS")
    print("=" * 70)
    
    df = market.copy()
    
    # Forward returns and drawdown
    df['fwd_return'] = df['return'].rolling(window).sum().shift(-window)
    df['fwd_min'] = df['return'].rolling(window).min().shift(-window)
    df['had_drawdown'] = df['fwd_min'] < -drawdown_thresh
    
    variants = [
        ('FDI_base_zscore', 'Base FDI'),
        ('Lambda_FDI_zscore', 'Λ-FDI'),
        ('Hybrid_FDI_zscore', 'Hybrid FDI'),
        ('G_net', 'Feedback Pressure'),
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
            
            hit_rate = high['had_drawdown'].mean()
            avg_ret = high['fwd_return'].mean()
            
            results.append({
                'variant': name,
                'threshold': thresh,
                'hit_rate': hit_rate,
                'avg_return': avg_ret,
                'n_events': len(high)
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n   {'Variant':<20} {'Thresh':>8} {'Hit Rate':>10} {'Avg Ret':>10} {'Events':>8}")
    print("   " + "-" * 60)
    
    for _, row in results_df.iterrows():
        print(f"   {row['variant']:<20} {row['threshold']:>8.1f} {row['hit_rate']:>9.1%} {row['avg_return']:>9.2%} {row['n_events']:>8}")
    
    # Find best per variant
    print("\n   📊 BEST PER VARIANT:")
    for variant in results_df['variant'].unique():
        subset = results_df[results_df['variant'] == variant]
        if len(subset) > 0:
            best = subset.loc[subset['hit_rate'].idxmax()]
            print(f"      {variant}: {best['hit_rate']:.1%} @ thresh={best['threshold']}")
    
    return results_df


def correlation_comparison(market):
    """Compare correlations with forward returns."""
    print("\n" + "=" * 70)
    print("  CORRELATION WITH FORWARD RETURNS")
    print("=" * 70)
    
    df = market.copy()
    df['fwd_5d'] = df['return'].rolling(5).sum().shift(-5)
    df['fwd_10d'] = df['return'].rolling(10).sum().shift(-10)
    
    variants = ['FDI_base_zscore', 'Lambda_FDI_zscore', 'Hybrid_FDI_zscore', 'G_net']
    
    print(f"\n   {'Metric':<25} {'5d Corr':>10} {'10d Corr':>10}")
    print("   " + "-" * 45)
    
    correlations = {}
    for v in variants:
        if v in df.columns:
            c5 = df[v].corr(df['fwd_5d'])
            c10 = df[v].corr(df['fwd_10d'])
            correlations[v] = {'5d': c5, '10d': c10}
            print(f"   {v:<25} {c5:>+10.4f} {c10:>+10.4f}")
    
    # Check if Λ-FDI beats Base
    if 'FDI_base_zscore' in correlations and 'Lambda_FDI_zscore' in correlations:
        base_corr = correlations['FDI_base_zscore']['10d']
        lambda_corr = correlations['Lambda_FDI_zscore']['10d']
        
        if abs(lambda_corr) > abs(base_corr):
            improvement = (abs(lambda_corr) - abs(base_corr)) / abs(base_corr + 1e-10) * 100
            print(f"\n   ✅ Λ-FDI correlation is {improvement:.1f}% stronger than Base FDI")
        else:
            print(f"\n   🟡 Base FDI has stronger correlation")
    
    return correlations


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_lambda_fdi(market):
    """Create visualization of Λ-FDI components."""
    print("\n📊 Creating visualizations...")
    
    df = market.dropna(subset=['Lambda_FDI_zscore']).copy()
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    
    # 1. Base FDI vs Λ-FDI
    ax1 = axes[0]
    ax1.plot(df.index, df['FDI_base_zscore'], 'b-', alpha=0.5, label='Base FDI', linewidth=1)
    ax1.plot(df.index, df['Lambda_FDI_zscore'], 'r-', alpha=0.8, label='Λ-FDI', linewidth=1.5)
    ax1.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-1.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Z-Score')
    ax1.set_title('Base FDI vs Λ-FDI (Lambda-FDI)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Regime Dynamics F_q
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, df['F_regime'], alpha=0.5, color='blue')
    ax2.set_ylabel('F_q (Regime)')
    ax2.set_title('Regime Dynamics: F_q(ρ,c)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Feedback Pressure G
    ax3 = axes[2]
    ax3.fill_between(df.index, 0, df['G_trend'], where=df['G_trend'] > 0, 
                     color='green', alpha=0.5, label='G+ (Trend)')
    ax3.fill_between(df.index, 0, -df['G_revert'], where=df['G_revert'] > 0, 
                     color='red', alpha=0.5, label='G- (Revert)')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('G (Pressure)')
    ax3.set_title('Feedback Pressure: G+ (trend) vs G- (revert)', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Dissipation D
    ax4 = axes[3]
    ax4.stackplot(df.index, 
                  df['D_persistence'], df['D_shock'], df['D_crowding'],
                  labels=['Persistence', 'Shock', 'Crowding'],
                  colors=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.7)
    ax4.set_ylabel('D (Dissipation)')
    ax4.set_title('Dissipation: D(ρ) = αλρ + βφρ + γρ²', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Market returns with Λ-FDI signals
    ax5 = axes[4]
    cumret = (1 + df['return']).cumprod()
    ax5.plot(df.index, cumret, 'k-', linewidth=1)
    
    # Highlight high Λ-FDI periods
    high_lambda = df['Lambda_FDI_zscore'] > 1.5
    for i in range(len(df)):
        if high_lambda.iloc[i]:
            ax5.axvspan(df.index[i], df.index[min(i+1, len(df)-1)], 
                       alpha=0.3, color='red')
    
    ax5.set_ylabel('Cumulative Return')
    ax5.set_xlabel('Date')
    ax5.set_title('Market Returns (Red = High Λ-FDI)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lambda_fdi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/lambda_fdi.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  Λ-FDI: LAMBDA FEEDBACK DOMINANCE INDEX")
    print("  Based on Λ-Vol Framework (Quant Research Decoded)")
    print("=" * 70)
    
    # Load and aggregate data
    stock_data = load_intraday_data()
    if len(stock_data) == 0:
        print("❌ No stock data found")
        return None
    
    market = aggregate_to_daily(stock_data)
    
    # Compute base FDI
    market = compute_base_fdi(market)
    
    # Add Λ-FDI components
    market = compute_dissipation(market)
    market = compute_feedback_pressure(market)
    market = compute_regime_dynamics(market)
    market = compute_lambda_fdi(market)
    market = compute_hybrid_fdi(market)  # NEW: Ensemble
    
    # Save output
    market.to_csv('lambda_fdi_output.csv')
    print(f"\n   ✅ Saved: lambda_fdi_output.csv")
    
    # Compare variants
    comparison = compare_fdi_variants(market)
    correlations = correlation_comparison(market)
    
    # Visualize
    visualize_lambda_fdi(market)
    
    # Summary
    print("\n" + "=" * 70)
    print("  Λ-FDI SUMMARY")
    print("=" * 70)
    
    print("\n   Components Implemented:")
    print("   ✅ F_q(ρ,c): Regime dynamics (linear + nonlinear + smoothing)")
    print("   ✅ G(x,y): Feedback pressure (trend + revert)")
    print("   ✅ D(ρ): Dissipation (persistence + shock + crowding)")
    print("   ✅ Λ-FDI: Combined signal")
    
    print("\n   Master Equation:")
    print("   ρ̇ = F_q(ρ,c) + Σκ_i G(...) - D(ρ; λ, φ)")
    
    # Check improvement
    if 'FDI_base_zscore' in correlations and 'Lambda_FDI_zscore' in correlations:
        base = abs(correlations['FDI_base_zscore']['10d'])
        lambda_c = abs(correlations['Lambda_FDI_zscore']['10d'])
        
        if lambda_c > base:
            print(f"\n   🏆 Λ-FDI OUTPERFORMS Base FDI")
            print(f"      Base correlation: {base:.4f}")
            print(f"      Λ-FDI correlation: {lambda_c:.4f}")
            print(f"      Improvement: {(lambda_c/base - 1)*100:.1f}%")
            return True, market
        else:
            print(f"\n   🟡 Base FDI still better (consider tuning)")
            return False, market
    
    return None, market


if __name__ == "__main__":
    success, data = main()
