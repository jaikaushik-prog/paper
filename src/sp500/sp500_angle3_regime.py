"""
S&P 500 Angle 3: Regime Dependence Analysis
Tests whether market efficiency varies between Calm and Stress regimes.

Hypothesis: If the inefficiency is a "crisis artifact", it should only appear
during high-volatility (stress) periods. If it's structural, it should persist
across all regimes.

Regime Definition:
- CALM: Bottom 20% of rolling volatility days
- STRESS: Top 20% of rolling volatility days
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
DATA_FILE = r"c:\Users\DELL\Desktop\project_nifty_liquid\datasets\data\all_stocks_5yr.csv"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

# Rolling window for volatility calculation
VOL_WINDOW = 20  # 20 trading days

def load_data():
    """Load S&P 500 data."""
    print("Loading S&P 500 data...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df

def compute_market_volatility(df):
    """
    Compute market-wide volatility using SPY-like aggregate.
    Since we don't have SPY, we use equal-weighted average volatility.
    """
    print("\nComputing market volatility regimes...")
    
    # Calculate individual stock volatility
    all_vols = []
    
    for symbol in df['Name'].unique():
        stock = df[df['Name'] == symbol].copy().sort_values('date')
        
        if len(stock) < 50:
            continue
        
        stock['ret'] = np.log(stock['close'] / stock['close'].shift(1))
        stock['vol'] = stock['ret'].rolling(VOL_WINDOW).std() * np.sqrt(252)  # Annualized
        
        stock = stock.dropna()
        all_vols.append(stock[['date', 'vol']])
    
    # Aggregate to market volatility
    vol_df = pd.concat(all_vols)
    market_vol = vol_df.groupby('date')['vol'].mean().reset_index()
    market_vol.columns = ['date', 'market_vol']
    
    # Define regimes based on percentiles
    vol_20 = market_vol['market_vol'].quantile(0.20)
    vol_80 = market_vol['market_vol'].quantile(0.80)
    
    market_vol['regime'] = 'normal'
    market_vol.loc[market_vol['market_vol'] <= vol_20, 'regime'] = 'calm'
    market_vol.loc[market_vol['market_vol'] >= vol_80, 'regime'] = 'stress'
    
    print(f"\nVolatility Thresholds:")
    print(f"  Calm (Bottom 20%): Vol <= {vol_20:.2%}")
    print(f"  Stress (Top 20%): Vol >= {vol_80:.2%}")
    print(f"\nRegime Distribution:")
    print(market_vol['regime'].value_counts())
    
    return market_vol

def compute_regime_surfaces(df, market_vol):
    """
    Compute Push-Response surfaces for each regime.
    """
    print("\n" + "="*60)
    print("COMPUTING REGIME-DEPENDENT SURFACES")
    print("="*60)
    
    # Load Whales and Minnows
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_minnows.csv"))
    
    results = []
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        print(f"\nProcessing {group_name}...")
        
        for regime in ['calm', 'stress']:
            print(f"  Regime: {regime.upper()}")
            
            regime_dates = set(market_vol[market_vol['regime'] == regime]['date'].tolist())
            
            all_data = []
            
            for symbol in stock_list['symbol'].tolist():
                stock = df[df['Name'] == symbol].copy().sort_values('date')
                
                if len(stock) < 100:
                    continue
                
                # Filter to regime dates
                stock = stock[stock['date'].isin(regime_dates)]
                
                if len(stock) < 30:
                    continue
                
                # Returns
                stock['ret'] = np.log(stock['close'] / stock['close'].shift(1))
                
                # Z-score of returns
                stock['z_ret'] = (stock['ret'] - stock['ret'].rolling(20).mean()) / stock['ret'].rolling(20).std()
                
                # Push (lag 5 days within regime)
                stock['z_push'] = stock['z_ret'].rolling(5).sum()
                
                # Response (next 5 days)
                stock['z_resp'] = stock['z_ret'].shift(-5).rolling(5).sum()
                
                stock = stock.dropna()
                
                if len(stock) > 10:
                    all_data.append(stock[['z_push', 'z_resp']])
            
            if not all_data:
                continue
            
            combined = pd.concat(all_data, ignore_index=True)
            
            # Crash response (Z < -2)
            crash_data = combined[combined['z_push'] < -2]
            if len(crash_data) > 10:
                crash_resp = crash_data['z_resp'].mean()
                crash_std = crash_data['z_resp'].std()
                crash_count = len(crash_data)
                t_stat = crash_resp / (crash_std / np.sqrt(crash_count))
                
                results.append({
                    'group': group_name,
                    'regime': regime,
                    'crash_response': crash_resp,
                    'crash_std': crash_std,
                    'count': crash_count,
                    't_stat': t_stat
                })
                
                print(f"    Crash Response = {crash_resp:+.4f} ± {crash_std:.4f} "
                      f"(n={crash_count}, t={t_stat:.2f})")
            
            # Rally response (Z > 2)
            rally_data = combined[combined['z_push'] > 2]
            if len(rally_data) > 10:
                rally_resp = rally_data['z_resp'].mean()
                
                results.append({
                    'group': group_name,
                    'regime': regime + '_rally',
                    'crash_response': rally_resp,
                    'crash_std': rally_data['z_resp'].std(),
                    'count': len(rally_data),
                    't_stat': rally_resp / (rally_data['z_resp'].std() / np.sqrt(len(rally_data)))
                })
    
    return pd.DataFrame(results)

def analyze_regime_results(results_df):
    """Analyze and interpret regime dependence results."""
    print("\n" + "="*60)
    print("REGIME DEPENDENCE ANALYSIS")
    print("="*60)
    
    # Filter to crash responses only
    crash_results = results_df[~results_df['regime'].str.contains('rally')].copy()
    
    # Pivot for easy comparison
    pivot = crash_results.pivot(index='group', columns='regime', values='crash_response')
    
    print("\n=== CRASH RESPONSE BY REGIME ===")
    print(f"\n{'Group':<12} {'CALM':<15} {'STRESS':<15} {'Ratio':<10}")
    print("-"*55)
    
    for group in ['whales', 'minnows']:
        if group in pivot.index:
            calm = pivot.loc[group, 'calm'] if 'calm' in pivot.columns else np.nan
            stress = pivot.loc[group, 'stress'] if 'stress' in pivot.columns else np.nan
            ratio = stress / calm if calm != 0 and not np.isnan(calm) else np.nan
            print(f"{group.capitalize():<12} {calm:+.4f}        {stress:+.4f}        {ratio:.2f}x")
    
    # Analysis
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if 'calm' in pivot.columns and 'stress' in pivot.columns:
        whale_calm = pivot.loc['whales', 'calm'] if 'whales' in pivot.index else 0
        minnow_calm = pivot.loc['minnows', 'calm'] if 'minnows' in pivot.index else 0
        whale_stress = pivot.loc['whales', 'stress'] if 'whales' in pivot.index else 0
        minnow_stress = pivot.loc['minnows', 'stress'] if 'minnows' in pivot.index else 0
        
        # Check if response exists in both regimes
        calm_ratio = abs(minnow_calm / whale_calm) if whale_calm != 0 else 0
        stress_ratio = abs(minnow_stress / whale_stress) if whale_stress != 0 else 0
        
        print(f"\nMinnow/Whale Inefficiency Ratio:")
        print(f"  CALM regime:   {calm_ratio:.2f}x")
        print(f"  STRESS regime: {stress_ratio:.2f}x")
        
        if abs(calm_ratio - stress_ratio) < 0.5:
            print("\n✓ STRUCTURAL INEFFICIENCY: Pattern persists across regimes")
            print("  The inefficiency is NOT a crisis artifact")
        else:
            print("\n≈ REGIME-DEPENDENT: Pattern varies by market conditions")
    
    return crash_results

def create_regime_plots(results_df, market_vol):
    """Create visualization plots."""
    print("\nGenerating plots...")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Filter to crash responses
    crash_results = results_df[~results_df['regime'].str.contains('rally')]
    
    # Bar chart comparison
    fig = go.Figure()
    
    for regime in ['calm', 'stress']:
        regime_data = crash_results[crash_results['regime'] == regime]
        
        fig.add_trace(go.Bar(
            x=regime_data['group'],
            y=regime_data['crash_response'],
            name=regime.upper(),
            error_y=dict(type='data', array=regime_data['crash_std'] / np.sqrt(regime_data['count']))
        ))
    
    fig.update_layout(
        title="S&P 500 Crash Response: Calm vs Stress Regimes",
        xaxis_title="Liquidity Group",
        yaxis_title="Average Response (Z-score)",
        barmode='group',
        width=700,
        height=500
    )
    
    fig.write_html(os.path.join(PLOTS_DIR, "sp500_angle3_regime.html"))
    print(f"  Saved: sp500_angle3_regime.html")
    
    # Volatility time series
    fig2 = go.Figure()
    
    colors = {'calm': 'green', 'normal': 'gray', 'stress': 'red'}
    for regime in ['calm', 'normal', 'stress']:
        regime_data = market_vol[market_vol['regime'] == regime]
        fig2.add_trace(go.Scatter(
            x=regime_data['date'],
            y=regime_data['market_vol'],
            mode='markers',
            marker=dict(size=3, color=colors[regime]),
            name=regime.upper()
        ))
    
    fig2.update_layout(
        title="S&P 500 Market Volatility Regimes (2013-2018)",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        width=900,
        height=400
    )
    
    fig2.write_html(os.path.join(PLOTS_DIR, "sp500_angle3_volatility.html"))
    print(f"  Saved: sp500_angle3_volatility.html")

def main():
    # Load data
    df = load_data()
    
    print("\n" + "="*70)
    print("ANGLE 3: REGIME DEPENDENCE - S&P 500")
    print("="*70)
    
    # 1. Compute market volatility and regimes
    market_vol = compute_market_volatility(df)
    
    # 2. Compute surfaces for each regime
    results_df = compute_regime_surfaces(df, market_vol)
    
    # 3. Analyze results
    crash_results = analyze_regime_results(results_df)
    
    # 4. Create plots
    create_regime_plots(results_df, market_vol)
    
    # 5. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "sp500_angle3_regimes.csv"), index=False)
    market_vol.to_csv(os.path.join(RESULTS_DIR, "sp500_market_volatility.csv"), index=False)
    
    print("\n" + "="*70)
    print("ANGLE 3 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
