"""
S&P 500 Angle 2: Volume Time Analysis
Tests the Volume Clock Hypothesis using daily data.

Adaptation for Daily Data:
- Instead of constructing volume-time bars (requires intraday), we analyze:
  1. Volume-normalized returns: ret / sqrt(relative_volume)
  2. Kurtosis comparison: Clock Time vs Volume-normalized
  3. Volume regime stratification: High vs Low volume day patterns
  4. Push-Response surfaces conditioned on volume
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

def load_data():
    """Load S&P 500 data."""
    print("Loading S&P 500 data...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    return df

def compute_volume_normalized_returns(df):
    """
    Compute volume-normalized returns for each stock.
    
    Theory: If Volume Time recovers normality, then:
    ret_voltime = ret_clocktime / sqrt(relative_volume)
    should have lower kurtosis.
    """
    print("\nComputing volume-normalized returns...")
    
    results = []
    
    for symbol in df['Name'].unique():
        stock = df[df['Name'] == symbol].copy().sort_values('date')
        
        if len(stock) < 100:
            continue
        
        # Clock time returns (log returns)
        stock['ret'] = np.log(stock['close'] / stock['close'].shift(1))
        
        # Rolling average volume (20-day)
        stock['avg_volume'] = stock['volume'].rolling(20).mean()
        
        # Relative volume
        stock['rel_volume'] = stock['volume'] / stock['avg_volume']
        
        # Volume-normalized return
        # Intuition: High volume days should have "faster" price discovery
        stock['ret_volnorm'] = stock['ret'] / np.sqrt(stock['rel_volume'].clip(lower=0.1))
        
        stock = stock.dropna()
        
        if len(stock) < 50:
            continue
        
        # Calculate kurtosis
        clock_kurt = stats.kurtosis(stock['ret'].dropna(), fisher=True)
        vol_kurt = stats.kurtosis(stock['ret_volnorm'].dropna(), fisher=True)
        
        results.append({
            'symbol': symbol,
            'clock_kurtosis': clock_kurt,
            'vol_kurtosis': vol_kurt,
            'n_obs': len(stock),
            'avg_rel_vol': stock['rel_volume'].mean()
        })
    
    return pd.DataFrame(results)

def analyze_kurtosis_impact(kurt_df):
    """Analyze whether volume normalization reduces kurtosis."""
    print("\n" + "="*60)
    print("KURTOSIS ANALYSIS: Clock Time vs Volume Time")
    print("="*60)
    
    # Filter extreme outliers
    kurt_df = kurt_df[(kurt_df['clock_kurtosis'] < 100) & (kurt_df['vol_kurtosis'] < 100)].copy()
    
    clock_mean = kurt_df['clock_kurtosis'].mean()
    vol_mean = kurt_df['vol_kurtosis'].mean()
    
    improvement = (clock_mean - vol_mean) / clock_mean * 100
    
    print(f"\nNumber of stocks analyzed: {len(kurt_df)}")
    print(f"\nClock Time Kurtosis (mean): {clock_mean:.2f}")
    print(f"Volume Time Kurtosis (mean): {vol_mean:.2f}")
    print(f"\nImprovement: {improvement:+.1f}%")
    
    # Count how many stocks improved
    kurt_df['improved'] = kurt_df['vol_kurtosis'] < kurt_df['clock_kurtosis']
    pct_improved = kurt_df['improved'].mean() * 100
    
    print(f"Stocks with lower Vol Kurtosis: {pct_improved:.1f}%")
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(kurt_df['clock_kurtosis'], kurt_df['vol_kurtosis'])
    print(f"\nPaired t-test: t={t_stat:.2f}, p={p_value:.4f}")
    
    return {
        'clock_mean': clock_mean,
        'vol_mean': vol_mean,
        'improvement': improvement,
        'pct_improved': pct_improved,
        't_stat': t_stat,
        'p_value': p_value
    }

def compute_volume_regime_surfaces(df):
    """
    Compute Push-Response surfaces for High vs Low volume regimes.
    Tests if efficiency differs by trading activity.
    """
    print("\n" + "="*60)
    print("VOLUME REGIME ANALYSIS")
    print("="*60)
    
    # Load Whales and Minnows
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_minnows.csv"))
    
    results = []
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        print(f"\nProcessing {group_name}...")
        
        all_data = []
        
        for symbol in stock_list['symbol'].tolist():
            stock = df[df['Name'] == symbol].copy().sort_values('date')
            
            if len(stock) < 100:
                continue
            
            # Returns
            stock['ret'] = np.log(stock['close'] / stock['close'].shift(1))
            
            # Volume regime (relative to 20-day average)
            stock['avg_volume'] = stock['volume'].rolling(20).mean()
            stock['rel_volume'] = stock['volume'] / stock['avg_volume']
            
            # Z-score of returns (standardized)
            stock['z_ret'] = (stock['ret'] - stock['ret'].rolling(60).mean()) / stock['ret'].rolling(60).std()
            
            # Push (lag 5 days)
            stock['z_push'] = stock['z_ret'].rolling(5).sum()
            
            # Response (next 5 days)
            stock['z_resp'] = stock['z_ret'].shift(-5).rolling(5).sum()
            
            # Volume regime classification
            stock['vol_regime'] = pd.cut(
                stock['rel_volume'], 
                bins=[0, 0.8, 1.2, np.inf], 
                labels=['low', 'normal', 'high']
            )
            
            stock = stock.dropna()
            all_data.append(stock[['z_push', 'z_resp', 'vol_regime']])
        
        if not all_data:
            continue
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Analyze by volume regime
        for regime in ['low', 'high']:
            regime_data = combined[combined['vol_regime'] == regime]
            
            if len(regime_data) < 100:
                continue
            
            # Crash response (Z < -2)
            crash_data = regime_data[regime_data['z_push'] < -2]
            if len(crash_data) > 10:
                crash_resp = crash_data['z_resp'].mean()
                crash_count = len(crash_data)
                
                results.append({
                    'group': group_name,
                    'regime': regime,
                    'crash_response': crash_resp,
                    'count': crash_count
                })
                
                print(f"  {group_name.capitalize()} - {regime.upper()} volume: "
                      f"Crash Response = {crash_resp:+.4f} (n={crash_count})")
    
    return pd.DataFrame(results)

def create_volume_analysis_plots(kurt_df, regime_df):
    """Create visualization plots for Volume Time analysis."""
    print("\nGenerating plots...")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Kurtosis comparison scatter plot
    fig1 = go.Figure()
    
    # Filter for visualization
    plot_df = kurt_df[(kurt_df['clock_kurtosis'] < 50) & (kurt_df['vol_kurtosis'] < 50)]
    
    fig1.add_trace(go.Scatter(
        x=plot_df['clock_kurtosis'],
        y=plot_df['vol_kurtosis'],
        mode='markers',
        marker=dict(size=6, opacity=0.6),
        name='Stocks'
    ))
    
    # Add 45-degree line
    max_val = max(plot_df['clock_kurtosis'].max(), plot_df['vol_kurtosis'].max())
    fig1.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='No Change Line'
    ))
    
    fig1.update_layout(
        title="S&P 500 Kurtosis: Clock Time vs Volume Time",
        xaxis_title="Clock Time Kurtosis",
        yaxis_title="Volume Time Kurtosis",
        width=700,
        height=600
    )
    
    fig1.write_html(os.path.join(PLOTS_DIR, "sp500_angle2_kurtosis.html"))
    print(f"  Saved: sp500_angle2_kurtosis.html")
    
    # 2. Volume regime comparison bar chart
    if not regime_df.empty:
        fig2 = go.Figure()
        
        for group in ['whales', 'minnows']:
            group_data = regime_df[regime_df['group'] == group]
            
            fig2.add_trace(go.Bar(
                x=group_data['regime'],
                y=group_data['crash_response'],
                name=group.capitalize()
            ))
        
        fig2.update_layout(
            title="S&P 500 Crash Response by Volume Regime",
            xaxis_title="Volume Regime",
            yaxis_title="Average Response (Z-score)",
            barmode='group',
            width=700,
            height=500
        )
        
        fig2.write_html(os.path.join(PLOTS_DIR, "sp500_angle2_volume_regime.html"))
        print(f"  Saved: sp500_angle2_volume_regime.html")

def main():
    # Load data
    df = load_data()
    
    # 1. Kurtosis Analysis
    print("\n" + "="*70)
    print("ANGLE 2: VOLUME TIME ANALYSIS - S&P 500")
    print("="*70)
    
    kurt_df = compute_volume_normalized_returns(df)
    kurt_results = analyze_kurtosis_impact(kurt_df)
    
    # 2. Volume Regime Surfaces
    regime_df = compute_volume_regime_surfaces(df)
    
    # 3. Create plots
    create_volume_analysis_plots(kurt_df, regime_df)
    
    # 4. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    kurt_df.to_csv(os.path.join(RESULTS_DIR, "sp500_angle2_kurtosis.csv"), index=False)
    if not regime_df.empty:
        regime_df.to_csv(os.path.join(RESULTS_DIR, "sp500_angle2_regimes.csv"), index=False)
    
    # Summary
    print("\n" + "="*70)
    print("ANGLE 2 SUMMARY")
    print("="*70)
    print(f"\nKurtosis Reduction: {kurt_results['improvement']:+.1f}%")
    print(f"Stocks Improved: {kurt_results['pct_improved']:.1f}%")
    print(f"Statistical Significance: p = {kurt_results['p_value']:.4f}")
    
    if kurt_results['p_value'] < 0.05 and kurt_results['improvement'] > 0:
        print("\n✓ Volume Time SUCCEEDS in reducing fat tails")
    else:
        print("\n✗ Volume Time shows NO significant improvement")
    
    print("\n" + "="*70)
    print("ANGLE 2 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
