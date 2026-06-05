"""
S&P 500 Angle 4: Z-Gap Analysis
Analyzes overnight/day-to-day gaps and tests Fade vs Follow hypothesis.

For daily data, "gaps" are defined as:
- Gap = Open(t) / Close(t-1) - 1
- This measures the overnight return (after-hours + pre-market)

Hypothesis:
- FADE: Gaps reverse during the trading day (mean reversion)
- FOLLOW: Gaps continue during the trading day (momentum)

We test if Whales and Minnows respond differently to gaps.
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

def compute_gap_analysis(df):
    """
    Compute gap and intraday response for each stock.
    
    Gap = Open(t) / Close(t-1) - 1 (overnight return)
    Intraday = Close(t) / Open(t) - 1 (same-day return)
    """
    print("\nComputing gap analysis...")
    
    # Load Whales and Minnows
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_minnows.csv"))
    
    results = []
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        print(f"\nProcessing {group_name}...")
        
        all_gaps = []
        
        for symbol in stock_list['symbol'].tolist():
            stock = df[df['Name'] == symbol].copy().sort_values('date')
            
            if len(stock) < 100:
                continue
            
            # Gap calculation
            stock['prev_close'] = stock['close'].shift(1)
            stock['gap'] = (stock['open'] / stock['prev_close'] - 1) * 100  # In percentage
            
            # Intraday return
            stock['intraday'] = (stock['close'] / stock['open'] - 1) * 100
            
            # Full day return
            stock['full_day'] = (stock['close'] / stock['prev_close'] - 1) * 100
            
            # Standardize gap (Z-score)
            gap_mean = stock['gap'].rolling(60).mean()
            gap_std = stock['gap'].rolling(60).std().replace(0, np.nan)
            stock['z_gap'] = (stock['gap'] - gap_mean) / gap_std
            
            stock = stock.dropna()
            
            if len(stock) > 50:
                all_gaps.append(stock[['z_gap', 'gap', 'intraday', 'full_day']])
        
        if not all_gaps:
            continue
        
        combined = pd.concat(all_gaps, ignore_index=True)
        
        # Analyze by gap size
        bins = [(-np.inf, -2), (-2, -1), (-1, 1), (1, 2), (2, np.inf)]
        labels = ['Large Down (<-2σ)', 'Down (-1 to -2σ)', 'Neutral', 'Up (1 to 2σ)', 'Large Up (>2σ)']
        
        for (low, high), label in zip(bins, labels):
            subset = combined[(combined['z_gap'] > low) & (combined['z_gap'] <= high)]
            
            if len(subset) > 20:
                avg_gap = subset['gap'].mean()
                avg_intraday = subset['intraday'].mean()
                avg_full = subset['full_day'].mean()
                
                # Fade ratio: how much of gap is reversed
                fade_ratio = -avg_intraday / avg_gap if avg_gap != 0 else 0
                
                results.append({
                    'group': group_name,
                    'gap_category': label,
                    'avg_gap': avg_gap,
                    'avg_intraday': avg_intraday,
                    'avg_full_day': avg_full,
                    'fade_ratio': fade_ratio,
                    'count': len(subset)
                })
                
                # Determine fade or follow
                behavior = "FADE" if fade_ratio > 0.1 else ("FOLLOW" if fade_ratio < -0.1 else "NEUTRAL")
                
                print(f"  {label}: Gap={avg_gap:+.3f}%, Intraday={avg_intraday:+.3f}%, "
                      f"Fade Ratio={fade_ratio:.2f} ({behavior}), n={len(subset)}")
    
    return pd.DataFrame(results)

def analyze_extreme_gaps(df):
    """
    Deep dive into extreme gap behavior (±2σ).
    """
    print("\n" + "="*60)
    print("EXTREME GAP ANALYSIS")
    print("="*60)
    
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_minnows.csv"))
    
    results = []
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        all_gaps = []
        
        for symbol in stock_list['symbol'].tolist():
            stock = df[df['Name'] == symbol].copy().sort_values('date')
            
            if len(stock) < 100:
                continue
            
            stock['prev_close'] = stock['close'].shift(1)
            stock['gap'] = (stock['open'] / stock['prev_close'] - 1) * 100
            stock['intraday'] = (stock['close'] / stock['open'] - 1) * 100
            
            gap_mean = stock['gap'].rolling(60).mean()
            gap_std = stock['gap'].rolling(60).std().replace(0, np.nan)
            stock['z_gap'] = (stock['gap'] - gap_mean) / gap_std
            
            stock = stock.dropna()
            
            if len(stock) > 30:
                all_gaps.append(stock[['z_gap', 'gap', 'intraday']])
        
        combined = pd.concat(all_gaps, ignore_index=True)
        
        # Extreme down gaps
        down_gaps = combined[combined['z_gap'] < -2]
        if len(down_gaps) > 10:
            avg_gap = down_gaps['gap'].mean()
            avg_intraday = down_gaps['intraday'].mean()
            fade_pct = -avg_intraday / avg_gap * 100 if avg_gap != 0 else 0
            
            results.append({
                'group': group_name,
                'direction': 'down',
                'avg_gap': avg_gap,
                'avg_intraday': avg_intraday,
                'fade_pct': fade_pct,
                'count': len(down_gaps)
            })
            
            print(f"\n{group_name.upper()} - DOWN GAPS (Z < -2σ):")
            print(f"  Average Gap: {avg_gap:+.3f}%")
            print(f"  Average Intraday: {avg_intraday:+.3f}%")
            print(f"  Fade %: {fade_pct:.1f}% of gap recovered")
            print(f"  Count: {len(down_gaps)}")
        
        # Extreme up gaps
        up_gaps = combined[combined['z_gap'] > 2]
        if len(up_gaps) > 10:
            avg_gap = up_gaps['gap'].mean()
            avg_intraday = up_gaps['intraday'].mean()
            fade_pct = -avg_intraday / avg_gap * 100 if avg_gap != 0 else 0
            
            results.append({
                'group': group_name,
                'direction': 'up',
                'avg_gap': avg_gap,
                'avg_intraday': avg_intraday,
                'fade_pct': fade_pct,
                'count': len(up_gaps)
            })
            
            print(f"\n{group_name.upper()} - UP GAPS (Z > 2σ):")
            print(f"  Average Gap: {avg_gap:+.3f}%")
            print(f"  Average Intraday: {avg_intraday:+.3f}%")
            print(f"  Fade %: {fade_pct:.1f}% of gap faded")
            print(f"  Count: {len(up_gaps)}")
    
    return pd.DataFrame(results)

def compute_gap_decay(df):
    """
    Analyze multi-day decay after extreme gaps.
    """
    print("\n" + "="*60)
    print("GAP DECAY ANALYSIS (Multi-Day)")
    print("="*60)
    
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "sp500_minnows.csv"))
    
    results = []
    lags = [1, 2, 3, 5, 10]
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        print(f"\n{group_name.upper()} - Response to DOWN GAPS (Z < -2σ):")
        
        all_data = []
        
        for symbol in stock_list['symbol'].tolist():
            stock = df[df['Name'] == symbol].copy().sort_values('date').reset_index(drop=True)
            
            if len(stock) < 100:
                continue
            
            stock['prev_close'] = stock['close'].shift(1)
            stock['gap'] = (stock['open'] / stock['prev_close'] - 1) * 100
            
            gap_mean = stock['gap'].rolling(60).mean()
            gap_std = stock['gap'].rolling(60).std().replace(0, np.nan)
            stock['z_gap'] = (stock['gap'] - gap_mean) / gap_std
            
            # Forward returns
            for lag in lags:
                stock[f'fwd_{lag}d'] = (stock['close'].shift(-lag) / stock['close'] - 1) * 100
            
            stock = stock.dropna()
            
            if len(stock) > 30:
                all_data.append(stock)
        
        if not all_data:
            continue
        
        combined = pd.concat(all_data, ignore_index=True)
        down_gaps = combined[combined['z_gap'] < -2]
        
        for lag in lags:
            col = f'fwd_{lag}d'
            avg_ret = down_gaps[col].mean()
            
            results.append({
                'group': group_name,
                'lag': lag,
                'avg_return': avg_ret
            })
            
            print(f"  Day {lag}: {avg_ret:+.3f}%")
    
    return pd.DataFrame(results)

def create_gap_plots(gap_df, extreme_df, decay_df):
    """Create visualization plots."""
    print("\nGenerating plots...")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Fade ratio comparison
    fig1 = go.Figure()
    
    for group in ['whales', 'minnows']:
        group_data = gap_df[gap_df['group'] == group]
        
        fig1.add_trace(go.Bar(
            x=group_data['gap_category'],
            y=group_data['fade_ratio'],
            name=group.capitalize()
        ))
    
    fig1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig1.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Full Fade")
    
    fig1.update_layout(
        title="S&P 500 Gap Fade Ratio by Category",
        xaxis_title="Gap Category",
        yaxis_title="Fade Ratio (1 = full reversal)",
        barmode='group',
        width=800,
        height=500
    )
    
    fig1.write_html(os.path.join(PLOTS_DIR, "sp500_angle4_fade_ratio.html"))
    print(f"  Saved: sp500_angle4_fade_ratio.html")
    
    # 2. Gap decay over time
    if not decay_df.empty:
        fig2 = go.Figure()
        
        for group in ['whales', 'minnows']:
            group_data = decay_df[decay_df['group'] == group]
            
            fig2.add_trace(go.Scatter(
                x=group_data['lag'],
                y=group_data['avg_return'],
                mode='lines+markers',
                name=group.capitalize()
            ))
        
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig2.update_layout(
            title="S&P 500 Return After Extreme Down Gaps (Z < -2σ)",
            xaxis_title="Days After Gap",
            yaxis_title="Cumulative Return (%)",
            width=700,
            height=500
        )
        
        fig2.write_html(os.path.join(PLOTS_DIR, "sp500_angle4_gap_decay.html"))
        print(f"  Saved: sp500_angle4_gap_decay.html")

def main():
    # Load data
    df = load_data()
    
    print("\n" + "="*70)
    print("ANGLE 4: Z-GAP ANALYSIS - S&P 500")
    print("="*70)
    
    # 1. Basic gap analysis
    gap_df = compute_gap_analysis(df)
    
    # 2. Extreme gap analysis
    extreme_df = analyze_extreme_gaps(df)
    
    # 3. Gap decay over multiple days
    decay_df = compute_gap_decay(df)
    
    # 4. Create plots
    create_gap_plots(gap_df, extreme_df, decay_df)
    
    # 5. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gap_df.to_csv(os.path.join(RESULTS_DIR, "sp500_angle4_gaps.csv"), index=False)
    extreme_df.to_csv(os.path.join(RESULTS_DIR, "sp500_angle4_extreme.csv"), index=False)
    decay_df.to_csv(os.path.join(RESULTS_DIR, "sp500_angle4_decay.csv"), index=False)
    
    # Summary
    print("\n" + "="*70)
    print("ANGLE 4 SUMMARY")
    print("="*70)
    
    print("\nKey Findings:")
    for group in ['whales', 'minnows']:
        down = extreme_df[(extreme_df['group'] == group) & (extreme_df['direction'] == 'down')]
        if not down.empty:
            fade = down['fade_pct'].values[0]
            behavior = "FADE" if fade > 20 else ("FOLLOW" if fade < -20 else "NEUTRAL")
            print(f"  {group.capitalize()} Down Gaps: {fade:.1f}% fade ({behavior})")
    
    print("\n" + "="*70)
    print("ANGLE 4 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
