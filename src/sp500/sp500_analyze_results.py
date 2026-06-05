"""
S&P 500 Analyze Results
Computes quantitative metrics comparing Whales vs Minnows efficiency.
"""

import pandas as pd
import numpy as np
import os

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

def load_surface(group_name):
    """Load surface data from CSV."""
    file_path = os.path.join(RESULTS_DIR, f"sp500_surface_{group_name}.csv")
    return pd.read_csv(file_path)

def calculate_metrics(df, group_name):
    """
    Calculate key efficiency metrics from surface data.
    """
    metrics = {}
    
    # Average absolute response strength (|z_resp|)
    metrics['avg_abs_response'] = df['avg_response'].abs().mean()
    
    # Max absolute response
    metrics['max_abs_response'] = df['avg_response'].abs().max()
    
    # Asymmetry: Compare negative push response vs positive push response
    neg_push = df[df['push_bin'] < -1]['avg_response'].mean()  # Response to crashes
    pos_push = df[df['push_bin'] > 1]['avg_response'].mean()   # Response to rallies
    
    # For crashes, we expect mean reversion (positive response)
    # For rallies, we expect momentum (positive) or reversion (negative)
    metrics['crash_response'] = neg_push  # + means reversion
    metrics['rally_response'] = pos_push
    
    # Asymmetry ratio (how much more reactive to crashes)
    # If crashes cause reversion (+) and rallies are flat (0), ratio is high
    if abs(pos_push) > 0.001:
        metrics['asymmetry_ratio'] = abs(neg_push) / abs(pos_push)
    else:
        metrics['asymmetry_ratio'] = abs(neg_push) * 100  # Very asymmetric
    
    # Extreme event response (Z < -3 and Z > 3)
    extreme_neg = df[df['push_bin'] < -3]['avg_response'].mean()
    extreme_pos = df[df['push_bin'] > 3]['avg_response'].mean()
    metrics['extreme_crash_response'] = extreme_neg
    metrics['extreme_rally_response'] = extreme_pos
    
    return metrics

def analyze_lag_decay(df, group_name):
    """
    Analyze how response strength decays with lag.
    """
    print(f"\n{group_name.upper()} - Response by Lag:")
    print("-" * 50)
    
    # Focus on crash response (Z < -2)
    crash_data = df[df['push_bin'] < -2].copy()
    
    if crash_data.empty:
        print("  No crash data available")
        return
    
    lag_summary = crash_data.groupby('lag')['avg_response'].mean()
    
    for lag, resp in lag_summary.items():
        direction = "reversion" if resp > 0 else "momentum"
        print(f"  Lag {lag:2d} days: {resp:+.4f} ({direction})")
    
    return lag_summary

def main():
    print("="*60)
    print("S&P 500 EFFICIENCY ANALYSIS")
    print("="*60)
    
    # Load surfaces
    whales = load_surface("whales")
    minnows = load_surface("minnows")
    
    # Calculate metrics
    whales_metrics = calculate_metrics(whales, "whales")
    minnows_metrics = calculate_metrics(minnows, "minnows")
    
    # Display comparison
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<30} {'Whales':<15} {'Minnows':<15} {'Ratio':<10}")
    print("-" * 70)
    
    # Average Response Strength
    w_avg = whales_metrics['avg_abs_response']
    m_avg = minnows_metrics['avg_abs_response']
    ratio = m_avg / w_avg if w_avg > 0 else float('inf')
    print(f"{'Avg Response Strength':<30} {w_avg:<15.4f} {m_avg:<15.4f} {ratio:.2f}x")
    
    # Max Response Strength
    w_max = whales_metrics['max_abs_response']
    m_max = minnows_metrics['max_abs_response']
    ratio = m_max / w_max if w_max > 0 else float('inf')
    print(f"{'Max Response Strength':<30} {w_max:<15.4f} {m_max:<15.4f} {ratio:.2f}x")
    
    # Crash Response (Mean Reversion after -2σ push)
    w_crash = whales_metrics['crash_response']
    m_crash = minnows_metrics['crash_response']
    print(f"{'Crash Response (Z<-1)':<30} {w_crash:<15.4f} {m_crash:<15.4f}")
    
    # Rally Response
    w_rally = whales_metrics['rally_response']
    m_rally = minnows_metrics['rally_response']
    print(f"{'Rally Response (Z>1)':<30} {w_rally:<15.4f} {m_rally:<15.4f}")
    
    # Extreme Events
    w_ext = whales_metrics['extreme_crash_response']
    m_ext = minnows_metrics['extreme_crash_response']
    print(f"{'Extreme Crash (Z<-3)':<30} {w_ext:<15.4f} {m_ext:<15.4f}")
    
    # Lag decay analysis
    print("\n" + "="*60)
    print("LAG DECAY ANALYSIS (Response to Z < -2)")
    print("="*60)
    
    analyze_lag_decay(whales, "whales")
    analyze_lag_decay(minnows, "minnows")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    inefficiency_ratio = m_avg / w_avg if w_avg > 0 else float('inf')
    
    if inefficiency_ratio > 2:
        print(f"\n✓ INEFFICIENCY DETECTED: Minnows are {inefficiency_ratio:.1f}x more predictable")
        print("  This suggests liquidity-driven inefficiency exists in S&P 500 small caps")
    elif inefficiency_ratio > 1.2:
        print(f"\n◐ WEAK INEFFICIENCY: Minnows are {inefficiency_ratio:.1f}x more predictable")
        print("  Some liquidity premium exists but weaker than Nifty 500")
    else:
        print(f"\n✗ EFFICIENT MARKET: Whales and Minnows show similar efficiency")
        print("  S&P 500 appears more efficient than Nifty 500")
    
    # Compare to Nifty 500 baseline
    print("\n" + "-"*60)
    print("COMPARISON TO NIFTY 500 (Indian Market)")
    print("-"*60)
    print(f"  Nifty 500 Inefficiency Ratio: ~3.0x")
    print(f"  S&P 500 Inefficiency Ratio:  {inefficiency_ratio:.2f}x")
    
    if inefficiency_ratio < 2:
        print("\n  → S&P 500 is MORE EFFICIENT than Nifty 500")
        print("  → US market makers provide better liquidity")
    else:
        print("\n  → S&P 500 shows SIMILAR inefficiency to Nifty 500")
        print("  → Liquidity stratification is a universal phenomenon")

if __name__ == "__main__":
    main()
