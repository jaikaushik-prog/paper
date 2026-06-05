"""Same-Period Comparison: Nifty 500 vs S&P 500"""
import pandas as pd
import numpy as np
import os

RESULTS_DIR = 'c:/Users/DELL/Desktop/project_nifty_liquid/results'

# Load all surfaces
nw = pd.read_csv(os.path.join(RESULTS_DIR, 'nifty_sameperiod_surface_whales.csv'))
nm = pd.read_csv(os.path.join(RESULTS_DIR, 'nifty_sameperiod_surface_minnows.csv'))
sw = pd.read_csv(os.path.join(RESULTS_DIR, 'sp500_surface_whales.csv'))
sm = pd.read_csv(os.path.join(RESULTS_DIR, 'sp500_surface_minnows.csv'))

# Filter out inf values
nw = nw.replace([np.inf, -np.inf], np.nan).dropna()
nm = nm.replace([np.inf, -np.inf], np.nan).dropna()
sw = sw.replace([np.inf, -np.inf], np.nan).dropna()
sm = sm.replace([np.inf, -np.inf], np.nan).dropna()

print('='*70)
print('SAME-PERIOD COMPARISON: NIFTY 500 vs S&P 500')
print('Period: Feb 2015 - Feb 2018')
print('='*70)

# Metrics
def metrics(df, name):
    avg = df['avg_response'].abs().mean()
    crash = df[df['push_bin'] < -2]['avg_response'].mean()
    return {'name': name, 'avg': avg, 'crash': crash}

nw_m = metrics(nw, 'Nifty Whales')
nm_m = metrics(nm, 'Nifty Minnows')
sw_m = metrics(sw, 'S&P Whales')
sm_m = metrics(sm, 'S&P Minnows')

print(f"\n{'Metric':<25} {'Nifty Whales':<15} {'Nifty Minnows':<15} {'S&P Whales':<15} {'S&P Minnows':<15}")
print('-'*85)
print(f"{'Avg |Response|':<25} {nw_m['avg']:<15.4f} {nm_m['avg']:<15.4f} {sw_m['avg']:<15.4f} {sm_m['avg']:<15.4f}")
print(f"{'Crash Resp (Z<-2)':<25} {nw_m['crash']:<15.4f} {nm_m['crash']:<15.4f} {sw_m['crash']:<15.4f} {sm_m['crash']:<15.4f}")

nifty_ratio = nm_m['avg'] / nw_m['avg'] if nw_m['avg'] > 0 else 0
sp500_ratio = sm_m['avg'] / sw_m['avg'] if sw_m['avg'] > 0 else 0

print()
print(f"{'INEFFICIENCY RATIO':<25} {nifty_ratio:<15.2f}x {'':<15} {sp500_ratio:<15.2f}x")

print()
print('='*70)
print('INTERPRETATION')
print('='*70)
if nifty_ratio > 1.5:
    print(f'Nifty 500 Minnows are {nifty_ratio:.1f}x more predictable than Whales')
elif nifty_ratio > 1.0:
    print(f'Nifty 500 shows weak liquidity-based inefficiency ({nifty_ratio:.2f}x)')
else:
    print('Nifty 500 shows similar efficiency across liquidity groups')

if sp500_ratio < 1.0:
    print('S&P 500 is fully efficient - no liquidity advantage')
elif sp500_ratio > 1.5:
    print(f'S&P 500 Minnows are {sp500_ratio:.1f}x more predictable than Whales')
    
print()
print(f'CONCLUSION: Nifty 500 ({nifty_ratio:.2f}x) vs S&P 500 ({sp500_ratio:.2f}x)')

if nifty_ratio > sp500_ratio:
    diff = (nifty_ratio / sp500_ratio - 1) * 100
    print(f'  -> Nifty 500 is {diff:.0f}% MORE INEFFICIENT than S&P 500')
else:
    diff = (sp500_ratio / nifty_ratio - 1) * 100
    print(f'  -> S&P 500 is {diff:.0f}% MORE INEFFICIENT than Nifty 500')

