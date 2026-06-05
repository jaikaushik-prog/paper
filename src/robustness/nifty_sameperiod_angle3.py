"""
Nifty 500 Angle 3: Regime Dependence - Same Period as S&P 500
Period: Feb 2015 to Feb 2018

Compares Calm vs Stress regime efficiency for direct comparison with S&P 500.
"""

import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

START_DATE = pd.Timestamp("2015-02-01", tz="UTC")
END_DATE = pd.Timestamp("2018-02-28", tz="UTC")

VOL_WINDOW = 20 * 75  # 20 days in 5-min bars

def load_stock_data(file_path, start_date, end_date):
    """Load and filter stock data."""
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except:
        return None

def compute_market_volatility():
    """Compute market-wide volatility for Nifty 500."""
    print("\nComputing Nifty 500 market volatility...")
    
    stock_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))[:100]  # Sample for speed
    
    all_vols = []
    
    for fp in stock_files:
        df = load_stock_data(fp, START_DATE, END_DATE)
        if df is None or len(df) < 1000:
            continue
        
        df['ret'] = np.log(df['close'] / df['close'].shift(1))
        df['vol'] = df['ret'].rolling(VOL_WINDOW).std() * np.sqrt(252 * 75)  # Annualized
        df['day'] = df['date'].dt.date
        
        # Daily volatility
        daily_vol = df.groupby('day')['vol'].mean().reset_index()
        daily_vol.columns = ['date', 'vol']
        all_vols.append(daily_vol)
    
    vol_df = pd.concat(all_vols)
    market_vol = vol_df.groupby('date')['vol'].mean().reset_index()
    market_vol.columns = ['date', 'market_vol']
    
    # Define regimes
    vol_20 = market_vol['market_vol'].quantile(0.20)
    vol_80 = market_vol['market_vol'].quantile(0.80)
    
    market_vol['regime'] = 'normal'
    market_vol.loc[market_vol['market_vol'] <= vol_20, 'regime'] = 'calm'
    market_vol.loc[market_vol['market_vol'] >= vol_80, 'regime'] = 'stress'
    
    print(f"\nVolatility Thresholds:")
    print(f"  Calm (Bottom 20%): Vol <= {vol_20:.2%}")
    print(f"  Stress (Top 20%): Vol >= {vol_80:.2%}")
    
    return market_vol

def process_stock_regime(args):
    """Process one stock for regime analysis."""
    symbol, start_date, end_date, calm_dates, stress_dates = args
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    
    df = load_stock_data(file_path, start_date, end_date)
    if df is None or len(df) < 2000:
        return None
    
    df['day'] = df['date'].dt.date
    df['ret'] = np.log(df['close'] / df['close'].shift(1))
    df['z_ret'] = (df['ret'] - df['ret'].rolling(1500).mean()) / df['ret'].rolling(1500).std()
    
    # Lag in 5-min bars (5 days = 375 bars)
    lag = 375
    df['z_push'] = df['z_ret'].rolling(lag).sum()
    df['z_resp'] = df['z_ret'].shift(-lag).rolling(lag).sum()
    
    df = df.dropna()
    
    results = {}
    for regime, dates in [('calm', calm_dates), ('stress', stress_dates)]:
        regime_data = df[df['day'].isin(dates)]
        
        if len(regime_data) < 100:
            continue
        
        crash = regime_data[regime_data['z_push'] < -2]
        if len(crash) > 10:
            results[regime] = {
                'crash_resp': crash['z_resp'].mean(),
                'count': len(crash)
            }
    
    return results if results else None

def compute_regime_surfaces(market_vol):
    """Compute regime-dependent surfaces for Nifty 500."""
    print("\n" + "="*60)
    print("COMPUTING NIFTY 500 REGIME-DEPENDENT SURFACES")
    print("="*60)
    
    # Get regime dates
    calm_dates = set(market_vol[market_vol['regime'] == 'calm']['date'].tolist())
    stress_dates = set(market_vol[market_vol['regime'] == 'stress']['date'].tolist())
    
    # Load Whales and Minnows from same-period analysis
    whales = pd.read_csv(os.path.join(PROCESSED_DIR, "nifty_sameperiod_whales.csv"))
    minnows = pd.read_csv(os.path.join(PROCESSED_DIR, "nifty_sameperiod_minnows.csv"))
    
    results = []
    
    for group_name, stock_list in [('whales', whales), ('minnows', minnows)]:
        print(f"\nProcessing {group_name}...")
        
        task_args = [(sym, START_DATE, END_DATE, calm_dates, stress_dates) 
                     for sym in stock_list['symbol'].tolist()]
        
        regime_data = {'calm': [], 'stress': []}
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_stock_regime, arg): arg for arg in task_args}
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    for regime in ['calm', 'stress']:
                        if regime in res:
                            regime_data[regime].append(res[regime])
        
        for regime in ['calm', 'stress']:
            if regime_data[regime]:
                total_count = sum(d['count'] for d in regime_data[regime])
                weighted_resp = sum(d['crash_resp'] * d['count'] for d in regime_data[regime]) / total_count
                
                results.append({
                    'group': group_name,
                    'regime': regime,
                    'crash_response': weighted_resp,
                    'count': total_count
                })
                
                print(f"  {group_name.capitalize()} - {regime.upper()}: "
                      f"Response = {weighted_resp:+.4f} (n={total_count})")
    
    return pd.DataFrame(results)

def compare_markets():
    """Compare Nifty 500 vs S&P 500 regime results."""
    print("\n" + "="*70)
    print("CROSS-MARKET REGIME COMPARISON (Same Period)")
    print("="*70)
    
    # Load results
    sp500 = pd.read_csv(os.path.join(RESULTS_DIR, "sp500_angle3_regimes.csv"))
    sp500 = sp500[~sp500['regime'].str.contains('rally')]
    
    nifty = pd.read_csv(os.path.join(RESULTS_DIR, "nifty_sameperiod_angle3.csv"))
    
    print(f"\n{'Market':<12} {'Group':<10} {'CALM':<12} {'STRESS':<12}")
    print("-"*50)
    
    for market, df in [('S&P 500', sp500), ('Nifty 500', nifty)]:
        for group in ['whales', 'minnows']:
            calm = df[(df['group'] == group) & (df['regime'] == 'calm')]['crash_response'].values
            stress = df[(df['group'] == group) & (df['regime'] == 'stress')]['crash_response'].values
            
            calm_val = f"{calm[0]:+.4f}" if len(calm) > 0 else "N/A"
            stress_val = f"{stress[0]:+.4f}" if len(stress) > 0 else "N/A"
            
            print(f"{market:<12} {group.capitalize():<10} {calm_val:<12} {stress_val:<12}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Calculate inefficiency ratios
    for market, df in [('S&P 500', sp500), ('Nifty 500', nifty)]:
        for regime in ['calm', 'stress']:
            whale = df[(df['group'] == 'whales') & (df['regime'] == regime)]['crash_response'].values
            minnow = df[(df['group'] == 'minnows') & (df['regime'] == regime)]['crash_response'].values
            
            if len(whale) > 0 and len(minnow) > 0 and whale[0] != 0:
                ratio = minnow[0] / whale[0]
                print(f"{market} {regime.upper()} Inefficiency Ratio: {ratio:.2f}x")

def main():
    print("="*70)
    print("NIFTY 500 ANGLE 3: REGIME DEPENDENCE (Same Period as S&P 500)")
    print("Period: Feb 2015 - Feb 2018")
    print("="*70)
    
    # 1. Compute market volatility
    market_vol = compute_market_volatility()
    
    # 2. Compute regime surfaces
    results_df = compute_regime_surfaces(market_vol)
    
    # 3. Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "nifty_sameperiod_angle3.csv"), index=False)
    
    # 4. Compare markets
    compare_markets()
    
    print("\n" + "="*70)
    print("ANGLE 3 SAME-PERIOD COMPARISON COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
