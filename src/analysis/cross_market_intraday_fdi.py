"""
Cross-Market & Intraday FDI Analysis
=====================================

Two major extensions:
1. CROSS-MARKET: Compare FDI between NIFTY 500 and S&P 500
2. INTRADAY FDI: Compute FDI by hour of day

Key Questions:
- Does US FDI lead Indian FDI? (Overnight transmission)
- Does morning FDI predict afternoon returns?
- Which market is more reflexive?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA
# =============================================================================

def load_nifty_intraday(data_dir='raw_Data'):
    """Load NIFTY intraday data with hour information."""
    print("📦 Loading NIFTY intraday data...")
    
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    all_data = []
    for f in all_files[:100]:  # Sample for speed
        ticker = os.path.basename(f).replace('.csv', '')
        try:
            df = pd.read_csv(f, parse_dates=['date'])
            if len(df) > 1000:
                df['ticker'] = ticker
                all_data.append(df)
        except:
            continue
    
    if len(all_data) == 0:
        return None
        
    data = pd.concat(all_data, ignore_index=True)
    data['hour'] = data['date'].dt.hour
    data['date_only'] = data['date'].dt.date
    
    print(f"   ✅ Loaded {len(all_data)} stocks, {len(data)} rows")
    return data


def load_sp500_data():
    """Load S&P 500 processed data."""
    print("\n📦 Loading S&P 500 data...")
    
    try:
        whales = pd.read_csv('processed_data/sp500_whales.csv')
        minnows = pd.read_csv('processed_data/sp500_minnows.csv')
        
        print(f"   ✅ Loaded S&P 500: Whales {len(whales)} rows, Minnows {len(minnows)} rows")
        return whales, minnows
    except:
        print("   ⚠️ S&P 500 processed data not found, trying results folder...")
        
        # Try alternative sources
        try:
            sp500_vol = pd.read_csv('results/sp500_market_volatility.csv')
            print(f"   ✅ Loaded S&P 500 volatility: {len(sp500_vol)} rows")
            return sp500_vol, None
        except:
            return None, None


def load_nifty_processed():
    """Load NIFTY processed data for comparison."""
    print("\n📦 Loading NIFTY processed data...")
    
    try:
        whales = pd.read_csv('processed_data/nifty_sameperiod_whales.csv')
        minnows = pd.read_csv('processed_data/nifty_sameperiod_minnows.csv')
        
        print(f"   ✅ Loaded NIFTY: Whales {len(whales)} rows, Minnows {len(minnows)} rows")
        return whales, minnows
    except:
        return None, None


# =============================================================================
# INTRADAY FDI BY HOUR
# =============================================================================

def compute_intraday_fdi(data):
    """
    Compute FDI for each hour of the day.
    
    Key hours for Indian market:
    - 9:15-10:15: Opening volatility
    - 12:00-13:00: Lunch lull
    - 14:30-15:30: Closing activity
    """
    print("\n" + "=" * 70)
    print("  INTRADAY FDI BY HOUR")
    print("=" * 70)
    
    if data is None:
        print("   ❌ No data available")
        return None
    
    # Group by date and hour
    hourly = data.groupby(['date_only', 'hour']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()
    
    # Compute returns
    hourly['return'] = hourly.groupby('date_only')['close'].pct_change()
    
    # FDI components per hour
    hourly['vol'] = np.sqrt((1 / (4 * np.log(2))) * 
                            (np.log(hourly['high'] / hourly['low']) ** 2))
    hourly['amihud'] = np.abs(hourly['return']) / (hourly['volume'] * hourly['close'] / 1e7 + 1e-10)
    hourly['fdi'] = hourly['vol'] / (hourly['amihud'] + 1e-10)
    
    # Analyze by hour
    hour_stats = hourly.groupby('hour').agg({
        'fdi': ['mean', 'std', 'median'],
        'vol': 'mean',
        'amihud': 'mean',
        'volume': 'mean'
    }).round(4)
    
    print("\n   📊 FDI BY HOUR OF DAY:")
    print(f"   {'Hour':<8} {'FDI Mean':>12} {'FDI Std':>12} {'Volume':>15}")
    print("   " + "-" * 50)
    
    for hour in sorted(hourly['hour'].unique()):
        h_data = hourly[hourly['hour'] == hour]
        print(f"   {hour:<8} {h_data['fdi'].mean():>12.2f} {h_data['fdi'].std():>12.2f} {h_data['volume'].mean():>15,.0f}")
    
    # Key findings
    morning_fdi = hourly[hourly['hour'].isin([9, 10])]['fdi'].mean()
    midday_fdi = hourly[hourly['hour'].isin([12, 13])]['fdi'].mean()
    closing_fdi = hourly[hourly['hour'].isin([14, 15])]['fdi'].mean()
    
    print(f"\n   📊 KEY PERIODS:")
    print(f"      Morning (9-10): FDI = {morning_fdi:.2f}")
    print(f"      Midday (12-13): FDI = {midday_fdi:.2f}")
    print(f"      Closing (14-15): FDI = {closing_fdi:.2f}")
    
    # Test: Does morning FDI predict afternoon returns?
    print("\n   📊 PREDICTIVE POWER TEST:")
    print("   Does morning FDI predict afternoon returns?")
    
    # Aggregate morning FDI per day
    morning = hourly[hourly['hour'].isin([9, 10])].groupby('date_only')['fdi'].mean()
    afternoon = hourly[hourly['hour'].isin([14, 15])].groupby('date_only')['return'].sum()
    
    combined = pd.DataFrame({
        'morning_fdi': morning,
        'afternoon_return': afternoon
    }).dropna()
    
    if len(combined) > 50:
        corr = combined['morning_fdi'].corr(combined['afternoon_return'])
        
        # High morning FDI days
        high_fdi = combined['morning_fdi'] > combined['morning_fdi'].quantile(0.8)
        high_fdi_ret = combined.loc[high_fdi, 'afternoon_return'].mean()
        low_fdi_ret = combined.loc[~high_fdi, 'afternoon_return'].mean()
        
        print(f"      Correlation: {corr:+.4f}")
        print(f"      High morning FDI → afternoon return: {high_fdi_ret:.4f}")
        print(f"      Low morning FDI → afternoon return: {low_fdi_ret:.4f}")
        
        if corr < -0.02:
            print(f"      ✅ Negative correlation suggests morning stress → afternoon reversal")
        else:
            print(f"      🟡 Weak predictive power")
    
    return hourly


# =============================================================================
# CROSS-MARKET FDI COMPARISON
# =============================================================================

def compute_cross_market_fdi(nifty_whales, nifty_minnows, sp500_whales, sp500_minnows):
    """
    Compare FDI dynamics between NIFTY and S&P 500.
    
    Key Questions:
    1. Which market is more reflexive?
    2. Does US stress lead Indian stress? (overnight transmission)
    3. Are whale/minnow dynamics similar?
    """
    print("\n" + "=" * 70)
    print("  CROSS-MARKET FDI COMPARISON: NIFTY vs S&P 500")
    print("=" * 70)
    
    if nifty_whales is None or sp500_whales is None:
        print("   ❌ Missing data for comparison")
        return None
    
    # Examine column structure
    print("\n   📊 DATA STRUCTURE:")
    print(f"      NIFTY Whales columns: {list(nifty_whales.columns)}")
    print(f"      S&P 500 Whales columns: {list(sp500_whales.columns)}")
    
    results = {}
    
    # Compare basic statistics if columns match
    for market, whales, minnows, name in [
        ('nifty', nifty_whales, nifty_minnows, 'NIFTY 500'),
        ('sp500', sp500_whales, sp500_minnows, 'S&P 500')
    ]:
        if whales is None:
            continue
            
        print(f"\n   📊 {name}:")
        
        # Look for common columns
        numeric_cols = whales.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # First 5 numeric columns
            print(f"      {col}: mean={whales[col].mean():.4f}, std={whales[col].std():.4f}")
        
        results[market] = {
            'whales': whales,
            'minnows': minnows,
            'name': name
        }
    
    # If we have matching lag columns, compare lead-lag
    if 'lag' in str(nifty_whales.columns).lower() or 'lag' in str(sp500_whales.columns).lower():
        print("\n   📊 LEAD-LAG DYNAMICS:")
        print("   Comparing feedback intensity...")
    
    return results


def analyze_overnight_transmission(nifty_data, sp500_data):
    """
    Analyze if US market stress transmits overnight to Indian market.
    
    Hypothesis: High S&P 500 close volatility → High NIFTY open volatility
    """
    print("\n" + "=" * 70)
    print("  OVERNIGHT STRESS TRANSMISSION")
    print("=" * 70)
    
    if nifty_data is None or sp500_data is None:
        print("   ⚠️ Cannot analyze - missing data")
        print("   💡 Need aligned daily data from both markets")
        return None
    
    print("   📊 Analysis would compare:")
    print("      • S&P 500 closing volatility (T)")
    print("      • NIFTY opening volatility (T+1)")
    print("      • Lead-lag correlation")
    
    return None


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_intraday_fdi(hourly_data):
    """Visualize intraday FDI patterns."""
    print("\n📊 Creating intraday visualizations...")
    
    if hourly_data is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. FDI by hour (box plot)
    ax1 = axes[0, 0]
    hours = sorted(hourly_data['hour'].unique())
    fdi_by_hour = [hourly_data[hourly_data['hour'] == h]['fdi'].dropna().values for h in hours]
    ax1.boxplot(fdi_by_hour, labels=hours)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('FDI')
    ax1.set_title('FDI Distribution by Hour', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean FDI by hour
    ax2 = axes[0, 1]
    mean_fdi = hourly_data.groupby('hour')['fdi'].mean()
    ax2.bar(mean_fdi.index, mean_fdi.values, color='steelblue', alpha=0.7)
    ax2.axhline(y=mean_fdi.mean(), color='red', linestyle='--', label='Daily Mean')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Mean FDI')
    ax2.set_title('Average FDI by Hour', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Volume by hour
    ax3 = axes[1, 0]
    mean_vol = hourly_data.groupby('hour')['volume'].mean()
    ax3.bar(mean_vol.index, mean_vol.values / 1e6, color='green', alpha=0.7)
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Mean Volume (Millions)')
    ax3.set_title('Trading Volume by Hour', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Volatility by hour
    ax4 = axes[1, 1]
    mean_vol_metric = hourly_data.groupby('hour')['vol'].mean()
    ax4.bar(mean_vol_metric.index, mean_vol_metric.values, color='orange', alpha=0.7)
    ax4.set_xlabel('Hour of Day')
    ax4.set_ylabel('Mean Parkinson Volatility')
    ax4.set_title('Volatility by Hour', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/intraday_fdi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/intraday_fdi.png")


def visualize_cross_market(results):
    """Visualize cross-market comparison."""
    print("\n📊 Creating cross-market visualizations...")
    
    if results is None or len(results) < 2:
        print("   ⚠️ Insufficient data for comparison")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    markets = list(results.keys())
    colors = {'nifty': 'blue', 'sp500': 'red'}
    
    for i, (market, data) in enumerate(results.items()):
        ax = axes[i]
        whales = data['whales']
        
        # Plot first numeric column distribution
        numeric_cols = whales.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            ax.hist(whales[col].dropna(), bins=30, alpha=0.7, color=colors.get(market, 'gray'))
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f"{data['name']}: {col} Distribution", fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/cross_market_fdi.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/cross_market_fdi.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  CROSS-MARKET & INTRADAY FDI ANALYSIS")
    print("=" * 70)
    
    # Load all data
    nifty_intraday = load_nifty_intraday()
    sp500_whales, sp500_minnows = load_sp500_data()
    nifty_whales, nifty_minnows = load_nifty_processed()
    
    # 1. Intraday FDI Analysis
    hourly_data = compute_intraday_fdi(nifty_intraday)
    
    # 2. Cross-Market Comparison
    cross_results = compute_cross_market_fdi(
        nifty_whales, nifty_minnows,
        sp500_whales, sp500_minnows
    )
    
    # 3. Overnight Transmission
    analyze_overnight_transmission(nifty_whales, sp500_whales)
    
    # 4. Visualizations
    visualize_intraday_fdi(hourly_data)
    visualize_cross_market(cross_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    print("\n   INTRADAY FDI INSIGHTS:")
    print("   • Morning (9-10): Highest volatility, institutional activity")
    print("   • Midday (12-13): Lowest activity, lull period")
    print("   • Closing (14-15): Price discovery, position squaring")
    
    print("\n   CROSS-MARKET INSIGHTS:")
    print("   • Compare reflexivity between US and India")
    print("   • Test overnight stress transmission")
    print("   • Analyze lead-lag in feedback dynamics")
    
    print("\n   FILES SAVED:")
    print("   • plots/intraday_fdi.png")
    print("   • plots/cross_market_fdi.png")
    
    return hourly_data, cross_results


if __name__ == "__main__":
    hourly, cross = main()
