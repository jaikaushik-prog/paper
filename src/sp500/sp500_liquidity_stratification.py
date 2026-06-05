"""
S&P 500 Liquidity Stratification
Segments S&P 500 stocks into Whales (high liquidity) and Minnows (low liquidity)
based on Average Daily Turnover (ADT).

Data: all_stocks_5yr.csv (daily OHLCV, 2013-2018)
"""

import pandas as pd
import numpy as np
import os

# Configuration
DATA_FILE = r"c:\Users\DELL\Desktop\project_nifty_liquid\datasets\data\all_stocks_5yr.csv"
OUTPUT_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def load_sp500_data():
    """Load the combined S&P 500 dataset."""
    print("Loading S&P 500 data...")
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Name', 'date'])
    print(f"Loaded {len(df):,} rows, {df['Name'].nunique()} stocks")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df

def calculate_adt(df):
    """
    Calculate Average Daily Turnover (ADT) for each stock.
    ADT = Mean(Close * Volume) over the full period.
    """
    print("\nCalculating Average Daily Turnover (ADT)...")
    
    # Calculate daily turnover
    df['turnover'] = df['close'] * df['volume']
    
    # Group by stock and calculate mean
    adt_stats = df.groupby('Name').agg({
        'turnover': 'mean',
        'date': 'count'  # Number of trading days
    }).reset_index()
    
    adt_stats.columns = ['symbol', 'adt', 'days_count']
    adt_stats = adt_stats.sort_values('adt', ascending=False).reset_index(drop=True)
    
    print(f"Calculated ADT for {len(adt_stats)} stocks")
    return adt_stats

def segment_stocks(adt_df, top_n=50, bottom_n=50):
    """
    Segment stocks into Whales (top N by ADT) and Minnows (bottom N by ADT).
    """
    # Filter out stocks with very few trading days (< 100)
    valid_stocks = adt_df[adt_df['days_count'] >= 100].copy()
    print(f"\nFiltered to {len(valid_stocks)} stocks with >= 100 trading days")
    
    # Get top and bottom
    whales = valid_stocks.head(top_n).copy()
    minnows = valid_stocks.tail(bottom_n).copy()
    
    return whales, minnows

def main():
    # Load data
    df = load_sp500_data()
    
    # Calculate ADT
    adt_df = calculate_adt(df)
    
    # Segment into Whales and Minnows
    whales, minnows = segment_stocks(adt_df)
    
    # Display results
    print("\n" + "="*60)
    print("TOP 10 WHALES (Highest Liquidity)")
    print("="*60)
    print(whales[['symbol', 'adt', 'days_count']].head(10).to_string(index=False))
    
    print("\n" + "="*60)
    print("TOP 10 MINNOWS (Lowest Liquidity)")
    print("="*60)
    print(minnows[['symbol', 'adt', 'days_count']].tail(10).to_string(index=False))
    
    # Summary statistics
    print("\n" + "="*60)
    print("LIQUIDITY SUMMARY")
    print("="*60)
    print(f"Whales ADT Range: ${whales['adt'].min()/1e6:.1f}M - ${whales['adt'].max()/1e6:.1f}M")
    print(f"Minnows ADT Range: ${minnows['adt'].min()/1e6:.1f}M - ${minnows['adt'].max()/1e6:.1f}M")
    print(f"Liquidity Ratio (Whales/Minnows Mean): {whales['adt'].mean() / minnows['adt'].mean():.1f}x")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    whales_file = os.path.join(OUTPUT_DIR, "sp500_whales.csv")
    minnows_file = os.path.join(OUTPUT_DIR, "sp500_minnows.csv")
    
    whales.to_csv(whales_file, index=False)
    minnows.to_csv(minnows_file, index=False)
    
    print(f"\nSaved Whales to: {whales_file}")
    print(f"Saved Minnows to: {minnows_file}")
    
    return whales, minnows

if __name__ == "__main__":
    whales, minnows = main()
