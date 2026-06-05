import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
OUTPUT_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def load_and_preprocess(file_path):
    """
    Load a stock CSV, parse dates, and filter for the last year of data.
    """
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], utc=True)
        # Sort just in case
        df = df.sort_values('date')
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_liquidity_groups(data_dir, lookback_days=365):
    """
    Calculate Average Daily Turnover (ADT) and segment stocks.
    """
    print("Calculating Average Daily Turnover (ADT) for all stocks...")
    stock_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    adt_stats = []
    
    # We need to find the global last date first to define the "last year" consistently
    # But reading all files just for that is expensive. We'll assume the user provided end date 
    # or just use the max date from the first few files or the sample we saw (2025-08-26).
    # Better approach: Read tail of files to update max_date.
    
    
    # Global max date hardcoded below

# Worker function must be top-level for multiprocessing on Windows
def process_file_worker(args):
    file_path, start_date, global_max_date = args
    symbol = os.path.basename(file_path).replace('.csv', '')
    try:
        df = pd.read_csv(file_path, usecols=['date', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        mask = (df['date'] >= start_date) & (df['date'] <= global_max_date)
        recent_df = df[mask].copy()
        
        if recent_df.empty:
            return None
            
        recent_df['turnover'] = recent_df['close'] * recent_df['volume']
        daily_turnover = recent_df.groupby(recent_df['date'].dt.date)['turnover'].sum()
        avg_daily_turnover = daily_turnover.mean()
        
        return {
            'symbol': symbol,
            'adt': avg_daily_turnover,
            'days_count': len(daily_turnover)
        }
    except Exception as e:
        return None

def get_liquidity_groups(data_dir, lookback_days=365):
    """
    Calculate Average Daily Turnover (ADT) and segment stocks.
    """
    print("Calculating Average Daily Turnover (ADT) for all stocks...")
    stock_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    # Hardcoded reference based on exploration
    global_max_date = pd.Timestamp("2025-08-26", tz="UTC")
    start_date = global_max_date - timedelta(days=lookback_days)
    
    print(f"Reference Date Range: {start_date} to {global_max_date}")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    adt_stats = []
    max_workers = 8 
    print(f"Starting parallel processing with {max_workers} workers...")
    
    # Prepare arguments for each task
    task_args = [(fp, start_date, global_max_date) for fp in stock_files]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_worker, arg): arg for arg in task_args}
        
        count = 0
        total = len(stock_files)
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                adt_stats.append(res)
            
            count += 1
            if count % 50 == 0:
                print(f"Processed {count}/{total} files...")

    adt_df = pd.DataFrame(adt_stats)
    adt_df = adt_df.sort_values('adt', ascending=False).reset_index(drop=True)
    
    # Segment
    top_50 = adt_df.head(50)
    bottom_50 = adt_df.tail(50) # Or valid bottom 50 (exclude zeros?)
    
    # Remove very illiquid or empty ones if necessary? 
    # Paper compares liquid vs illiquid. Minnows should still trade.
    
    return top_50, bottom_50

if __name__ == "__main__":
    top_50, bottom_50 = get_liquidity_groups(DATA_DIR)
    
    print("\nTop 5 Whales:")
    print(top_50[['symbol', 'adt']].head())
    
    print("\nTop 5 Minnows (lowest liquidity in the set):")
    print(bottom_50[['symbol', 'adt']].tail()) # Tail of bottom 50 is the absolute lowest
    
    # Save lists for next step
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    top_50.to_csv(os.path.join(OUTPUT_DIR, "whales.csv"), index=False)
    bottom_50.to_csv(os.path.join(OUTPUT_DIR, "minnows.csv"), index=False)
