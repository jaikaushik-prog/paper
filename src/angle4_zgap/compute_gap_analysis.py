import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATA_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\raw_Data"
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PROCESSED_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\processed_data"

def process_stock_gap(symbol):
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    try:
        # Load 5-min Data
        df = pd.read_csv(file_path, usecols=['date', 'open', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'], utc=True)
        
        # 1. Calculate Daily Volatility (Prev Day)
        # Resample to Daily to get Open, Close, and Volatility
        df['ret_5m'] = np.log(df['close'] / df['close'].shift(1))
        
        # Group by Date
        # We need:
        # - Daily Open (first open)
        # - Daily Close (last close)
        # - Daily Vol (std of 5m returns)
        
        # Trick: Use resampling
        # But for 'std' we need to operate on the 5m returns
        
        # Add 'day' column
        df['day'] = df['date'].dt.date
        
        # Aggregation
        daily_stats = df.groupby('day').agg({
            'open': 'first',
            'close': 'last',
            'ret_5m': 'std' # This gives daily volatility of 5m returns
        }).rename(columns={'ret_5m': 'volatility'})
        
        # Shift Volatility and Close to get "Previous Day" stats for "Today"
        daily_stats['prev_close'] = daily_stats['close'].shift(1)
        daily_stats['prev_vol'] = daily_stats['volatility'].shift(1)
        
        # Calculate Z-Gap
        # gap = ln(Open_today) - ln(Close_prev)
        daily_stats['gap_log'] = np.log(daily_stats['open'] / daily_stats['prev_close'])
        
        # Z-Gap
        # Note: If prev_vol is 0, we have an issue. Replace 0 with NaN or small number
        daily_stats['z_gap'] = daily_stats['gap_log'] / daily_stats['prev_vol'].replace(0, np.nan)
        
        # 2. Extract 30-min Response
        # We need the price at 09:45 for each day.
        # Let's find the row corresponding to Open + 30 mins
        # Assuming Open is the first timestamp of the day.
        # Alternatively, strict 09:45 check?
        # Indian market opens 09:15. 30 mins later is 09:45.
        
        # Filter for rows at 09:45
        # The timestamp in CSV is likely "+05:30". 
        # let's look for time component.
        
        # Isolate 09:45 rows
        # We can extract time.
        # NOTE: If 09:45 is missing, we might miss data.
        # Ideally: "Price 30 mins after open".
        
        # Better: Group by day, take 6th row? (5 min bars. 09:15, 20, 25, 30, 35, 40, 45.. 09:15 is bar 1?
        # If timestamps are "End of bar"?
        # 09:15 open -> 09:20 bar?
        # Let's try to match 09:45 time explicitly first.
        
        # Convert to Pytz or local to ensure we check 09:45 IST
        # Data is "2024-08-26 09:15:00+05:30"
        
        # Let's create a lookup for 09:45 close prices
        mask_0945 = df['date'].dt.hour == 9
        mask_0945 &= df['date'].dt.minute == 45
        
        # If timezone conversion needed:
        # df['date'] is UTC aware?
        # 09:45 IST = 04:15 UTC.
        # Let's check the date format in a previous step...
        # It was +05:30 offset in CSV.
        
        # Let's rely on finding the 09:45 bar.
        # If we just take 'nth' bar, it's safer against missing timestamps?
        # 09:15 to 09:45 is 30 mins.
        # 5 min candles: 15-20, 20-25, 25-30, 30-35, 35-40, 40-45.
        # That is 6 candles.
        # So we want the Close of the 6th candle of the day.
        
        # Get 6th candle per day
        # df is sorted.
        day_groups = df.groupby('day')
        
        # We need a function to get 6th close
        def get_30m_close(group):
            if len(group) >= 6:
                return group.iloc[5]['close'] # 0-indexed, 5 is 6th
            return np.nan
            
        # Apply is slow. fast way?
        # filter by cumcount?
        df['n'] = day_groups.cumcount()
        response_rows = df[df['n'] == 5].set_index('day')['close'] # 6th candle
        
        daily_stats['close_30m'] = response_rows
        
        # Response Return
        daily_stats['response_30m'] = np.log(daily_stats['close_30m'] / daily_stats['open'])
        
        # Clean
        final_df = daily_stats[['z_gap', 'response_30m']].dropna()
        # Filter outliers? z_gap +/- 10?
        final_df = final_df[final_df['z_gap'].abs() < 10]
        
        return final_df
        
    except Exception as e:
        # print(e)
        return None

def process_group_gaps(group_name, stock_list):
    print(f"Processing GAPS for {group_name}...")
    symbols = stock_list['symbol'].tolist()
    
    all_gaps = []
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_stock_gap, sym): sym for sym in symbols}
        
        count = 0
        for future in as_completed(futures):
            res = future.result()
            count += 1
            if count % 10 == 0:
                print(f"[{group_name}] Processed {count}/{len(symbols)}...")
                
            if res is not None:
                all_gaps.append(res)
                
    if not all_gaps:
        print(f"No gap data for {group_name}")
        return
        
    merged = pd.concat(all_gaps)
    
    # Save raw for safety
    # merged.to_csv(os.path.join(RESULTS_DIR, f"gaps_raw_{group_name}.csv"))
    
    # Binning for Curve
    print(f"Binning {len(merged)} gap events...")
    bins = np.linspace(-4, 4, 50) # -4 sigma to +4 sigma
    merged['bin'] = pd.cut(merged['z_gap'], bins)
    
    curve = merged.groupby('bin', observed=True)['response_30m'].mean()
    
    # Convert to DF for saving
    curve_df = curve.reset_index()
    curve_df['bin_mid'] = curve_df['bin'].apply(lambda x: x.mid).astype(float)
    
    out_file = os.path.join(RESULTS_DIR, f"gap_curve_{group_name}.csv")
    curve_df[['bin_mid', 'response_30m']].to_csv(out_file, index=False)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    # Load Groups
    whales_path = os.path.join(PROCESSED_DIR, "whales.csv")
    minnows_path = os.path.join(PROCESSED_DIR, "minnows.csv")
    
    if os.path.exists(whales_path):
        whales = pd.read_csv(whales_path)
        process_group_gaps("whales", whales)
        
    if os.path.exists(minnows_path):
        minnows = pd.read_csv(minnows_path)
        process_group_gaps("minnows", minnows)
