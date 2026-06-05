
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, time

def prepare_data_from_csv():
    # 1. Setup paths
    # Try multiple standard locations
    potential_paths = ["raw_Data", "datasets/raw_Data", "../raw_Data"]
    raw_dir = None
    csv_files = []
    
    for p in potential_paths:
        if os.path.exists(p):
            found = glob.glob(os.path.join(p, "*.csv"))
            # Filter non-stock
            found = [f for f in found if "optimized_results" not in f and "output_template" not in f]
            if len(found) > 0:
                raw_dir = p
                csv_files = found
                print(f"✅ Found {len(found)} CSVs in {p}")
                break
    
    if not raw_dir:
        print("❌ No CSV files found in any standard location.")
        exit(1)
            
    output_dir = "datasets/Nifty500"
    os.makedirs(output_dir, exist_ok=True)

    
    # 2. Read and Align
    # To avoid memory explosion, we'll read 'close' and 'date' only.
    dfs = []
    
    print("⏳ Reading CSVs (this may take a while)...")
    for i, fpath in enumerate(csv_files):
        try:
            # Ticker name from filename
            ticker = os.path.basename(fpath).replace(".csv", "")
            
            # Read only date and close
            df = pd.read_csv(fpath, usecols=['date', 'close'])
            
            # Rename close to ticker
            df = df.rename(columns={'close': ticker})
            
            # Parse dates
            # Format: 2015-02-02 09:15:00+05:30
            # Pandas to_datetime is smart but can be slow. Specify format if possible?
            # ISO format is usually auto-detected fast.
            df['date'] = pd.to_datetime(df['date'], utc=True) 
            
            # Set index
            df = df.set_index('date')
            
            # Remove duplicates if any
            df = df[~df.index.duplicated(keep='first')]
            
            dfs.append(df)
            
            if (i+1) % 50 == 0:
                print(f"   Processed {i+1}/{len(csv_files)}...", flush=True)
                
        except Exception as e:
            print(f"⚠️ Error reading {fpath}: {e}")

    print("🔗 Concatenating and Aligning...")
    # Outer join to get union of all timestamps
    full_df = pd.concat(dfs, axis=1)
    
    print(f"Raw Shape: {full_df.shape}")
    
    # Sort index
    full_df = full_df.sort_index()
    
    # Forward Fill (Liquid logic handles irregular sampling, but for batch training we prefer aligned)
    # We limit ffill to avoid stale prices across days
    full_df = full_df.ffill(limit=12) # Fill up to 1 hour gap
    
    # Drop rows where > 50% stocks are missing (Market holidays/Closed)
    # Count non-NaN
    valid_counts = full_df.notna().sum(axis=1)
    threshold = full_df.shape[1] * 0.5
    full_df = full_df[valid_counts > threshold]
    
    # Fill remaining NaNs with 0?? Or specific value?
    # Log returns of fill-forward is 0 (good).
    # If we fill with mean, it adds noise.
    # Let's ffill again with no limit for the remaining, or drop columns.
    # Actually, let's just ffill completely for simplicity in this proto.
    full_df = full_df.ffill().bfill()
    
    print(f"Aligned Shape: {full_df.shape}")
    
    # 3. Log Returns & Volatility Targets
    print("📉 Calculating Log Returns...")
    # Log prices
    # Add epsilon? Prices are > 0.
    log_prices = np.log(full_df)
    
    # Diff
    log_returns = log_prices.diff().dropna()
    
    # Check for Infs (if price was 0 or negative?)
    # Replace Inf with NaN then drop or fill? 
    # Log Returns of 0 price change is 0. 
    # Log of 0 price is -Inf. If any price is 0, we get -Inf.
    # Replace -Inf/Inf with NaN, then dropNA.
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Clip extreme outliers (e.g. > 100% return in 5 mins is noise)
    # Clip at +/- 50% returns (log returns +/- 0.4)
    # Actually standard vol methodology uses 3-4 sigma.
    # Let's just hard clip to avoid explosions.
    log_returns = log_returns.clip(-0.5, 0.5)
    
    # 4. Generate Timestamps & Masks
    # Datetime Index
    dates = log_returns.index
    
    # Time Embeddings (Sin/Cos of TimeOfDay)
    # Market Open 9:15 to 15:30 = 375 minutes = 6.25 hours
    # We can use normalized minutes from midnight, or better:
    # fraction of trading day?
    # Let's use standard minute-of-day.
    
    # Convert to local time (IST is +5:30) if it's UTC. 
    # The CSVs said +05:30. pd.to_datetime(utc=True) makes it UTC.
    # So we should convert to IST for correct 9:15 check.
    dates_ist = dates.tz_convert('Asia/Kolkata')
    
    # Minute of day
    minutes = dates_ist.hour * 60 + dates_ist.minute
    # Normalize 0 to 1440 (24h) or 555 (9:15) to 930 (15:30)?
    # Cyclic on 24h cycle is safer.
    
    tod_sin = np.sin(2 * np.pi * minutes / 1440.0)
    tod_cos = np.cos(2 * np.pi * minutes / 1440.0)
    
    # Day of Week
    dow = dates_ist.dayofweek # 0=Mon, 6=Sun
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    
    time_emb = np.stack([tod_sin, tod_cos, dow_sin, dow_cos], axis=1) # (Steps, 4)
    
    # 5. Expiration Flag (Last Thursday of Month)
    print("🚩 Identifying Expiration Days...")
    is_expiry = np.zeros(len(dates), dtype=bool)
    
    # Get unique dates
    unique_days = np.unique(dates_ist.date)
    
    # Find last thursdays
    expiry_dates = set()
    from calendar import monthrange
    
    # Iterate through months present in data
    # Create a range of months
    start_date = dates_ist[0].date()
    end_date = dates_ist[-1].date()
    
    # Efficient way: For each unique month-year, find last thursday.
    # We can iterate unique (Year, Month) pairs.
    unique_months = sorted(list(set((d.year, d.month) for d in unique_days)))
    
    for y, m in unique_months:
        # Get all days in this month
        days_in_month = monthrange(y, m)[1]
        # Iterate backwards from end
        for d in range(days_in_month, 0, -1):
            try:
                date_obj = datetime(y, m, d).date()
                if date_obj.weekday() == 3: # Thursday
                    expiry_dates.add(date_obj)
                    break # Found the last one
            except:
                pass
                
    # Mark in array
    # This is slow if we iterate element-wise. Vectorize?
    # Create a Series and map.
    date_series = pd.Series(dates_ist.date)
    is_expiry = date_series.isin(expiry_dates).values
    
    print(f"Found {len(expiry_dates)} Expiration DMY. Marked {is_expiry.sum()} intraday steps.")

    # 6. Save Arrays
    # Train/Val/Test Split (70/10/20)
    L = len(log_returns)
    train_len = int(L * 0.7)
    val_len = int(L * 0.1)
    
    # Returns (X)
    X = log_returns.values.astype(np.float32)
    # Target (Y) = r^2
    Y = X ** 2
    
    # Splits
    train_x = X[:train_len]
    val_x   = X[train_len:train_len+val_len]
    test_x  = X[train_len+val_len:]
    
    train_y = Y[:train_len]
    val_y   = Y[train_len:train_len+val_len]
    test_y  = Y[train_len+val_len:]
    
    train_t = time_emb[:train_len]
    val_t   = time_emb[train_len:train_len+val_len]
    test_t  = time_emb[train_len+val_len:]
    
    train_exp = is_expiry[:train_len]
    val_exp   = is_expiry[train_len:train_len+val_len]
    test_exp  = is_expiry[train_len+val_len:]
    
    print("💾 Saving Datasets...")
    np.save(os.path.join(output_dir, "train_logret.npy"), train_x)
    np.save(os.path.join(output_dir, "val_logret.npy"), val_x)
    np.save(os.path.join(output_dir, "test_logret.npy"), test_x)
    
    np.save(os.path.join(output_dir, "train_vol_target.npy"), train_y)
    np.save(os.path.join(output_dir, "val_vol_target.npy"), val_y)
    np.save(os.path.join(output_dir, "test_vol_target.npy"), test_y)
    
    np.save(os.path.join(output_dir, "train_time_emb.npy"), train_t.astype(np.float32))
    np.save(os.path.join(output_dir, "val_time_emb.npy"), val_t.astype(np.float32))
    np.save(os.path.join(output_dir, "test_time_emb.npy"), test_t.astype(np.float32))
    
    np.save(os.path.join(output_dir, "train_expiry.npy"), train_exp)
    np.save(os.path.join(output_dir, "val_expiry.npy"), val_exp)
    np.save(os.path.join(output_dir, "test_expiry.npy"), test_exp)
    
    # Save Dates for reference
    # Convert index to string array?
    # np.save(os.path.join(output_dir, "train_dates.npy"), dates_ist[:train_len].astype(str))
    
    print("✅ Complete.")

if __name__ == "__main__":
    prepare_data_from_csv()
