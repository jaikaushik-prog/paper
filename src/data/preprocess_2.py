import polars as pl
import numpy as np
import glob
import os
import pickle

# --- CONFIGURATION ---
DATA_PATH = "./raw_data"
OUTPUT_PATH = "./processed_data"
start_date = "2015-01-01"
end_date = "2025-01-01"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def preprocess_with_polars():
    print("🚀 Starting Polars Turbo Processing...")

    # 1. Create Master Timeline (Lazy)
    # We generate a DataFrame with just the timestamps we want
    dates = pl.datetime_range(
        start=pl.lit(start_date).str.to_datetime(),
        end=pl.lit(end_date).str.to_datetime(),
        interval="1d",
        eager=True
    ).alias("day")
    
    # Filter weekends (approximate check using weekday)
    dates = dates.filter(dates.dt.weekday() < 6)

    # Create 5-min intervals for a single day
    times = pl.time_range(
        start=pl.time(9, 15),
        end=pl.time(15, 30),
        interval="5m",
        eager=True
    ).alias("time")

    # Cross join to create the full Grid (Days x Times)
    master_df = (
        dates.to_frame()
        .join(times.to_frame(), how="cross")
        .select(
            pl.col("day").dt.combine(pl.col("time")).alias("timestamp")
        )
        .sort("timestamp")
    )
    
    print(f"✅ Master Timeline Created: {master_df.height} rows")

    files = glob.glob(f"{DATA_PATH}/*.csv")
    valid_stocks = []
    processed_arrays = []
    
    # 2. Process Files (Polars is fast enough to loop, but we use Lazy API inside)
    # Note: For maximum speed, we process stock-by-stock to keep RAM low 
    # and stack directly into a list for the final Tensor.
    
    print(f"Processing {len(files)} files...")
    
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        
        try:
            # Lazy Load the CSV
            q = (
                pl.scan_csv(file)
                # Rename 'date' to 'timestamp'
                .rename({"date": "timestamp"}) 
                # FIX: Explicit format string with %z for timezone (+05:30)
                # AND: Remove timezone (.replace_time_zone(None)) to match Master Index
                .with_columns(
                    pl.col("timestamp")
                    .str.to_datetime(format="%Y-%m-%d %H:%M:%S%z")
                    .dt.replace_time_zone(None) 
                )
                # Ensure we only have the columns we need
                .select(["timestamp", "open", "high", "low", "close", "volume"])
            )

            # Left Join with Master Timeline
            joined_df = (
                master_df.lazy()
                .join(q, on="timestamp", how="left")
                .with_columns(pl.all().forward_fill())
                .fill_null(0)
                .collect() 
            )
            
            # Extract just the features as a numpy array
            stock_data = joined_df.select(["open", "high", "low", "close", "volume"]).to_numpy()
            
            processed_arrays.append(stock_data)
            valid_stocks.append(stock_name)

        except Exception as e:
            print(f"⚠️ Error with {stock_name}: {e}")

    # 3. Stack into 3D Tensor
    print("📦 Stacking 3D Tensor...")
    # List of (Time, 5) arrays -> (Time, Stocks, 5)
    # We use swapaxes to get (Time, Stocks, Features) because default stack is (Stocks, Time, Features)
    data_tensor = np.stack(processed_arrays, axis=1)
    print(f"Final Shape: {data_tensor.shape}")

    # 4. Generate Adjacency Matrix
    print("🕸️ Generating Correlation Graph...")
    
    # We need a matrix of just Closing prices to run correlation
    # We can reconstruct this from our list of arrays to avoid storing a huge DF
    close_prices = np.stack([arr[:, 3] for arr in processed_arrays], axis=1) # Index 3 is Close
    
    # Polars correlation is fast, but numpy is fine for the final matrix
    # We convert the numpy array to a Polars DF just to use its fast corr method if we wanted,
    # but numpy.corrcoef is highly optimized for dense matrices.
    corr_matrix = np.corrcoef(close_prices, rowvar=False) # rowvar=False means columns are variables (stocks)
    
    adj_mx = np.where(corr_matrix > 0.6, 1, 0)

    # 5. Save
    with open(f"{OUTPUT_PATH}/data.pkl", "wb") as f:
        pickle.dump(data_tensor, f)
        
    with open(f"{OUTPUT_PATH}/adj_mx.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
        
    with open(f"{OUTPUT_PATH}/stock_names.txt", "w") as f:
        f.write("\n".join(valid_stocks))

    print("✅ Done. Polars Speed.")

if __name__ == "__main__":
    preprocess_with_polars()