
import pandas as pd
import os
import glob

def test():
    print("🔍 Testing single file read...")
    raw_dir = "datasets/raw_Data"
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not files:
        print("❌ No files found in datasets/raw_Data")
        # Try raw_Data
        raw_dir = "raw_Data"
        files = glob.glob(os.path.join(raw_dir, "*.csv"))
        
    if not files:
        print("❌ No files found in raw_Data either.")
        return

    f = files[0]
    print(f"📄 Reading {f}...")
    try:
        df = pd.read_csv(f)
        print("✅ Read successful.")
        print(df.head())
        print(f"Columns: {df.columns}")
        
        # Test Date Parse
        print("Parsing date...")
        df['date'] = pd.to_datetime(df['date'], utc=True)
        print("✅ Date parse successful.")
        print(df.dtypes)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test()
