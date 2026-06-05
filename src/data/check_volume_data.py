import pandas as pd
import os

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

def check_data():
    for regime in ['high_vol', 'low_vol']:
        path = os.path.join(RESULTS_DIR, f"surface_minnows_{regime}.csv")
        if not os.path.exists(path): continue
        
        df = pd.read_csv(path)
        # Filter for Strong Bullish Push (z_push > 2.0) and Lag 10
        subset = df[(df['push_bin'] > 2.0) & (df['lag'] == 10)]
        avg_resp = subset['avg_response'].mean()
        
        print(f"[{regime.upper()}] Z_Push>2.0, Lag 10 => Mean Response: {avg_resp:.4f}")

if __name__ == "__main__":
    check_data()
