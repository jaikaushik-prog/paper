
import numpy as np
import os

def check():
    print("🔍 Checking datasets for NaNs/Infs...")
    files = ["train_logret.npy", "train_vol_target.npy", "train_time_emb.npy"]
    data_dir = "datasets/Nifty500"
    
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"⚠️ {f} not found.")
            continue
            
        data = np.load(path)
        if np.isnan(data).any():
            print(f"❌ {f} CONTAINS NaNs! Count: {np.isnan(data).sum()}")
        elif np.isinf(data).any():
            print(f"❌ {f} CONTAINS Infs! Count: {np.isinf(data).sum()}")
        else:
            print(f"✅ {f} is clean. Range: [{data.min():.4f}, {data.max():.4f}]")

if __name__ == "__main__":
    check()
