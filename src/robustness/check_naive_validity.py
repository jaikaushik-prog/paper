
import numpy as np

def check_integrity():
    print("📦 Loading Validation Data...")
    try:
        # Load raw validation data (Steps, Nodes)
        data = np.load("datasets/Nifty500/val_data.npy")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print(f"    Shape: {data.shape} (Steps, Nodes)")
    
    # Calculate Naive Error (Predicting No Change)
    # Target: Price at t
    # Prediction: Price at t-1
    # Error = |P_t - P_{t-1}|
    
    # Slice to match the effective validation range usually used (skipping first step)
    targets = data[1:]
    predictions = data[:-1]
    
    # Absolute Error per element
    abs_error = np.abs(targets - predictions)
    
    # Mean across all steps and all nodes
    naive_mae = np.mean(abs_error)
    
    print("-" * 40)
    print(f"🧐 DATA INTEGRITY CHECK")
    print("-" * 40)
    print(f"NAIVE BASELINE MAE: {naive_mae:.4f} INR")
    print(f"(Average price movement per 5-min candle)")
    print("-" * 40)
    print(f"Comparing against reported models:")
    print(f"1. STID Baseline:   4.4500 INR")
    print(f"2. Liquid-KAN:      4.1426 INR")
    
    if naive_mae < 4.45:
        print("\n⚠️  WARNING: STID Baseline is WORSE than Naive!")
        print("    The models might be failing to learn anything useful.")
    else:
        print(f"\n✅ VALIDITY CONFIRMED.")
        print(f"    STID is beating Naive by: {naive_mae - 4.45:.2f} INR")
        print(f"    Liquid is beating Naive by: {naive_mae - 4.14:.2f} INR")
        
if __name__ == "__main__":
    check_integrity()
