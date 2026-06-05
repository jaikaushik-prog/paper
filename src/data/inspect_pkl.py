
import pickle
import pandas as pd
import numpy as np
import os

def check():
    path = "datasets/Nifty500/data.pkl"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"Loading {path}...")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        
        if isinstance(data, pd.DataFrame):
            print("It is a DataFrame.")
            print(f"Index: {data.index}")
            print(f"Columns: {data.columns}")
            print(data.head())
        elif isinstance(data, np.ndarray):
            print("It is a NumPy Array.")
            print(f"Shape: {data.shape}")
        else:
            print("Unknown type.")
            
    except Exception as e:
        print(f"Error loading: {e}")

if __name__ == "__main__":
    check()
