"""
INVESTIGATE PI INDUSTRIES (PIIND) JAN 2019
==========================================
Suspected Data Error: PIIND shows +99.8% return on Jan 3-4, 2019.
"""
import os, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, 'features_nifty500.parquet')

print(f"Loading {path} ...")
df = pd.read_parquet(path)
df = df.sort_index(level='Date')

# Filter for PIIND in Jan 2019
piind = df.xs('PIIND', level='Ticker')
data_2019 = piind.loc['2018-12-25':'2019-01-15', ['Close', 'fwd_ret_5d']]

print("\nPIIND Data (Dec 2018 - Jan 2019):")
print(data_2019)
