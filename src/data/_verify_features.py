import pandas as pd
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', '{:.4f}'.format)

df = pd.read_parquet('features_nifty500.parquet')
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Unique tickers: {df.index.get_level_values('Ticker').nunique()}")
print(f"Date range: {df.index.get_level_values('Date').min().date()} -> {df.index.get_level_values('Date').max().date()}")
print()

t = df['target_binary']
print("Target distribution:")
print(f"  target_binary=1: {(t==1).sum():>8,} ({(t==1).mean()*100:.1f}%)")
print(f"  target_binary=0: {(t==0).sum():>8,} ({(t==0).mean()*100:.1f}%)")
print()

s1 = df.loc[df['is_sip_window']==1, 'fwd_ret_5d'].mean()
s0 = df.loc[df['is_sip_window']==0, 'fwd_ret_5d'].mean()
print("SIP Window verification:")
print(f"  SIP window mean 5d fwd ret : {s1*100:.3f}%")
print(f"  Non-SIP mean 5d fwd ret    : {s0*100:.3f}%")
print(f"  Ratio                      : {s1/s0:.2f}x")
print()

print("First 5 rows:")
print(df.head(5))
print()

print("Scaled features stats:")
print(df[['rsi_14','dist_to_200ma','roc_5','vol_breakout']].describe())
