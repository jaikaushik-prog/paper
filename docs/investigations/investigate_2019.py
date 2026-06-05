"""
INVESTIGATE 2019 ANOMALY
========================
User flagged 2019 returns (+119.5%) as suspiciously high compared to benchmark (+4.8%).
Objective: Deconstruct 2019 performance to identify the source of alpha.

Hypothesis:
1. Low Sample Size (1708 trades vs 15k typical).
2. Regime Artefact (Bull Low Vol = Uncapped Upside).
3. Specific Ticker Concentration (Adani / Bajaj).
"""

import os, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'investigation_2019.txt')

# ══════════════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════════════
def log(msg):
    print(msg)
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("INVESTIGATION REPORT: 2019 ANOMALY\n==================================\n")

# ══════════════════════════════════════════════════════════════════════════
#  LOAD & TRAIN (Simplified from audit logic)
# ══════════════════════════════════════════════════════════════════════════
def load_data():
    path = os.path.join(BASE_DIR, 'features_nifty500.parquet')
    log(f"Loading {path} ...")
    df = pd.read_parquet(path)
    df = df.sort_index(level='Date')
    df = df.dropna(subset=['target_binary', 'fwd_ret_5d', 'market_regime_val', 'atr_14'])
    df['target_binary'] = df['target_binary'].astype(int)
    return df

FEATURE_COLS = [
    'day_of_month', 'is_sip_window', 'day_of_week', 'is_monday',
    'market_regime_val', 'market_breadth', 'market_adx', 'market_vol_rank', 'market_volatility', 'is_high_vol',
    'rsi_14', 'dist_to_200ma', 'roc_5', 'vol_breakout', 'atr_14',
    'rs_20', 'rs_63', 'rs_rank_20', 'vol_rank_20', 'mom_vol_20'
]
TARGET_COL = 'target_binary'

# Only need Split 1 & 2 logic relevant to 2019
SPLITS = [
    {'name': 'Split 1', 'train': ('2015', '2018'), 'test': ('2019', '2019')},
]

def get_model():
    return XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
        use_label_encoder=False, random_state=42, n_jobs=-1, verbosity=0
    )

def analyze_2019(df):
    dates = df.index.get_level_values('Date')
    current_years = dates.year
    
    # Train heavily on pre-2019 to reproduce 2019 preds exactly
    log("Training Model for 2019 prediction...")
    tr_start, tr_end = '2015', '2018'
    te_start, te_end = '2019', '2019'
    
    train_mask = (current_years >= int(tr_start)) & (current_years <= int(tr_end))
    test_mask  = (current_years >= int(te_start)) & (current_years <= int(te_end))
    
    X_train = df.loc[train_mask, FEATURE_COLS].copy()
    y_train = df.loc[train_mask, TARGET_COL]
    X_test  = df.loc[test_mask, FEATURE_COLS].copy()
    
    # Scale
    scaler = RobustScaler()
    cols_to_scale = ['rsi_14', 'dist_to_200ma', 'roc_5', 'vol_breakout', 'rs_20', 'rs_63', 'mom_vol_20']
    valid_cols = [c for c in cols_to_scale if c in X_train.columns]
    if valid_cols:
        X_train[valid_cols] = scaler.fit_transform(X_train[valid_cols])
        X_test[valid_cols]  = scaler.transform(X_test[valid_cols])
        
    model = get_model()
    model.fit(X_train, y_train)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Create Test DF
    test_df = df.loc[test_mask, ['Open', 'High', 'Low', 'Close', 'fwd_ret_5d', 
                                 'target_binary', 'market_regime_val', 
                                 'atr_14', 'market_breadth']].copy()
    test_df['proba'] = y_proba
    test_df['ticker'] = test_df.index.get_level_values('Ticker')
    test_df['date'] = test_df.index.get_level_values('Date')
    
    # Filter Trades > 0.53
    trades = test_df[test_df['proba'] > 0.53].copy()
    
    log(f"\n2019 TRADE ANALYSIS")
    log(f"-------------------")
    log(f"Total Trades: {len(trades)}")
    
    # Regime Distribution
    log(f"\nRegime Distribution in 2019 Trades:")
    log(trades['market_regime_val'].value_counts().to_string())
    # 2=Bull Low Vol, 3=Bull High Vol, 1=Neutral, 0=Bear
    
    # Apply Logic (Standard)
    conditions = [
        (trades['market_regime_val'] == 2), 
        (trades['market_regime_val'] == 3), 
        (trades['market_regime_val'] == 1), 
        (trades['market_regime_val'] == 0)  
    ]
    choices = [1.5, 1.0, 0.9, 0.8]
    trades['k_factor'] = np.select(conditions, choices, default=0.9)
    
    trades['atr_monthly'] = trades['atr_14'] * np.sqrt(21)
    trades['sl_pct'] = -1.0 * (trades['k_factor'] * trades['atr_monthly']) / trades['Close']
    trades['sl_pct'] = trades['sl_pct'].clip(upper=-0.02, lower=-0.50)
    
    trades['tp_pct'] = np.where(trades['market_regime_val'].isin([2, 3]), 100.0, trades['sl_pct'].abs() * 2.0)
    
    profit_mask = trades['fwd_ret_5d'] >= trades['tp_pct']
    loss_mask   = trades['fwd_ret_5d'] <= trades['sl_pct']
    
    trades['trade_return'] = trades['fwd_ret_5d']
    trades.loc[profit_mask, 'trade_return'] = trades['tp_pct']
    trades.loc[loss_mask, 'trade_return'] = trades['sl_pct']
    trades['trade_return'] -= 0.002 # 20bps cost
    
    # Top Winners
    log(f"\nTop 20 Winners (2019):")
    top_winners = trades.sort_values('trade_return', ascending=False).head(20)
    for idx, row in top_winners.iterrows():
        log(f"{row['date'].date()} {row['ticker']:<15} Ret: {row['trade_return']:>6.1%}  Regime: {row['market_regime_val']}")
        
    # Ticker Concentration
    log(f"\nTicker Contribution (Sum of Returns):")
    ticker_contrib = trades.groupby('ticker')['trade_return'].sum().sort_values(ascending=False).head(10)
    log(ticker_contrib.to_string())

    # Sector? (Don't have sector data but ticker names give clue)
    
    # Monthly Breakdown
    trades['month'] = trades['date'].dt.month
    log(f"\nMonthly Performance 2019:")
    monthly = trades.groupby('month')['trade_return'].mean() * 100
    log(monthly.to_string())

    
def main():
    df = load_data()
    analyze_2019(df)

if __name__ == '__main__':
    main()
