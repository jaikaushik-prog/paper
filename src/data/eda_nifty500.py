"""
Nifty 500 Comprehensive EDA Analysis  (Polars-accelerated)
===========================================================
Covers all 7 phases:
  1. Data Integrity (missing values, flash crashes, zero volume)
  2. Distribution Analysis (fat tails, kurtosis, sector volatility)
  3. Time-Series Structure (ADF stationarity, seasonality)
  4. Correlation & Clustering (rolling correlation, sector matrix, dendrogram)
  5. Liquidity Stress Test (turnover filter, spread proxy)
  6. Market Regime Classification (volatility regimes, trend regimes)
  7. Tail Risk & Calendar Alpha (drawdown, turn-of-month, pre-budget)

Data: 499 Nifty 500 stocks, 5-min intraday OHLCV (2015-2025)
Uses Polars for fast CSV loading, converts to Pandas for analysis/plotting.
"""

import os, json, time, warnings
import numpy as np
import polars as pl
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='deep')
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 11})

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, 'raw_Data')
PLOT_DIR    = os.path.join(BASE_DIR, 'plots', 'eda')
SECTOR_FILE = os.path.join(BASE_DIR, 'sector_mappings.json')
os.makedirs(PLOT_DIR, exist_ok=True)

def load_sector_mappings():
    with open(SECTOR_FILE, 'r') as f:
        data = json.load(f)
    data.pop('comment', None)
    t2s = {}
    for sector, tickers in data.items():
        for t in tickers:
            t2s[t] = sector
    return data, t2s

# ══════════════════════════════════════════════════════════════════════════
#  STEP 0 : LOAD ALL DATA FAST WITH POLARS, RESAMPLE TO DAILY
# ══════════════════════════════════════════════════════════════════════════
def load_all_daily():
    csv_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} stock CSV files. Loading with Polars ...")
    t0 = time.time()

    required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
    frames = []
    skipped = []
    for i, fname in enumerate(csv_files):
        ticker = fname.replace('.csv', '')
        fpath = os.path.join(RAW_DIR, fname)
        try:
            # Quick header check — only load files with OHLCV columns
            with open(fpath, 'r') as fh:
                header = set(fh.readline().strip().split(','))
            if not required_cols.issubset(header):
                skipped.append(ticker)
                continue
            lf = pl.scan_csv(fpath, try_parse_dates=True)
            daily = (
                lf
                .with_columns(pl.col('date').cast(pl.Datetime('us')).dt.date().alias('trade_date'))
                .group_by('trade_date')
                .agg([
                    pl.col('open').first().alias('open'),
                    pl.col('high').max().alias('high'),
                    pl.col('low').min().alias('low'),
                    pl.col('close').last().alias('close'),
                    pl.col('volume').sum().alias('volume'),
                ])
                .with_columns(pl.lit(ticker).alias('ticker'))
                .sort('trade_date')
                .collect()
            )
            frames.append(daily)
        except Exception as e:
            skipped.append(ticker)
            print(f"  SKIP {ticker}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(csv_files)} ... ({time.time()-t0:.1f}s)")
    if skipped:
        print(f"  Skipped {len(skipped)} non-stock files: {skipped}")

    print(f"  All CSVs loaded in {time.time()-t0:.1f}s. Pivoting to wide panels ...")

    # Concat into one tall Polars DataFrame
    big = pl.concat(frames)
    big_pd = big.to_pandas()
    big_pd['trade_date'] = pd.to_datetime(big_pd['trade_date'])
    big_pd.set_index('trade_date', inplace=True)

    # Pivot to wide format: index=date, columns=ticker
    close_df    = big_pd.pivot_table(values='close',  index=big_pd.index, columns='ticker')
    volume_df   = big_pd.pivot_table(values='volume', index=big_pd.index, columns='ticker')
    high_df     = big_pd.pivot_table(values='high',   index=big_pd.index, columns='ticker')
    low_df      = big_pd.pivot_table(values='low',    index=big_pd.index, columns='ticker')
    open_df     = big_pd.pivot_table(values='open',   index=big_pd.index, columns='ticker')
    turnover_df = (close_df * volume_df) / 1e7  # In Crores

    # ── Data Cleaning: replace zero close prices with NaN ──
    close_df = close_df.replace(0, np.nan)
    high_df  = high_df.replace(0, np.nan)
    low_df   = low_df.replace(0, np.nan)
    open_df  = open_df.replace(0, np.nan)

    print(f"\nDaily panel: {close_df.shape[0]} trading days × {close_df.shape[1]} stocks")
    print(f"Date range : {close_df.index.min().date()} → {close_df.index.max().date()}")
    print(f"Total load time: {time.time()-t0:.1f}s")
    return close_df, volume_df, high_df, low_df, open_df, turnover_df


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 1 : DATA INTEGRITY
# ══════════════════════════════════════════════════════════════════════════
def phase1_integrity(close_df, volume_df):
    print("\n" + "="*70)
    print("  PHASE 1 : DATA INTEGRITY CHECK")
    print("="*70)

    # 1a. Missing value heatmap
    missing_pct = close_df.isnull().sum() / len(close_df) * 100
    total_missing = close_df.isnull().sum().sum() / close_df.size * 100
    print(f"Overall missing data: {total_missing:.2f}%")
    print(f"Stocks with >50% missing: {(missing_pct > 50).sum()}")
    print(f"Stocks fully present   : {(missing_pct == 0).sum()}")

    subset = close_df.iloc[:, :50]
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(subset.isnull().T, cbar=False, cmap='YlOrRd', yticklabels=True, ax=ax)
    ax.set_title('Missing Data Heatmap (First 50 Stocks)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date Index'); ax.set_ylabel('Ticker')
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '01_missing_data_heatmap.png'), dpi=150); plt.close(fig)
    print("  → 01_missing_data_heatmap.png")

    # 1b. Flash Crash / Split Detection
    returns = close_df.pct_change()
    # Clean returns: cap at ±90% to remove inf and extreme artifacts
    returns = returns.clip(-0.9, 10.0)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    extreme_counts = (returns.abs() > 0.20).sum()
    flash_stocks = extreme_counts[extreme_counts > 0].sort_values(ascending=False)
    print(f"\nFlash Crash Detection (|daily return| > 20%):")
    print(f"  Stocks with ≥1 extreme day : {len(flash_stocks)}")
    print(f"  Total extreme events       : {extreme_counts.sum()}")
    if len(flash_stocks) > 0:
        print(f"  Top 10 worst offenders:")
        for tkr, cnt in flash_stocks.head(10).items():
            print(f"    {tkr:16s} : {cnt} extreme days")

    fig, ax = plt.subplots(figsize=(12, 5))
    flash_stocks.head(30).plot(kind='bar', ax=ax, color='crimson', edgecolor='black')
    ax.set_title('Flash Crash / Potential Split Events (|Return| > 20%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Extreme Days'); ax.set_xlabel('Ticker')
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '02_flash_crash_detection.png'), dpi=150); plt.close(fig)
    print("  → 02_flash_crash_detection.png")

    # 1c. Zero Volume Trap
    zero_vol_pct = (volume_df == 0).sum() / volume_df.notna().sum() * 100
    illiquid = zero_vol_pct[zero_vol_pct > 30].sort_values(ascending=False)
    print(f"\nZero Volume Trap:")
    print(f"  Stocks with >30% zero-volume days: {len(illiquid)}")
    if len(illiquid) > 0:
        for tkr, pct in illiquid.head(10).items():
            print(f"    {tkr:16s} : {pct:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(zero_vol_pct, bins=50, kde=True, color='steelblue', ax=ax)
    ax.axvline(x=30, color='red', linestyle='--', linewidth=2, label='30% Threshold')
    ax.set_title('Zero Volume Day Distribution Across Stocks', fontsize=13, fontweight='bold')
    ax.set_xlabel('% of Days with Zero Volume'); ax.set_ylabel('Number of Stocks'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '03_zero_volume_distribution.png'), dpi=150); plt.close(fig)
    print("  → 03_zero_volume_distribution.png")

    return returns, illiquid.index.tolist()


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 2 : DISTRIBUTION ANALYSIS (FAT TAILS)
# ══════════════════════════════════════════════════════════════════════════
def phase2_distribution(close_df, returns, sector_map, ticker_to_sector):
    print("\n" + "="*70)
    print("  PHASE 2 : DISTRIBUTION ANALYSIS (FAT TAIL HUNT)")
    print("="*70)

    # 2a. Log Returns Histogram vs Normal
    log_returns = np.log(close_df / close_df.shift(1))
    all_ret = log_returns.values.flatten()
    all_ret = all_ret[np.isfinite(all_ret)]  # Remove NaN AND inf
    kurtosis_val = pd.Series(all_ret).kurtosis()
    skewness_val = pd.Series(all_ret).skew()
    print(f"Universe Log Returns:  Mean={np.mean(all_ret):.6f}  Std={np.std(all_ret):.6f}")
    print(f"  Skewness : {skewness_val:.4f}")
    print(f"  Kurtosis : {kurtosis_val:.2f}  {'→ LEPTOKURTIC: Use RobustScaler!' if kurtosis_val > 3 else '→ Near Normal'}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(all_ret, bins=200, kde=True, stat='density', label='Nifty 500 Log Returns',
                 color='steelblue', alpha=0.6, ax=ax)
    mu, std = np.mean(all_ret), np.std(all_ret)
    x = np.linspace(mu - 4*std, mu + 4*std, 300)
    ax.plot(x, 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2)),
            linewidth=2.5, color='red', label='Normal Distribution')
    ax.set_xlim(-0.10, 0.10)
    ax.set_title(f'Return Distribution  |  Kurtosis: {kurtosis_val:.2f}  |  Skew: {skewness_val:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Daily Log Return'); ax.legend(fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '04_return_distribution_vs_normal.png'), dpi=150); plt.close(fig)
    print("  → 04_return_distribution_vs_normal.png")

    # 2b. Sector Volatility Box Plots
    annual_vol = returns.std() * np.sqrt(252)
    vol_df = pd.DataFrame({'Ticker': annual_vol.index, 'Annual_Vol': annual_vol.values})
    vol_df['Sector'] = vol_df['Ticker'].map(ticker_to_sector).fillna('Other')
    vol_df = vol_df[vol_df['Sector'] != 'Other']

    fig, ax = plt.subplots(figsize=(14, 6))
    order = vol_df.groupby('Sector')['Annual_Vol'].median().sort_values(ascending=False).index
    sns.boxplot(data=vol_df, x='Sector', y='Annual_Vol', order=order, palette='Set2', ax=ax, fliersize=3)
    ax.set_title('Annual Volatility by Sector', fontsize=13, fontweight='bold')
    ax.set_ylabel('Annualized Volatility'); ax.set_xlabel('')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '05_sector_volatility_boxplot.png'), dpi=150); plt.close(fig)
    print("  → 05_sector_volatility_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 3 : TIME-SERIES STRUCTURE
# ══════════════════════════════════════════════════════════════════════════
def phase3_timeseries(close_df, returns):
    print("\n" + "="*70)
    print("  PHASE 3 : TIME-SERIES STRUCTURE (STATIONARITY & SEASONALITY)")
    print("="*70)

    # 3a. ADF Tests
    test_tickers = [t for t in ['RELIANCE','HDFCBANK','INFY','TCS','TATAMOTORS'] if t in close_df.columns]
    if not test_tickers: test_tickers = list(close_df.columns[:5])

    print(f"\nADF Stationarity Tests:")
    print(f"{'Ticker':>14s}  {'Price p-val':>12s}  {'Return p-val':>13s}  {'Price':>10s}  {'Return':>10s}")
    print("-"*65)
    for tkr in test_tickers:
        ps = close_df[tkr].dropna()
        rs = ps.pct_change().dropna()
        try:
            pp = adfuller(ps, maxlag=20, autolag='AIC')[1]
            pr = adfuller(rs, maxlag=20, autolag='AIC')[1]
            lp = 'Non-Stat ✓' if pp >= 0.05 else 'Stationary'
            lr = 'Stationary ✓' if pr < 0.05 else 'Non-Stat ✗'
            print(f"{tkr:>14s}  {pp:>12.6f}  {pr:>13.6f}  {lp:>10s}  {lr:>10s}")
        except Exception as e:
            print(f"{tkr:>14s}  ERROR: {e}")

    # 3b/3c. Day-of-Week + Month-of-Year
    avg_ret = returns.mean(axis=1).dropna()
    dow = avg_ret.groupby(avg_ret.index.dayofweek).mean()
    dow_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri'}
    print(f"\nDay-of-Week Mean Returns (bps):")
    for d, v in dow.items():
        if d in dow_names:
            print(f"  {dow_names[d]} : {v*10000:+.2f} bps")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    d_vals = [dow.get(i, 0)*10000 for i in range(5)]
    d_cols = ['#e74c3c' if v < 0 else '#27ae60' for v in d_vals]
    axes[0].bar(['Mon','Tue','Wed','Thu','Fri'], d_vals, color=d_cols, edgecolor='black')
    axes[0].set_title('Day-of-Week Effect (bps)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Return (bps)'); axes[0].axhline(0, color='black', linewidth=0.8)

    monthly = avg_ret.groupby(avg_ret.index.month).mean()
    mn = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    m_vals = [monthly.get(i, 0)*10000 for i in range(1, 13)]
    m_cols = ['#e74c3c' if v < 0 else '#27ae60' for v in m_vals]
    axes[1].bar([mn[i] for i in range(1,13)], m_vals, color=m_cols, edgecolor='black')
    axes[1].set_title('Month-of-Year Effect (bps)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Return (bps)'); axes[1].axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=30)

    fig.suptitle('Calendar Seasonality Analysis', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, '06_seasonality_effects.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    print("  → 06_seasonality_effects.png")


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 4 : CORRELATION & CLUSTERING
# ══════════════════════════════════════════════════════════════════════════
def phase4_correlation(close_df, returns, sector_map, ticker_to_sector):
    print("\n" + "="*70)
    print("  PHASE 4 : CORRELATION & CLUSTERING")
    print("="*70)

    # 4a. Rolling Correlation with Market
    market_ret = returns.mean(axis=1)
    window = 126
    sample = returns.columns[:min(200, returns.shape[1])]
    corr_panel = returns[sample].rolling(window).corr(market_ret)
    avg_corr = corr_panel.mean(axis=1).dropna()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(avg_corr.index, avg_corr.values, color='purple', alpha=0.7, linewidth=0.8)
    ax.axhline(y=avg_corr.mean(), color='orange', linestyle='--', linewidth=1.5,
               label=f'Mean = {avg_corr.mean():.3f}')
    ax.fill_between(avg_corr.index, avg_corr.values, alpha=0.2, color='purple')
    ax.set_title('Avg Rolling Correlation with Market (126-day)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Correlation'); ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '07_rolling_correlation_market.png'), dpi=150); plt.close(fig)
    print(f"  → 07_rolling_correlation_market.png")
    print(f"  Mean corr: {avg_corr.mean():.3f}, Max: {avg_corr.max():.3f} on {avg_corr.idxmax().date()}")

    # 4b. Sector Correlation Matrix
    sector_returns = {}
    for sector, tickers in sector_map.items():
        valid = [t for t in tickers if t in returns.columns]
        if len(valid) >= 3:
            sector_returns[sector] = returns[valid].mean(axis=1)
    sector_corr = pd.DataFrame(sector_returns).dropna().corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(sector_corr, dtype=bool), k=1)
    sns.heatmap(sector_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                mask=mask, square=True, linewidths=0.5, ax=ax, vmin=-0.5, vmax=1.0)
    ax.set_title('Sector Correlation Matrix', fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '08_sector_correlation_matrix.png'), dpi=150); plt.close(fig)
    print("  → 08_sector_correlation_matrix.png")

    # 4c. Hierarchical Clustering
    recent_ret = returns.tail(252).dropna(axis=1, how='any')
    if recent_ret.shape[1] > 150:
        recent_ret = recent_ret[recent_ret.std().nlargest(150).index]
    corr_mat = recent_ret.corr()
    dist_mat = np.sqrt(2 * (1 - corr_mat))
    np.fill_diagonal(dist_mat.values, 0)
    linked = linkage(dist_mat, method='ward')

    fig, ax = plt.subplots(figsize=(18, 7))
    dendrogram(linked, orientation='top', labels=corr_mat.columns.tolist(),
               distance_sort='descending', leaf_rotation=90, leaf_font_size=6, ax=ax,
               color_threshold=0.7*max(linked[:,2]))
    ax.set_title('Hierarchical Clustering of Nifty 500 (Ward Linkage)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Stocks'); ax.set_ylabel('Distance')
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '09_hierarchical_clustering.png'), dpi=150); plt.close(fig)
    print("  → 09_hierarchical_clustering.png")


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 5 : LIQUIDITY STRESS TEST
# ══════════════════════════════════════════════════════════════════════════
def phase5_liquidity(close_df, volume_df, turnover_df, high_df, low_df):
    print("\n" + "="*70)
    print("  PHASE 5 : LIQUIDITY STRESS TEST")
    print("="*70)

    median_turnover = turnover_df.median()
    liquid_stocks = median_turnover[median_turnover > 5].index.tolist()
    print(f"Total Stocks           : {len(median_turnover)}")
    print(f"Investable (>₹5Cr/day) : {len(liquid_stocks)}")
    print(f"Illiquid   (≤₹5Cr/day) : {len(median_turnover) - len(liquid_stocks)}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    capped = median_turnover[median_turnover < 200]
    sns.histplot(capped, bins=60, kde=True, color='teal', ax=axes[0])
    axes[0].axvline(x=5, color='red', linestyle='--', linewidth=2, label='₹5Cr Threshold')
    axes[0].set_title('Median Daily Turnover Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Median Daily Turnover (₹ Crore)'); axes[0].legend()

    top30 = median_turnover.nlargest(30)
    axes[1].barh(range(len(top30)), top30.values, color='teal', edgecolor='black')
    axes[1].set_yticks(range(len(top30))); axes[1].set_yticklabels(top30.index, fontsize=8)
    axes[1].set_title('Top 30 Most Liquid Stocks', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Median Daily Turnover (₹ Crore)'); axes[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '10_liquidity_turnover.png'), dpi=150); plt.close(fig)
    print("  → 10_liquidity_turnover.png")

    # 5b. Spread Proxy
    spread_proxy = ((high_df - low_df) / close_df).median() * 100
    print(f"\nSpread Proxy (Median Daily Range %): Universe median = {spread_proxy.median():.2f}%")
    ghost_mask = (spread_proxy < 0.5) & (median_turnover < 2)
    print(f"  'Ghost' stocks (range<0.5% & turnover<₹2Cr): {ghost_mask.sum()}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(median_turnover, spread_proxy, alpha=0.4, s=15, c='steelblue')
    ax.axhline(y=0.5, color='orange', linestyle='--', label='0.5% range')
    ax.axvline(x=5, color='red', linestyle='--', label='₹5Cr turnover')
    ax.set_xlabel('Median Daily Turnover (₹ Crore)'); ax.set_ylabel('Median Daily Range %')
    ax.set_title('Liquidity vs Spread Proxy', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 200); ax.set_ylim(0, 8); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '11_liquidity_vs_spread.png'), dpi=150); plt.close(fig)
    print("  → 11_liquidity_vs_spread.png")

    return liquid_stocks


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 6 : MARKET REGIME CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
def phase6_regimes(close_df, returns, liquid_stocks):
    print("\n" + "="*70)
    print("  PHASE 6 : MARKET REGIME CLASSIFICATION")
    print("="*70)

    liquid_cols = [c for c in liquid_stocks if c in close_df.columns]
    market_price = close_df[liquid_cols].mean(axis=1).dropna()
    market_ret = market_price.pct_change().dropna()

    # 6a. Volatility Regimes
    rolling_vol = market_ret.rolling(21).std() * np.sqrt(252) * 100

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(rolling_vol.index, rolling_vol.values, color='purple', alpha=0.7, linewidth=0.8)
    axes[0].axhline(y=12, color='green', linestyle='--', linewidth=1.5, label='Low Vol (<12%)')
    axes[0].axhline(y=20, color='red', linestyle='--', linewidth=1.5, label='High Vol (>20%)')
    axes[0].fill_between(rolling_vol.index, 0, 12, alpha=0.1, color='green')
    axes[0].fill_between(rolling_vol.index, 12, 20, alpha=0.1, color='yellow')
    axes[0].fill_between(rolling_vol.index, 20, rolling_vol.max()*1.1, alpha=0.1, color='red')
    axes[0].set_title('Market Volatility Regime (21-Day Annualized)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Volatility (%)'); axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, min(float(rolling_vol.max())*1.2, 80))

    rv = rolling_vol.dropna()
    low_v = (rv < 12).sum(); norm_v = ((rv >= 12) & (rv <= 20)).sum(); high_v = (rv > 20).sum()
    total = low_v + norm_v + high_v
    print(f"Volatility Regime Distribution:")
    print(f"  Low  (<12%)  : {low_v:5d} days ({low_v/total*100:.1f}%)")
    print(f"  Normal       : {norm_v:5d} days ({norm_v/total*100:.1f}%)")
    print(f"  High (>20%)  : {high_v:5d} days ({high_v/total*100:.1f}%)")

    # 6b. Trend Regime (200 DMA)
    dma200 = market_price.rolling(200).mean()
    dma200_slope = dma200.diff(20)
    bull = (market_price > dma200) & (dma200_slope > 0)
    bear = (market_price < dma200) & (dma200_slope < 0)

    axes[1].plot(market_price.index, market_price.values, color='steelblue', alpha=0.8, linewidth=0.8, label='Market Price')
    axes[1].plot(dma200.index, dma200.values, color='orange', linewidth=1.5, label='200 DMA')
    axes[1].set_title('Trend Regime (Price vs 200 DMA)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Market Index Level'); axes[1].legend(loc='upper left')
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    bd = bull.sum(); bed = bear.sum(); cd = len(bull) - bd - bed
    total_t = bd + bed + cd
    print(f"\nTrend Regime Distribution:")
    print(f"  Bull : {bd:5d} days ({bd/total_t*100:.1f}%)")
    print(f"  Bear : {bed:5d} days ({bed/total_t*100:.1f}%)")
    print(f"  Chop : {cd:5d} days ({cd/total_t*100:.1f}%)")

    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '12_market_regimes.png'), dpi=150); plt.close(fig)
    print("  → 12_market_regimes.png")


# ══════════════════════════════════════════════════════════════════════════
#  PHASE 7 : TAIL RISK & CALENDAR ALPHA
# ══════════════════════════════════════════════════════════════════════════
def phase7_tail_risk(close_df, returns):
    print("\n" + "="*70)
    print("  PHASE 7 : TAIL RISK & CALENDAR ALPHA")
    print("="*70)

    # 7a. Maximum Drawdown
    mdd_per_stock = {}
    for col in close_df.columns:
        s = close_df[col].dropna()
        if len(s) > 252:
            dd = s / s.expanding().max() - 1
            mdd_per_stock[col] = dd.min()
    mdd_series = pd.Series(mdd_per_stock).sort_values()
    torpedo = mdd_series[mdd_series < -0.50]
    print(f"Max Drawdown Analysis:")
    print(f"  Universe median MDD: {mdd_series.median()*100:.1f}%")
    print(f"  'Torpedo' stocks (>50%): {len(torpedo)}")
    if len(torpedo) > 0:
        print(f"  Top 15 worst:")
        for tkr, val in torpedo.head(15).items():
            print(f"    {tkr:16s} : {val*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.histplot(mdd_series*100, bins=50, kde=True, color='crimson', ax=axes[0])
    axes[0].axvline(x=-50, color='black', linestyle='--', linewidth=2, label='-50% Threshold')
    axes[0].set_title('Distribution of Maximum Drawdowns', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Max Drawdown (%)'); axes[0].legend()

    worst20 = mdd_series.head(20)
    axes[1].barh(range(len(worst20)), worst20.values*100, color='crimson', edgecolor='black')
    axes[1].set_yticks(range(len(worst20))); axes[1].set_yticklabels(worst20.index, fontsize=8)
    axes[1].set_title('Top 20 "Torpedo" Stocks', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Max Drawdown (%)'); axes[1].invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '13_maximum_drawdowns.png'), dpi=150); plt.close(fig)
    print("  → 13_maximum_drawdowns.png")

    # 7b. Turn-of-Month + Pre-Budget
    avg_ret = returns.mean(axis=1).dropna()
    day_of_month = avg_ret.index.day
    is_tom = (day_of_month <= 5) | (day_of_month >= 27)
    tom_mean = avg_ret[is_tom].mean() * 10000
    mid_mean = avg_ret[~is_tom].mean() * 10000
    print(f"\nTurn-of-Month Effect:")
    print(f"  Turn-of-Month : {tom_mean:+.2f} bps/day")
    print(f"  Mid-Month     : {mid_mean:+.2f} bps/day")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    labels = ['Turn of Month\n(Day 27-31 & 1-5)', 'Mid Month\n(Day 6-26)']
    means = [tom_mean, mid_mean]
    colors = ['#27ae60' if m > 0 else '#e74c3c' for m in means]
    axes[0].bar(labels, means, color=colors, edgecolor='black', width=0.5)
    axes[0].set_title('Turn-of-Month Effect (SIP Flow Hypothesis)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Mean Daily Return (bps)'); axes[0].axhline(0, color='black', linewidth=0.8)

    # Pre-Budget
    rv_mkt = avg_ret.rolling(21).std() * np.sqrt(252) * 100
    jan_v = rv_mkt[rv_mkt.index.month == 1].dropna()
    feb_v = rv_mkt[rv_mkt.index.month == 2].dropna()
    rest_v = rv_mkt[(rv_mkt.index.month != 1) & (rv_mkt.index.month != 2)].dropna()
    vol_data = pd.DataFrame({
        'Period': (['Jan (Pre-Budget)']*len(jan_v) + ['Feb (Budget Month)']*len(feb_v) + ['Rest of Year']*len(rest_v)),
        'Volatility': pd.concat([jan_v, feb_v, rest_v]).values
    })
    sns.boxplot(data=vol_data, x='Period', y='Volatility', palette=['#f39c12','#e74c3c','#3498db'], ax=axes[1])
    axes[1].set_title('Pre-Budget Volatility Analysis', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('21-Day Rolling Volatility (%)'); axes[1].set_xlabel('')
    print(f"\nPre-Budget Volatility:")
    print(f"  Jan median: {jan_v.median():.1f}%  |  Feb: {feb_v.median():.1f}%  |  Rest: {rest_v.median():.1f}%")
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '14_calendar_alpha.png'), dpi=150); plt.close(fig)
    print("  → 14_calendar_alpha.png")

    # 7c. Universe Rolling Volatility with Event Annotations
    universe_vol = returns.rolling(22).std().mean(axis=1) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(universe_vol.index, universe_vol.values, color='darkblue', alpha=0.7, linewidth=0.8)
    ax.fill_between(universe_vol.index, universe_vol.values, alpha=0.2, color='steelblue')
    ax.set_title('Nifty 500: Average Rolling Volatility (22-Day Annualized)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Annualized Volatility')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for ds, lbl in {'2016-11-08':'Demonetization','2018-09-21':'IL&FS Crisis',
                     '2020-03-23':'COVID Crash','2022-02-24':'Russia-Ukraine'}.items():
        try:
            dt = pd.Timestamp(ds)
            nearest = universe_vol.index[universe_vol.index.get_indexer([dt], method='nearest')[0]]
            val = universe_vol.loc[nearest]
            ax.annotate(lbl, xy=(nearest, val), xytext=(nearest, val+0.05),
                       fontsize=8, ha='center', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
        except: pass
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, '15_volatility_clustering.png'), dpi=150); plt.close(fig)
    print("  → 15_volatility_clustering.png")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     NIFTY 500 : COMPREHENSIVE EDA (2015 – 2025)            ║")
    print("║     Polars-Accelerated                                     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    sector_map, ticker_to_sector = load_sector_mappings()
    close_df, volume_df, high_df, low_df, open_df, turnover_df = load_all_daily()

    returns, illiquid_list = phase1_integrity(close_df, volume_df)
    phase2_distribution(close_df, returns, sector_map, ticker_to_sector)
    phase3_timeseries(close_df, returns)
    phase4_correlation(close_df, returns, sector_map, ticker_to_sector)
    liquid_stocks = phase5_liquidity(close_df, volume_df, turnover_df, high_df, low_df)
    phase6_regimes(close_df, returns, liquid_stocks)
    phase7_tail_risk(close_df, returns)

    print("\n" + "="*70)
    print("  EDA COMPLETE")
    print("="*70)
    n_plots = len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])
    print(f"  All {n_plots} plots saved to: {PLOT_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
