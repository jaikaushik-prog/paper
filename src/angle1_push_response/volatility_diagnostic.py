"""
Minimal Λ-Volatility Diagnostic Engine (Phase 1)
=================================================

Computes:
- Volatility Pressure Field ρ_i(t)
- Feedback Dominance Index (FDI)
- Diagnostic plots for NIFTY 500

This is a market regime & risk diagnostic, NOT a trading strategy.

Author: Auto-generated
Date: 2026-02-09
"""

import os
import glob
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DATA_DIR = "raw_Data"
OUTPUT_DIR = "plots"
FDI_OUTPUT_FILE = "fdi_output.csv"

# Trading hours (already filtered in data, but we remove first 5 mins)
FIRST_BAR_TO_REMOVE = "09:15"  # Remove 09:15 bar (first 5 mins)

# Estimator parameters
ADV_WINDOW = 20        # Rolling window for Average Daily Volume
EWMA_HALFLIFE = 5      # Half-life for FDI smoothing (days)
RHO_WINSORIZE = 6      # Winsorize ρ at ±6

# Regime thresholds (z-scores from rolling mean)
# Positive z-score = more destabilizing than average
# Negative z-score = more stabilizing than average
FDI_DESTAB_ZSCORE = 1.5   # FDI z-score > +1.5 indicates destabilizing regime
FDI_STAB_ZSCORE = -1.5    # FDI z-score < -1.5 indicates stabilizing regime
FDI_ROLLING_WINDOW = 252  # 1 year rolling window for demeaning

# Shock Intensity Index (SII) threshold
# SII measures magnitude of cross-sectional stress
SII_HIGH_THRESHOLD = 1.5  # SII z-score > 1.5 indicates high shock intensity

# Known stress periods to highlight
STRESS_PERIODS = [
    ("2020-03-01", "2020-04-15", "COVID-19 Crash"),
    ("2022-06-01", "2022-07-15", "Jun 2022 Selloff"),
]


# =============================================================================
# STEP 1: DATA LOADING & SANITY CHECKS
# =============================================================================

def load_data(data_dir: str, max_stocks: Optional[int] = None) -> pd.DataFrame:
    """
    Load all stocks into a unified panel DataFrame.
    
    Financial meaning: Creates a clean, aligned dataset where each stock's
    intraday bars can be compared cross-sectionally.
    
    Args:
        data_dir: Directory containing individual stock CSV files
        max_stocks: If set, limit the number of stocks loaded (for testing)
    
    Returns:
        Panel DataFrame with MultiIndex [timestamp, stock_id]
        Columns: [open, high, low, close, volume]
    """
    print("📦 Loading data from individual CSV files...")
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if max_stocks:
        csv_files = csv_files[:max_stocks]
    
    print(f"   Found {len(csv_files)} stock files")
    
    all_data = []
    
    for i, filepath in enumerate(csv_files):
        stock_id = os.path.basename(filepath).replace(".csv", "")
        
        try:
            df = pd.read_csv(filepath, parse_dates=['date'])
            df['stock_id'] = stock_id
            
            # Remove timezone info for easier handling
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            
            all_data.append(df)
            
        except Exception as e:
            print(f"   ⚠️ Error loading {stock_id}: {e}")
            continue
        
        if (i + 1) % 100 == 0:
            print(f"   Loaded {i + 1}/{len(csv_files)} stocks...")
    
    print(f"   Concatenating {len(all_data)} stocks...")
    panel = pd.concat(all_data, ignore_index=True)
    
    # Remove first 5 minutes of each day (09:15 bar)
    # Financial meaning: Opening auction period has non-representative volatility
    panel['time'] = panel['date'].dt.time
    first_bar = pd.to_datetime(FIRST_BAR_TO_REMOVE).time()
    pre_filter = len(panel)
    panel = panel[panel['time'] != first_bar]
    post_filter = len(panel)
    print(f"   Removed {pre_filter - post_filter:,} opening bars (09:15)")
    
    # Extract date for grouping
    panel['trading_date'] = panel['date'].dt.date
    
    # Set MultiIndex
    panel = panel.set_index(['date', 'stock_id'])
    panel = panel.sort_index()
    
    # Sanity checks
    print(f"\n✅ Panel loaded: {len(panel):,} rows")
    print(f"   Stocks: {panel.index.get_level_values('stock_id').nunique()}")
    print(f"   Date range: {panel.index.get_level_values('date').min()} to {panel.index.get_level_values('date').max()}")
    
    # Check for missing critical columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        null_count = panel[col].isnull().sum()
        if null_count > 0:
            print(f"   ⚠️ {col} has {null_count:,} missing values")
    
    return panel


# =============================================================================
# STEP 2: INTRADAY REALIZED VOLATILITY (RV)
# =============================================================================

def compute_realized_vol(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily realized volatility from 5-min log returns.
    
    Financial meaning: Realized volatility is a non-parametric estimator of
    true volatility, computed by summing squared intraday returns. Unlike
    GARCH or implied vol, it uses actual price movements.
    
    Formula: RV_i(t) = sqrt(Σ_k r²_{i,t,k})
    
    Args:
        panel: MultiIndex DataFrame with intraday OHLCV data
    
    Returns:
        DataFrame: realized_vol[date, stock_id]
    """
    print("\n📈 Computing intraday realized volatility...")
    
    # Reset index for easier grouping
    df = panel.reset_index()
    
    # Compute 5-min log returns per stock
    # Financial meaning: Log returns are additive and approximately symmetric
    df = df.sort_values(['stock_id', 'date'])
    df['log_return'] = df.groupby('stock_id')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    # Remove first return of each day (overnight gap)
    # Financial meaning: Overnight returns are structurally different from intraday
    df['is_first_bar'] = df.groupby(['stock_id', 'trading_date']).cumcount() == 0
    df.loc[df['is_first_bar'], 'log_return'] = np.nan
    
    # Compute daily realized variance: sum of squared returns
    realized_var = df.groupby(['trading_date', 'stock_id'])['log_return'].apply(
        lambda x: (x ** 2).sum()
    ).reset_index()
    realized_var.columns = ['date', 'stock_id', 'realized_var']
    
    # Take square root to get realized volatility
    realized_var['realized_vol'] = np.sqrt(realized_var['realized_var'])
    
    # Pivot to wide format
    rv_wide = realized_var.pivot(index='date', columns='stock_id', values='realized_vol')
    
    print(f"   ✅ Daily RV computed for {rv_wide.shape[1]} stocks over {rv_wide.shape[0]} days")
    print(f"   Mean RV: {rv_wide.mean().mean():.4f}")
    
    return rv_wide


# =============================================================================
# STEP 3: LIQUIDITY NORMALIZATION (INDIA-SPECIFIC)
# =============================================================================

def compute_liquidity_adjustment(panel: pd.DataFrame, rv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute liquidity-adjusted volatility using Amihud impact proxy.
    
    Financial meaning: Raw volatility is confounded by liquidity. Low-liquidity
    stocks appear more volatile due to price impact. We normalize to get
    "true" volatility independent of market depth.
    
    Amihud illiquidity: impact_i(t) = |r_i(t)| / volume_i(t)
    This measures how much price moves per unit of volume traded.
    
    Liquidity-adjusted vol: σ̃_i(t) = σ_i(t) / (√ADV_i(t) · impact_i(t))
    
    Args:
        panel: MultiIndex DataFrame with intraday OHLCV data
        rv: Daily realized volatility DataFrame
    
    Returns:
        DataFrame: liq_adjusted_vol[date, stock_id]
    """
    print("\n💧 Computing liquidity-adjusted volatility...")
    
    df = panel.reset_index()
    
    # Compute daily metrics per stock
    daily = df.groupby(['trading_date', 'stock_id']).agg({
        'volume': 'sum',
        'close': ['first', 'last']
    }).reset_index()
    daily.columns = ['date', 'stock_id', 'daily_volume', 'open_price', 'close_price']
    
    # Daily return (for Amihud)
    daily['daily_return'] = np.log(daily['close_price'] / daily['open_price'])
    
    # Pivot to wide format
    volume_wide = daily.pivot(index='date', columns='stock_id', values='daily_volume')
    return_wide = daily.pivot(index='date', columns='stock_id', values='daily_return')
    
    # Align all DataFrames
    common_dates = rv.index.intersection(volume_wide.index).intersection(return_wide.index)
    common_stocks = rv.columns.intersection(volume_wide.columns).intersection(return_wide.columns)
    
    rv = rv.loc[common_dates, common_stocks]
    volume_wide = volume_wide.loc[common_dates, common_stocks]
    return_wide = return_wide.loc[common_dates, common_stocks]
    
    # 1. Rolling 20-day Average Daily Volume (ADV)
    # Financial meaning: ADV is a proxy for market depth and trading activity
    adv = volume_wide.rolling(window=ADV_WINDOW, min_periods=10).mean()
    
    # 2. Amihud impact proxy: |daily_return| / daily_volume
    # Financial meaning: Higher ratio = more illiquid (price moves more per volume)
    # Add small epsilon to avoid division by zero
    amihud = np.abs(return_wide) / (volume_wide + 1e-10)
    
    # Smooth Amihud with 20-day rolling mean to reduce noise
    amihud_smooth = amihud.rolling(window=ADV_WINDOW, min_periods=10).mean()
    
    # 3. Liquidity-adjusted volatility
    # σ̃_i(t) = σ_i(t) / (√ADV_i(t) · impact_i(t))
    # Financial meaning: Normalize vol by liquidity factors to compare apples-to-apples
    sqrt_adv = np.sqrt(adv + 1e-10)
    liq_adjustment = sqrt_adv * amihud_smooth
    
    # Avoid division by zero
    liq_adjustment = liq_adjustment.replace(0, np.nan)
    liq_adjusted_vol = rv / (liq_adjustment + 1e-10)
    
    # Handle infinities and extreme values
    liq_adjusted_vol = liq_adjusted_vol.replace([np.inf, -np.inf], np.nan)
    
    print(f"   ✅ Liquidity-adjusted vol computed")
    print(f"   Shape: {liq_adjusted_vol.shape}")
    print(f"   Non-null fraction: {liq_adjusted_vol.notna().mean().mean():.2%}")
    
    return liq_adjusted_vol


# =============================================================================
# STEP 4: VOLATILITY PRESSURE FIELD (ρ)
# =============================================================================

def compute_rho(liq_adjusted_vol: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility pressure field using cross-sectional normalization.
    
    Financial meaning: ρ measures how "stressed" a stock is relative to the
    cross-section. High ρ = stock volatility is elevated vs peers. Low ρ = calm.
    Using median/MAD instead of mean/std makes it robust to fat tails.
    
    Formula: ρ_i(t) = (σ̃_i(t) - median_j) / MAD_j
    
    Winsorizing at ±6 prevents extreme outliers from dominating.
    
    Args:
        liq_adjusted_vol: Liquidity-adjusted volatility DataFrame
    
    Returns:
        DataFrame: rho[date, stock_id]
    """
    print("\n🌡️ Computing volatility pressure field (ρ)...")
    
    # Cross-sectional median (compute for each date across all stocks)
    cs_median = liq_adjusted_vol.median(axis=1)
    
    # Cross-sectional MAD (Median Absolute Deviation)
    # MAD = median(|x - median(x)|)
    # Financial meaning: Robust measure of dispersion, less sensitive to outliers
    def compute_mad(row):
        """Compute MAD for a row (one date, all stocks)."""
        valid = row.dropna()
        if len(valid) < 2:
            return np.nan
        return np.median(np.abs(valid - np.median(valid)))
    
    cs_mad = liq_adjusted_vol.apply(compute_mad, axis=1)
    
    # Compute ρ: cross-sectional z-score using median/MAD
    rho = liq_adjusted_vol.subtract(cs_median, axis=0).divide(cs_mad, axis=0)
    
    # Winsorize at ±6
    # Financial meaning: Extreme outliers are clipped to prevent them from
    # overwhelming the feedback correlation calculation
    rho = rho.clip(lower=-RHO_WINSORIZE, upper=RHO_WINSORIZE)
    
    print(f"   ✅ ρ computed and winsorized at ±{RHO_WINSORIZE}")
    print(f"   Shape: {rho.shape}")
    print(f"   Mean ρ: {rho.mean().mean():.4f} (should be ~0)")
    print(f"   Std ρ: {rho.std().mean():.4f}")
    
    return rho


# =============================================================================
# STEP 4B: SHOCK INTENSITY INDEX (SII)
# =============================================================================

def compute_sii(rho: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Shock Intensity Index (SII).
    
    Financial meaning: SII measures the MAGNITUDE of cross-sectional volatility
    stress, regardless of feedback structure. High SII = market is stressed.
    
    SII(t) = mean_i |ρ_i(t)|
    
    Combined with FDI, this gives a 2x2 regime classification:
    - SII high + FDI > 0: Reflexive crash (bad - self-reinforcing stress)
    - SII high + FDI < 0: Absorbed shock (stress being absorbed, e.g., COVID)
    - SII low + FDI > 0: Hidden instability (calm surface, fragile underneath)
    - SII low + FDI < 0: Healthy market (low stress, mean-reverting)
    
    Args:
        rho: Volatility pressure field DataFrame
    
    Returns:
        DataFrame with SII values and z-scores
    """
    print("\n⚡ Computing Shock Intensity Index (SII)...")
    
    # SII = cross-sectional mean of |ρ|
    sii_raw = rho.abs().mean(axis=1)
    
    sii_df = pd.DataFrame({'SII_raw': sii_raw})
    sii_df.index = pd.to_datetime(sii_df.index)
    
    # Smooth with EWMA
    sii_df['SII'] = sii_df['SII_raw'].ewm(halflife=EWMA_HALFLIFE, min_periods=3).mean()
    
    # Compute z-scores for regime detection
    sii_df['SII_rolling_mean'] = sii_df['SII'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).mean()
    sii_df['SII_rolling_std'] = sii_df['SII'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).std()
    sii_df['SII_zscore'] = (sii_df['SII'] - sii_df['SII_rolling_mean']) / sii_df['SII_rolling_std']
    
    print(f"   ✅ SII computed for {len(sii_df)} dates")
    print(f"   Mean SII: {sii_df['SII'].mean():.4f}")
    print(f"   Std SII: {sii_df['SII'].std():.4f}")
    print(f"   Z-score range: [{sii_df['SII_zscore'].min():.2f}, {sii_df['SII_zscore'].max():.2f}]")
    
    return sii_df


# =============================================================================
# STEP 5: FEEDBACK ESTIMATION (CORE)
# =============================================================================

def compute_fdi(rho: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Feedback Dominance Index (FDI).
    
    Financial meaning: FDI measures whether volatility shocks are self-reinforcing
    (positive feedback → destabilizing) or mean-reverting (negative feedback → stabilizing).
    
    - FDI > 0: High-vol stocks tend to get more volatile (momentum/panic regime)
    - FDI < 0: High-vol stocks tend to revert (stabilizing/calm regime)
    
    Method:
    1. Compute Δρ_i(t+1) = ρ_i(t+1) - ρ_i(t)  [change in pressure]
    2. G+(t) = corr_i(ρ_i(t), Δρ_i(t+1))     [positive feedback strength]
    3. G-(t) = corr_i(ρ_i(t), -Δρ_i(t+1))    [negative feedback strength]
    4. FDI(t) = G+(t) - G-(t) = 2 * G+(t)     [net feedback dominance]
    5. Smooth with EWMA (half-life = 5 days)
    
    Args:
        rho: Volatility pressure field DataFrame
    
    Returns:
        DataFrame with columns: date, FDI_raw, FDI (smoothed)
    """
    print("\n🔄 Computing Feedback Dominance Index (FDI)...")
    
    # Compute Δρ(t+1) = ρ(t+1) - ρ(t)
    # This measures the change in volatility pressure from t to t+1
    delta_rho = rho.diff()
    
    # For each date t, compute cross-sectional correlation between ρ(t) and Δρ(t+1)
    # We need to align: at date t, use ρ(t) and Δρ(t+1)
    rho_lagged = rho.shift(1)  # ρ at t-1
    
    # Now delta_rho at t = ρ(t) - ρ(t-1)
    # So we correlate rho_lagged (ρ at t-1) with delta_rho (Δρ from t-1 to t)
    
    fdi_raw = []
    dates = rho.index[1:]  # Skip first date (no lag available)
    
    for i, date in enumerate(dates):
        if i == 0:
            continue  # Need at least 2 rows
        
        # Get ρ at previous date
        rho_t = rho_lagged.loc[date]
        # Get Δρ at current date (change from t-1 to t)
        delta_rho_t = delta_rho.loc[date]
        
        # Combine and drop NaN
        combined = pd.DataFrame({'rho': rho_t, 'delta_rho': delta_rho_t}).dropna()
        
        if len(combined) < 30:  # Need sufficient stocks for meaningful correlation
            fdi_raw.append({'date': date, 'FDI_raw': np.nan})
            continue
        
        # G+ = corr(ρ, Δρ) - positive feedback correlation
        # Note: G-(t) = corr(ρ, -Δρ) = -corr(ρ, Δρ) = -G+
        # So FDI = G+ - G- = G+ - (-G+) = 2*G+
        g_plus = combined['rho'].corr(combined['delta_rho'])
        fdi_value = 2 * g_plus  # This is G+ - G-
        
        fdi_raw.append({'date': date, 'FDI_raw': fdi_value})
    
    fdi_df = pd.DataFrame(fdi_raw)
    fdi_df['date'] = pd.to_datetime(fdi_df['date'])
    fdi_df = fdi_df.set_index('date')
    
    # EWMA smoothing with half-life of 5 days
    # Financial meaning: Reduces noise while preserving regime shifts
    # Half-life = 5 means influence of past observation halves every 5 days
    fdi_df['FDI'] = fdi_df['FDI_raw'].ewm(halflife=EWMA_HALFLIFE, min_periods=3).mean()
    
    # Demean FDI using rolling mean/std for relative regime detection
    # Financial meaning: Since volatility is naturally mean-reverting, FDI tends to
    # be negative on average. We compute z-scores to identify RELATIVE regime shifts.
    fdi_df['FDI_rolling_mean'] = fdi_df['FDI'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).mean()
    fdi_df['FDI_rolling_std'] = fdi_df['FDI'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).std()
    fdi_df['FDI_zscore'] = (fdi_df['FDI'] - fdi_df['FDI_rolling_mean']) / fdi_df['FDI_rolling_std']
    
    print(f"   ✅ FDI computed for {len(fdi_df)} dates")
    print(f"   Mean FDI: {fdi_df['FDI'].mean():.4f}")
    print(f"   Std FDI: {fdi_df['FDI'].std():.4f}")
    print(f"   Range: [{fdi_df['FDI'].min():.4f}, {fdi_df['FDI'].max():.4f}]")
    print(f"   Z-score range: [{fdi_df['FDI_zscore'].min():.2f}, {fdi_df['FDI_zscore'].max():.2f}]")
    
    return fdi_df


def combine_fdi_sii(fdi_df: pd.DataFrame, sii_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine FDI and SII into unified regime DataFrame with 2x2 classification.
    
    The 2x2 Grid:
    +-------------------+------------------+-------------------+
    |                   | SII Low          | SII High          |
    +-------------------+------------------+-------------------+
    | FDI > 0           | Hidden           | Reflexive         |
    | (destabilizing)   | Instability      | Crash             |
    +-------------------+------------------+-------------------+
    | FDI < 0           | Healthy          | Absorbed          |
    | (stabilizing)     | Market           | Shock             |
    +-------------------+------------------+-------------------+
    
    Args:
        fdi_df: DataFrame with FDI and FDI_zscore
        sii_df: DataFrame with SII and SII_zscore
    
    Returns:
        Combined DataFrame with regime classification
    """
    print("\n🎯 Combining FDI and SII for regime classification...")
    
    # Merge on date index
    combined = fdi_df[['FDI', 'FDI_zscore']].join(sii_df[['SII', 'SII_zscore']], how='inner')
    
    # 2x2 Regime Classification
    def classify_regime(row):
        fdi_z = row['FDI_zscore']
        sii_z = row['SII_zscore']
        
        if pd.isna(fdi_z) or pd.isna(sii_z):
            return 'unknown'
        
        sii_high = sii_z > SII_HIGH_THRESHOLD
        fdi_destab = fdi_z > 0  # Above rolling mean = more destabilizing than average
        
        if sii_high and fdi_destab:
            return 'reflexive_crash'      # 🔴 Worst: stress is self-reinforcing
        elif sii_high and not fdi_destab:
            return 'absorbed_shock'       # 🟡 Stress high but being absorbed
        elif not sii_high and fdi_destab:
            return 'hidden_instability'   # 🟠 Calm surface, fragile underneath
        else:
            return 'healthy'              # 🟢 Low stress, mean-reverting
    
    combined['regime_2x2'] = combined.apply(classify_regime, axis=1)
    
    # Print regime statistics
    regime_counts = combined['regime_2x2'].value_counts()
    print("\n   2x2 Regime Distribution:")
    emoji_map = {'healthy': '🟢', 'absorbed_shock': '🟡', 'hidden_instability': '🟠', 'reflexive_crash': '🔴', 'unknown': '⚪'}
    for regime in ['healthy', 'absorbed_shock', 'hidden_instability', 'reflexive_crash', 'unknown']:
        if regime in regime_counts:
            count = regime_counts[regime]
            pct = count / len(combined) * 100
            print(f"      {emoji_map.get(regime, '')} {regime}: {count} days ({pct:.1f}%)")
    
    return combined


# =============================================================================
# STEP 5C: SECTORAL FDI COMPUTATION
# =============================================================================

def load_sector_mappings(filepath: str = "sector_mappings.json") -> dict:
    """Load sector mappings from JSON file."""
    import json
    with open(filepath, 'r') as f:
        mappings = json.load(f)
    # Remove comment key if present
    mappings.pop('comment', None)
    return mappings


def compute_sectoral_fdi(rho: pd.DataFrame, sector_mappings: dict) -> pd.DataFrame:
    """
    Compute FDI separately for each sector.
    
    Financial meaning: Different sectors have different feedback dynamics.
    Banks often destabilize first during crises, IT tends to stabilize stress,
    Midcaps lag and amplify. This reveals sector-level alpha intuition.
    
    Args:
        rho: Volatility pressure field DataFrame
        sector_mappings: Dict mapping sector names to list of stock tickers
    
    Returns:
        DataFrame with FDI per sector over time
    """
    print("\n🏦 Computing Sectoral FDI...")
    
    available_stocks = set(rho.columns)
    sector_fdi_results = {}
    
    for sector, stocks in sector_mappings.items():
        # Find stocks that exist in our data
        sector_stocks = [s for s in stocks if s in available_stocks]
        
        if len(sector_stocks) < 10:
            print(f"   ⚠️ {sector}: Only {len(sector_stocks)} stocks, skipping (need ≥10)")
            continue
        
        # Subset rho for this sector
        rho_sector = rho[sector_stocks]
        
        # Compute FDI for this sector (same logic as main FDI)
        delta_rho = rho_sector.diff()
        rho_lagged = rho_sector.shift(1)
        
        fdi_raw = []
        dates = rho_sector.index[2:]  # Skip first 2 dates
        
        for date in dates:
            rho_t = rho_lagged.loc[date]
            delta_rho_t = delta_rho.loc[date]
            
            combined = pd.DataFrame({'rho': rho_t, 'delta_rho': delta_rho_t}).dropna()
            
            if len(combined) < 8:  # Need at least 8 stocks for sector correlation
                fdi_raw.append({'date': date, 'FDI_raw': np.nan})
                continue
            
            g_plus = combined['rho'].corr(combined['delta_rho'])
            fdi_value = 2 * g_plus
            fdi_raw.append({'date': date, 'FDI_raw': fdi_value})
        
        fdi_df = pd.DataFrame(fdi_raw)
        fdi_df['date'] = pd.to_datetime(fdi_df['date'])
        fdi_df = fdi_df.set_index('date')
        
        # EWMA smoothing
        fdi_df['FDI'] = fdi_df['FDI_raw'].ewm(halflife=EWMA_HALFLIFE, min_periods=3).mean()
        
        # Z-score
        fdi_df['FDI_rolling_mean'] = fdi_df['FDI'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).mean()
        fdi_df['FDI_rolling_std'] = fdi_df['FDI'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).std()
        fdi_df['FDI_zscore'] = (fdi_df['FDI'] - fdi_df['FDI_rolling_mean']) / fdi_df['FDI_rolling_std']
        
        sector_fdi_results[sector] = fdi_df['FDI_zscore']
        
        print(f"   ✅ {sector}: {len(sector_stocks)} stocks, mean FDI={fdi_df['FDI'].mean():.3f}")
    
    # Combine all sectors into one DataFrame
    sectoral_fdi = pd.DataFrame(sector_fdi_results)
    sectoral_fdi.index = pd.to_datetime(sectoral_fdi.index)
    
    print(f"\n   📊 Computed FDI for {len(sectoral_fdi.columns)} sectors")
    
    return sectoral_fdi


def create_sectoral_diagnostics(sectoral_fdi: pd.DataFrame, output_dir: str):
    """
    Create diagnostic plots for sectoral FDI.
    
    Plots:
    1. Sectoral FDI heatmap over time
    2. Sector lead/lag analysis
    3. Stress period sector breakdown
    
    Args:
        sectoral_fdi: DataFrame with FDI z-score per sector
        output_dir: Directory to save plots
    """
    print("\n📊 Creating sectoral FDI diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color palette for sectors
    sector_colors = {
        'Banks': '#e74c3c',      # Red
        'NBFCs': '#c0392b',      # Dark red
        'IT': '#3498db',         # Blue
        'Metals': '#7f8c8d',     # Gray
        'Pharma': '#2ecc71',     # Green
        'Auto': '#f39c12',       # Orange
        'FMCG': '#9b59b6',       # Purple
        'Infrastructure': '#1abc9c',  # Teal
        'Power_Utilities': '#e67e22'  # Dark orange
    }
    
    # --- PLOT 1: Sectoral FDI Time Series ---
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for sector in sectoral_fdi.columns:
        color = sector_colors.get(sector, '#333333')
        ax.plot(sectoral_fdi.index, sectoral_fdi[sector], 
                label=sector, color=color, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axhline(y=FDI_DESTAB_ZSCORE, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=FDI_STAB_ZSCORE, color='g', linestyle='--', linewidth=1, alpha=0.5)
    
    # Highlight stress periods
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= sectoral_fdi.index.min() and start_dt <= sectoral_fdi.index.max():
            ax.axvspan(start_dt, end_dt, alpha=0.2, color='red', label=label)
    
    ax.set_ylabel('FDI Z-score', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Sectoral FDI Z-scores - NIFTY 500', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sectoral_fdi_timeseries.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # --- PLOT 2: Stress Period Sector Analysis ---
    fig, axes = plt.subplots(1, len(STRESS_PERIODS), figsize=(6*len(STRESS_PERIODS), 6))
    if len(STRESS_PERIODS) == 1:
        axes = [axes]
    
    for idx, (start, end, label) in enumerate(STRESS_PERIODS):
        ax = axes[idx]
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # Get data during stress
        mask = (sectoral_fdi.index >= start_dt) & (sectoral_fdi.index <= end_dt)
        stress_data = sectoral_fdi.loc[mask]
        
        if len(stress_data) > 0:
            # Mean FDI during stress per sector
            mean_fdi = stress_data.mean().sort_values(ascending=False)
            colors = [sector_colors.get(s, '#333333') for s in mean_fdi.index]
            
            bars = ax.barh(range(len(mean_fdi)), mean_fdi.values, color=colors)
            ax.set_yticks(range(len(mean_fdi)))
            ax.set_yticklabels(mean_fdi.index)
            ax.axvline(x=0, color='k', linewidth=0.5)
            ax.axvline(x=FDI_DESTAB_ZSCORE, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Mean FDI Z-score')
            ax.set_title(f'{label}', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sectoral_fdi_stress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # --- PLOT 3: Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = sectoral_fdi.dropna().corr()
    
    im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    
    # Add labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title('Sector FDI Correlation Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Correlation')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sectoral_fdi_correlation.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


# =============================================================================
# STEP 5D: TIME-OF-DAY CONDITIONING
# =============================================================================

# Trading sessions (India market hours: 09:15-15:30)
TRADING_SESSIONS = {
    'morning': ('09:20', '11:00'),    # Opening volatility, gap absorption
    'midday': ('11:00', '13:30'),     # Lower volatility, lunch period
    'afternoon': ('13:30', '15:30'),  # Closing auction, institutional activity
}


def compute_intraday_fdi(panel: pd.DataFrame, rho: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FDI separately for each trading session.
    
    Financial meaning: Different times of day have different feedback structures.
    - Morning: Overnight gaps being absorbed, high initial volatility
    - Midday: Quieter period, lower information flow
    - Afternoon: Institutional positioning, closing auction effects
    
    Args:
        panel: Raw intraday data with timestamps
        rho: Daily volatility pressure field (for reference)
    
    Returns:
        DataFrame with session-specific FDI
    """
    print("\n⏰ Computing Time-of-Day FDI...")
    
    df = panel.reset_index()
    df['time'] = df['date'].dt.time
    df['trading_date'] = df['date'].dt.date
    
    session_fdi_results = {}
    
    for session_name, (start_time, end_time) in TRADING_SESSIONS.items():
        start = pd.to_datetime(start_time).time()
        end = pd.to_datetime(end_time).time()
        
        # Filter data for this session
        session_mask = (df['time'] >= start) & (df['time'] < end)
        session_df = df[session_mask].copy()
        
        if len(session_df) == 0:
            print(f"   ⚠️ {session_name}: No data")
            continue
        
        # Compute session-specific realized variance
        session_df = session_df.sort_values(['stock_id', 'date'])
        session_df['log_return'] = session_df.groupby('stock_id')['close'].transform(
            lambda x: np.log(x / x.shift(1))
        )
        
        # Session realized variance per stock per day
        session_rv = session_df.groupby(['trading_date', 'stock_id'])['log_return'].apply(
            lambda x: (x ** 2).sum()
        ).reset_index()
        session_rv.columns = ['date', 'stock_id', 'realized_var']
        session_rv['realized_vol'] = np.sqrt(session_rv['realized_var'])
        
        # Pivot to wide format
        rv_wide = session_rv.pivot(index='date', columns='stock_id', values='realized_vol')
        
        # Cross-sectional normalization (simplified ρ)
        cs_median = rv_wide.median(axis=1)
        cs_mad = rv_wide.apply(lambda row: np.median(np.abs(row.dropna() - np.median(row.dropna()))), axis=1)
        rho_session = rv_wide.subtract(cs_median, axis=0).divide(cs_mad, axis=0)
        rho_session = rho_session.clip(lower=-RHO_WINSORIZE, upper=RHO_WINSORIZE)
        
        # Compute FDI for this session
        delta_rho = rho_session.diff()
        rho_lagged = rho_session.shift(1)
        
        fdi_raw = []
        dates = rho_session.index[2:]
        
        for date in dates:
            rho_t = rho_lagged.loc[date]
            delta_rho_t = delta_rho.loc[date]
            combined = pd.DataFrame({'rho': rho_t, 'delta_rho': delta_rho_t}).dropna()
            
            if len(combined) < 30:
                fdi_raw.append({'date': date, 'FDI_raw': np.nan})
                continue
            
            g_plus = combined['rho'].corr(combined['delta_rho'])
            fdi_raw.append({'date': date, 'FDI_raw': 2 * g_plus})
        
        fdi_df = pd.DataFrame(fdi_raw)
        fdi_df['date'] = pd.to_datetime(fdi_df['date'])
        fdi_df = fdi_df.set_index('date')
        
        # EWMA smoothing and z-score
        fdi_df['FDI'] = fdi_df['FDI_raw'].ewm(halflife=EWMA_HALFLIFE, min_periods=3).mean()
        fdi_df['FDI_rolling_mean'] = fdi_df['FDI'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).mean()
        fdi_df['FDI_rolling_std'] = fdi_df['FDI'].rolling(window=FDI_ROLLING_WINDOW, min_periods=60).std()
        fdi_df['FDI_zscore'] = (fdi_df['FDI'] - fdi_df['FDI_rolling_mean']) / fdi_df['FDI_rolling_std']
        
        session_fdi_results[session_name] = fdi_df['FDI_zscore']
        print(f"   ✅ {session_name}: {len(fdi_df)} days, mean FDI={fdi_df['FDI'].mean():.3f}")
    
    intraday_fdi = pd.DataFrame(session_fdi_results)
    intraday_fdi.index = pd.to_datetime(intraday_fdi.index)
    
    return intraday_fdi


def create_intraday_diagnostics(intraday_fdi: pd.DataFrame, output_dir: str):
    """Create diagnostic plots for intraday FDI."""
    print("\n📊 Creating intraday FDI diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    session_colors = {'morning': '#e74c3c', 'midday': '#3498db', 'afternoon': '#2ecc71'}
    
    # Time series
    ax1 = axes[0]
    for session in intraday_fdi.columns:
        ax1.plot(intraday_fdi.index, intraday_fdi[session], 
                label=session.title(), color=session_colors.get(session, 'gray'), linewidth=1.2)
    
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=FDI_DESTAB_ZSCORE, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('FDI Z-score', fontsize=12)
    ax1.set_title('Intraday FDI: Morning vs Midday vs Afternoon', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= intraday_fdi.index.min() and start_dt <= intraday_fdi.index.max():
            ax1.axvspan(start_dt, end_dt, alpha=0.2, color='red')
    
    # Session comparison (rolling mean difference)
    ax2 = axes[1]
    if 'morning' in intraday_fdi.columns and 'afternoon' in intraday_fdi.columns:
        diff = (intraday_fdi['morning'] - intraday_fdi['afternoon']).rolling(20).mean()
        ax2.fill_between(diff.index, 0, diff.values, where=diff>0, color='red', alpha=0.5, label='Morning > Afternoon')
        ax2.fill_between(diff.index, 0, diff.values, where=diff<0, color='green', alpha=0.5, label='Afternoon > Morning')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Morning - Afternoon FDI', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'intraday_fdi.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


# =============================================================================
# STEP 5E: STRATEGY INTEGRATION
# =============================================================================

def compute_strategy_signals(combined: pd.DataFrame, sectoral_fdi: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate strategy signals based on FDI regime.
    
    Rules:
    1. FDI z-score > +1.5 → Reduce leverage to 50%
    2. FDI z-score > +2.0 → Reduce leverage to 25%
    3. Reflexive crash regime → Go to 0% leverage
    4. SII z-score > +2.0 → Hedge with options
    
    Args:
        combined: DataFrame with FDI, SII, and regime
        sectoral_fdi: Optional sectoral FDI for sector-specific signals
    
    Returns:
        DataFrame with strategy signals
    """
    print("\n📈 Generating strategy signals...")
    
    signals = combined[['FDI', 'FDI_zscore', 'SII', 'SII_zscore', 'regime_2x2']].copy()
    
    # Leverage adjustment based on FDI z-score
    def compute_leverage(row):
        fdi_z = row['FDI_zscore']
        regime = row['regime_2x2']
        
        if pd.isna(fdi_z):
            return 1.0  # Full leverage by default
        
        if regime == 'reflexive_crash':
            return 0.0  # Exit completely
        elif fdi_z > 2.0:
            return 0.25  # 25% leverage
        elif fdi_z > FDI_DESTAB_ZSCORE:
            return 0.50  # 50% leverage
        elif fdi_z < FDI_STAB_ZSCORE:
            return 1.5   # Can increase leverage in stabilizing regime
        else:
            return 1.0   # Normal leverage
    
    signals['leverage_mult'] = signals.apply(compute_leverage, axis=1)
    
    # Hedge signal (when to buy protection)
    signals['hedge_signal'] = (signals['SII_zscore'] > 2.0) | (signals['regime_2x2'] == 'reflexive_crash')
    
    # Early warning (hidden instability)
    signals['early_warning'] = signals['regime_2x2'] == 'hidden_instability'
    
    # Print signal statistics
    print(f"   Leverage Distribution:")
    print(f"      0% (exit): {(signals['leverage_mult'] == 0).sum()} days")
    print(f"      25%: {(signals['leverage_mult'] == 0.25).sum()} days")
    print(f"      50%: {(signals['leverage_mult'] == 0.5).sum()} days")
    print(f"      100%: {(signals['leverage_mult'] == 1.0).sum()} days")
    print(f"      150%: {(signals['leverage_mult'] == 1.5).sum()} days")
    print(f"   Hedge days: {signals['hedge_signal'].sum()}")
    print(f"   Early warning days: {signals['early_warning'].sum()}")
    
    return signals


def create_strategy_diagnostics(signals: pd.DataFrame, output_dir: str):
    """Create diagnostic plots for strategy signals."""
    print("\n📊 Creating strategy diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # FDI Z-score with thresholds
    ax1 = axes[0]
    ax1.plot(signals.index, signals['FDI_zscore'], 'b-', linewidth=1, label='FDI Z-score')
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axhline(y=FDI_DESTAB_ZSCORE, color='orange', linestyle='--', label='50% leverage')
    ax1.axhline(y=2.0, color='red', linestyle='--', label='25% leverage')
    ax1.fill_between(signals.index, FDI_DESTAB_ZSCORE, signals['FDI_zscore'], 
                     where=signals['FDI_zscore']>FDI_DESTAB_ZSCORE, 
                     color='orange', alpha=0.3)
    ax1.fill_between(signals.index, 2.0, signals['FDI_zscore'], 
                     where=signals['FDI_zscore']>2.0, 
                     color='red', alpha=0.3)
    ax1.set_ylabel('FDI Z-score', fontsize=12)
    ax1.set_title('Strategy Signals: Leverage Adjustment', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Leverage multiplier
    ax2 = axes[1]
    ax2.fill_between(signals.index, 0, signals['leverage_mult'], 
                     color='steelblue', alpha=0.7)
    ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax2.set_ylabel('Leverage Multiplier', fontsize=12)
    ax2.set_ylim(0, 1.6)
    ax2.grid(True, alpha=0.3)
    
    # Hedge and warning signals
    ax3 = axes[2]
    hedge_numeric = signals['hedge_signal'].astype(int) * 2
    warning_numeric = signals['early_warning'].astype(int)
    ax3.fill_between(signals.index, 0, hedge_numeric, color='red', alpha=0.5, label='Hedge Signal')
    ax3.fill_between(signals.index, 0, warning_numeric, color='orange', alpha=0.5, label='Early Warning')
    ax3.set_ylabel('Signal', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['None', 'Warning', 'Hedge'])
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'strategy_signals.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


# =============================================================================
# PHASE 4A: REGIME TRANSITION DYNAMICS
# =============================================================================

def compute_regime_transitions(combined: pd.DataFrame) -> dict:
    """
    Compute regime transition probabilities and expected durations.
    
    This converts the framework from DESCRIPTIVE to ANTICIPATORY.
    Key questions answered:
    - P(Hidden → Reflexive): How dangerous is hidden instability?
    - P(Absorbed → Healthy): How fast does market recover?
    - Expected duration: How long do regimes persist?
    
    Returns:
        Dict with transition matrix, danger scores, and expected durations
    """
    print("\n🔄 Computing Regime Transition Dynamics...")
    
    regimes = combined['regime_2x2'].dropna()
    regime_names = ['healthy', 'absorbed_shock', 'hidden_instability', 'reflexive_crash']
    
    # Filter to valid regimes
    regimes = regimes[regimes.isin(regime_names)]
    
    # Compute transition matrix
    transitions = pd.crosstab(regimes, regimes.shift(-1), normalize='index')
    
    # Ensure all regimes are represented
    for r in regime_names:
        if r not in transitions.index:
            transitions.loc[r] = 0
        if r not in transitions.columns:
            transitions[r] = 0
    
    transitions = transitions.reindex(index=regime_names, columns=regime_names, fill_value=0)
    
    # Key transition probabilities
    p_hidden_to_reflexive = transitions.loc['hidden_instability', 'reflexive_crash'] if 'hidden_instability' in transitions.index else 0
    p_absorbed_to_healthy = transitions.loc['absorbed_shock', 'healthy'] if 'absorbed_shock' in transitions.index else 0
    p_reflexive_persistence = transitions.loc['reflexive_crash', 'reflexive_crash'] if 'reflexive_crash' in transitions.index else 0
    
    # Expected regime durations (geometric distribution: E[T] = 1 / (1 - p_stay))
    expected_durations = {}
    for regime in regime_names:
        p_stay = transitions.loc[regime, regime] if regime in transitions.index else 0
        if p_stay < 1:
            expected_durations[regime] = 1 / (1 - p_stay)
        else:
            expected_durations[regime] = float('inf')
    
    # Danger score: combines probability and persistence
    danger_score = p_hidden_to_reflexive * (1 + expected_durations.get('reflexive_crash', 1))
    
    print(f"\n   📊 Transition Matrix:")
    print(transitions.round(3).to_string())
    print(f"\n   ⚠️ KEY METRICS:")
    print(f"      P(Hidden → Reflexive): {p_hidden_to_reflexive:.1%}")
    print(f"      P(Absorbed → Healthy): {p_absorbed_to_healthy:.1%}")
    print(f"      P(Reflexive → Reflexive): {p_reflexive_persistence:.1%}")
    print(f"\n   ⏱️ Expected Durations (days):")
    for regime, duration in expected_durations.items():
        emoji = {'healthy': '🟢', 'absorbed_shock': '🟡', 'hidden_instability': '🟠', 'reflexive_crash': '🔴'}
        print(f"      {emoji.get(regime, '')} {regime}: {duration:.1f} days")
    print(f"\n   🎯 Danger Score: {danger_score:.3f}")
    
    return {
        'transition_matrix': transitions,
        'p_hidden_to_reflexive': p_hidden_to_reflexive,
        'p_absorbed_to_healthy': p_absorbed_to_healthy,
        'p_reflexive_persistence': p_reflexive_persistence,
        'expected_durations': expected_durations,
        'danger_score': danger_score
    }


def create_transition_diagnostics(transition_data: dict, output_dir: str):
    """Create diagnostic plots for regime transitions."""
    print("\n📊 Creating transition diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Transition matrix heatmap
    ax1 = axes[0]
    tm = transition_data['transition_matrix']
    im = ax1.imshow(tm.values, cmap='YlOrRd', vmin=0, vmax=1)
    
    ax1.set_xticks(range(len(tm.columns)))
    ax1.set_yticks(range(len(tm.index)))
    labels = [r.replace('_', '\n') for r in tm.index]
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    
    for i in range(len(tm.index)):
        for j in range(len(tm.columns)):
            val = tm.iloc[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax1.text(j, i, f'{val:.1%}', ha='center', va='center', color=color, fontsize=10)
    
    ax1.set_xlabel('To Regime', fontsize=12)
    ax1.set_ylabel('From Regime', fontsize=12)
    ax1.set_title('Regime Transition Probabilities', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Probability')
    
    # Expected durations bar chart
    ax2 = axes[1]
    durations = transition_data['expected_durations']
    colors = {'healthy': '#2ecc71', 'absorbed_shock': '#f1c40f', 
              'hidden_instability': '#e67e22', 'reflexive_crash': '#e74c3c'}
    
    bars = ax2.bar(range(len(durations)), list(durations.values()), 
                   color=[colors.get(r, 'gray') for r in durations.keys()])
    ax2.set_xticks(range(len(durations)))
    ax2.set_xticklabels([r.replace('_', '\n') for r in durations.keys()], fontsize=9)
    ax2.set_ylabel('Expected Duration (days)', fontsize=12)
    ax2.set_title('Expected Regime Durations', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, durations.values()):
        if val < float('inf'):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.1f}d', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'regime_transitions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


# =============================================================================
# PHASE 4B: SECTOR LEAD-LAG ANALYSIS
# =============================================================================

def compute_sector_lead_lag(sectoral_fdi: pd.DataFrame, combined: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
    """
    Compute cross-correlation of sector-FDI vs market-FDI at different lags.
    
    Identifies which sectors LEAD market regime changes.
    Positive lag = sector leads market.
    
    This is EARLY WARNING ALPHA.
    
    Args:
        sectoral_fdi: Per-sector FDI z-scores
        combined: Market-level FDI
        max_lag: Maximum lag to test (days)
    
    Returns:
        DataFrame with lead-lag correlations per sector
    """
    print("\n🔍 Computing Sector Lead-Lag Analysis...")
    
    market_fdi = combined['FDI_zscore'].dropna()
    
    lead_lag_results = {}
    
    for sector in sectoral_fdi.columns:
        sector_series = sectoral_fdi[sector].dropna()
        
        # Align dates
        common_dates = market_fdi.index.intersection(sector_series.index)
        if len(common_dates) < 100:
            continue
        
        market_aligned = market_fdi.loc[common_dates]
        sector_aligned = sector_series.loc[common_dates]
        
        # Compute cross-correlation at different lags
        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                # Sector leads market (sector at t predicts market at t+lag)
                corr = sector_aligned.iloc[:-lag].corr(market_aligned.iloc[lag:])
            elif lag < 0:
                # Market leads sector
                corr = market_aligned.iloc[:lag].corr(sector_aligned.iloc[-lag:])
            else:
                corr = sector_aligned.corr(market_aligned)
            correlations[lag] = corr
        
        lead_lag_results[sector] = correlations
    
    lead_lag_df = pd.DataFrame(lead_lag_results)
    
    # Find optimal lead for each sector
    optimal_leads = {}
    for sector in lead_lag_df.columns:
        sector_corrs = lead_lag_df[sector].dropna()
        if len(sector_corrs) > 0:
            # Only positive lags (sector leads)
            positive_lags = sector_corrs[sector_corrs.index > 0]
            if len(positive_lags) > 0:
                best_lag = positive_lags.idxmax()
                best_corr = positive_lags.max()
                optimal_leads[sector] = {'lag': best_lag, 'corr': best_corr}
    
    print(f"\n   📊 Sector Lead Analysis (positive lag = sector leads market):")
    sorted_sectors = sorted(optimal_leads.items(), key=lambda x: x[1]['corr'], reverse=True)
    for sector, data in sorted_sectors[:5]:
        print(f"      {sector}: {data['lag']} day lead, corr={data['corr']:.3f}")
    
    return lead_lag_df, optimal_leads


def create_lead_lag_diagnostics(lead_lag_df: pd.DataFrame, optimal_leads: dict, output_dir: str):
    """Create diagnostic plots for sector lead-lag analysis."""
    print("\n📊 Creating lead-lag diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cross-correlation plot
    ax1 = axes[0]
    sector_colors = {
        'Banks': '#e74c3c', 'NBFCs': '#c0392b', 'IT': '#3498db',
        'Metals': '#7f8c8d', 'Pharma': '#2ecc71', 'Auto': '#f39c12',
        'FMCG': '#9b59b6', 'Infrastructure': '#1abc9c', 'Power_Utilities': '#e67e22'
    }
    
    for sector in lead_lag_df.columns:
        color = sector_colors.get(sector, 'gray')
        ax1.plot(lead_lag_df.index, lead_lag_df[sector], 
                label=sector, color=color, linewidth=1.5)
    
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=1)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.fill_betweenx([-1, 1], 0, lead_lag_df.index.max(), alpha=0.1, color='green', label='Sector Leads')
    ax1.set_xlabel('Lag (days)', fontsize=12)
    ax1.set_ylabel('Cross-correlation with Market FDI', fontsize=12)
    ax1.set_title('Sector Lead-Lag Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-10, 10)
    
    # Optimal lead bar chart
    ax2 = axes[1]
    if optimal_leads:
        sectors = list(optimal_leads.keys())
        lags = [optimal_leads[s]['lag'] for s in sectors]
        corrs = [optimal_leads[s]['corr'] for s in sectors]
        
        # Sort by correlation
        sorted_idx = sorted(range(len(corrs)), key=lambda i: corrs[i], reverse=True)
        sectors = [sectors[i] for i in sorted_idx]
        lags = [lags[i] for i in sorted_idx]
        corrs = [corrs[i] for i in sorted_idx]
        
        colors = [sector_colors.get(s, 'gray') for s in sectors]
        bars = ax2.barh(range(len(sectors)), lags, color=colors)
        ax2.set_yticks(range(len(sectors)))
        ax2.set_yticklabels(sectors)
        ax2.set_xlabel('Lead Days (positive = sector leads)', fontsize=12)
        ax2.set_title('Optimal Sector Lead Time', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Annotate with correlation
        for i, (bar, corr) in enumerate(zip(bars, corrs)):
            ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                    f'ρ={corr:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sector_lead_lag.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


# =============================================================================
# PHASE 4C: PORTFOLIO STRESS ROUTING
# =============================================================================

def compute_stress_routing(sectoral_fdi: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-sector leverage adjustments (stress routing).
    
    Instead of: "Reduce risk when market bad"
    We do: "Reduce exposure to sectors with positive FDI"
    
    Benefits:
    - Stay invested
    - Rotate risk
    - Avoid blunt de-risking
    
    Returns:
        DataFrame with per-sector allocation weights over time
    """
    print("\n🎯 Computing Portfolio Stress Routing...")
    
    # Merge sectoral FDI with market regime
    allocations = sectoral_fdi.copy()
    
    # Per-sector leverage based on sector FDI
    def sector_weight(fdi_z):
        """Convert FDI z-score to allocation weight (0-1.5)"""
        if pd.isna(fdi_z):
            return 1.0
        elif fdi_z > 2.0:
            return 0.0  # Zero allocation
        elif fdi_z > 1.5:
            return 0.25
        elif fdi_z > 0.5:
            return 0.5
        elif fdi_z < -1.5:
            return 1.5  # Overweight stabilizing sectors
        elif fdi_z < -0.5:
            return 1.25
        else:
            return 1.0  # Neutral
    
    weights = allocations.applymap(sector_weight)
    
    # Normalize to sum to 1 (for portfolio allocation)
    weights_normalized = weights.div(weights.sum(axis=1), axis=0)
    
    # Equal weight baseline
    n_sectors = len(sectoral_fdi.columns)
    equal_weight = 1 / n_sectors
    
    # Active weight deviation from equal
    active_weights = weights_normalized - equal_weight
    
    # Print summary
    print(f"\n   📊 Stress Routing Summary:")
    print(f"      Sectors: {n_sectors}")
    print(f"      Equal weight: {equal_weight:.1%}")
    
    # Mean allocations
    mean_weights = weights_normalized.mean()
    print(f"\n   Mean Allocations (stress-adjusted):")
    for sector in mean_weights.sort_values(ascending=False).index[:5]:
        deviation = (mean_weights[sector] - equal_weight) * 100
        direction = "↑" if deviation > 0 else "↓"
        print(f"      {sector}: {mean_weights[sector]:.1%} ({direction}{abs(deviation):.1f}pp)")
    
    return {
        'raw_weights': weights,
        'normalized_weights': weights_normalized,
        'active_weights': active_weights
    }


def create_stress_routing_diagnostics(routing_data: dict, output_dir: str):
    """Create diagnostic plots for portfolio stress routing."""
    print("\n📊 Creating stress routing diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    weights = routing_data['normalized_weights']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    sector_colors = {
        'Banks': '#e74c3c', 'NBFCs': '#c0392b', 'IT': '#3498db',
        'Metals': '#7f8c8d', 'Pharma': '#2ecc71', 'Auto': '#f39c12',
        'FMCG': '#9b59b6', 'Infrastructure': '#1abc9c', 'Power_Utilities': '#e67e22'
    }
    
    # Stacked area chart of allocations
    ax1 = axes[0]
    weights_filled = weights.fillna(method='ffill').fillna(1/len(weights.columns))
    
    # Plot stacked area
    ax1.stackplot(weights_filled.index, weights_filled.T.values,
                  labels=weights_filled.columns,
                  colors=[sector_colors.get(s, 'gray') for s in weights_filled.columns],
                  alpha=0.8)
    
    ax1.set_ylabel('Portfolio Allocation', fontsize=12)
    ax1.set_title('Dynamic Sector Allocation (Stress Routing)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.set_ylim(0, 1)
    
    # Highlight stress periods
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= weights.index.min() and start_dt <= weights.index.max():
            ax1.axvspan(start_dt, end_dt, alpha=0.3, color='red')
    
    # Active weight deviations
    ax2 = axes[1]
    active = routing_data['active_weights'].fillna(0)
    
    for sector in active.columns:
        color = sector_colors.get(sector, 'gray')
        ax2.plot(active.index, active[sector].rolling(20).mean(), 
                label=sector, color=color, linewidth=1.2)
    
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Active Weight (vs Equal)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Active Weight Deviations (20-day MA)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'stress_routing.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


# =============================================================================
# STEP 6: DIAGNOSTICS & PLOTS
# =============================================================================

def create_diagnostics(fdi_df: pd.DataFrame, rv: pd.DataFrame, output_dir: str):
    """
    Create diagnostic plots for FDI validation.
    
    Plots:
    1. FDI time series with stress periods highlighted
    2. FDI vs realized index volatility
    3. Rolling correlation between FDI and index volatility
    
    Args:
        fdi_df: DataFrame with FDI values
        rv: Daily realized volatility DataFrame
        output_dir: Directory to save plots
    """
    print("\n📊 Creating diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute market-wide realized volatility (equal-weighted average)
    market_rv = rv.mean(axis=1)
    market_rv.index = pd.to_datetime(market_rv.index)
    
    # Align dates
    common_dates = fdi_df.index.intersection(market_rv.index)
    fdi_aligned = fdi_df.loc[common_dates, 'FDI']
    rv_aligned = market_rv.loc[common_dates]
    
    # Compute a simple index return proxy (average of all stock returns)
    # This is a rough proxy for NIFTY index
    
    # --- PLOT 1: FDI Time Series ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # FDI
    ax1 = axes[0]
    ax1.plot(fdi_aligned.index, fdi_aligned.values, 'b-', linewidth=1.2, label='FDI (Smoothed)')
    ax1.axhline(y=fdi_df['FDI'].mean(), color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'Mean FDI ({fdi_df["FDI"].mean():.3f})')
    
    # Highlight stress periods
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= fdi_aligned.index.min() and start_dt <= fdi_aligned.index.max():
            ax1.axvspan(start_dt, end_dt, alpha=0.3, color='red', label=label)
    
    ax1.set_ylabel('FDI (Raw)', fontsize=12)
    ax1.set_title('Feedback Dominance Index (FDI) - NIFTY 500', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Market RV
    ax2 = axes[1]
    ax2.plot(rv_aligned.index, rv_aligned.values, 'purple', linewidth=1, label='Market Realized Vol')
    ax2.set_ylabel('Realized Volatility', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Highlight stress periods on RV plot too
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= rv_aligned.index.min() and start_dt <= rv_aligned.index.max():
            ax2.axvspan(start_dt, end_dt, alpha=0.3, color='red')
    
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, 'fdi_diagnostics.png')
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot1_path}")
    
    # --- PLOT 2: Rolling Correlation ---
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Rolling 60-day correlation between FDI and market RV
    rolling_corr = fdi_aligned.rolling(window=60, min_periods=30).corr(rv_aligned)
    
    ax.plot(rolling_corr.index, rolling_corr.values, 'darkgreen', linewidth=1.2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Rolling 60-day Correlation', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Rolling Correlation: FDI vs Market Realized Volatility', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, 'fdi_rolling_corr.png')
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot2_path}")


# =============================================================================
# STEP 7: VALIDATION CHECKS
# =============================================================================

def validate_and_export(fdi_df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Perform validation checks and export results.
    
    Validation:
    1. FDI z-score spikes (>1.5) before/during major drawdowns
    2. FDI z-score negative during stable bull phases
    3. Flag regimes using z-scores for relative comparison
    
    Args:
        fdi_df: DataFrame with FDI values
        output_file: Path to save CSV
    
    Returns:
        DataFrame with date, FDI, FDI_zscore, regime_flag columns
    """
    print("\n✅ Performing validation and exporting results...")
    
    output = fdi_df[['FDI', 'FDI_zscore']].copy()
    output = output.reset_index()
    output.columns = ['date', 'FDI', 'FDI_zscore']
    
    # Assign regime flags based on z-scores
    # This allows us to identify RELATIVE regime shifts
    def get_regime_flag(zscore):
        if pd.isna(zscore):
            return 'unknown'
        elif zscore > FDI_DESTAB_ZSCORE:
            return 'destabilizing'
        elif zscore < FDI_STAB_ZSCORE:
            return 'stabilizing'
        else:
            return 'neutral'
    
    output['regime_flag'] = output['FDI_zscore'].apply(get_regime_flag)
    
    # Print regime statistics
    regime_counts = output['regime_flag'].value_counts()
    print("\n   Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(output) * 100
        print(f"      {regime}: {count} days ({pct:.1f}%)")
    
    # Check stress periods
    print("\n   Stress Period Analysis:")
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        mask = (output['date'] >= start_dt) & (output['date'] <= end_dt)
        if mask.sum() > 0:
            stress_fdi = output.loc[mask, 'FDI']
            stress_zscore = output.loc[mask, 'FDI_zscore']
            print(f"      {label}:")
            print(f"         Mean FDI: {stress_fdi.mean():.4f}")
            print(f"         Mean Z-score: {stress_zscore.mean():.2f}")
            print(f"         Max Z-score: {stress_zscore.max():.2f}")
            print(f"         Days destabilizing: {(stress_zscore > FDI_DESTAB_ZSCORE).sum()}")
    
    # Export
    output.to_csv(output_file, index=False)
    print(f"\n   ✅ Exported: {output_file}")
    print(f"      Total rows: {len(output)}")
    
    return output


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 60)
    print("  Λ-VOLATILITY DIAGNOSTIC ENGINE (Phase 2)")
    print("=" * 60)
    
    # Step 1: Load data (full ~500 stocks)
    panel = load_data(RAW_DATA_DIR, max_stocks=None)
    
    # Step 2: Compute realized volatility
    rv = compute_realized_vol(panel)
    
    # Step 3: Liquidity normalization
    liq_vol = compute_liquidity_adjustment(panel, rv)
    
    # Step 4a: Compute volatility pressure field
    rho = compute_rho(liq_vol)
    
    # Step 4b: Compute Shock Intensity Index
    sii_df = compute_sii(rho)
    
    # Step 5: Compute FDI
    fdi_df = compute_fdi(rho)
    
    # Step 5b: Combine FDI and SII for 2x2 regime classification
    combined = combine_fdi_sii(fdi_df, sii_df)
    
    # Step 5c: Compute Sectoral FDI
    try:
        sector_mappings = load_sector_mappings()
        sectoral_fdi = compute_sectoral_fdi(rho, sector_mappings)
        
        # Export sectoral FDI
        sectoral_fdi.to_csv("sectoral_fdi_output.csv")
        print(f"   ✅ Exported: sectoral_fdi_output.csv")
        
        # Create sectoral diagnostics
        create_sectoral_diagnostics(sectoral_fdi, OUTPUT_DIR)
    except Exception as e:
        print(f"   ⚠️ Sectoral FDI skipped: {e}")
        sectoral_fdi = None
    
    # Step 5d: Time-of-Day Conditioning (optional, compute-intensive)
    try:
        intraday_fdi = compute_intraday_fdi(panel, rho)
        intraday_fdi.to_csv("intraday_fdi_output.csv")
        print(f"   ✅ Exported: intraday_fdi_output.csv")
        create_intraday_diagnostics(intraday_fdi, OUTPUT_DIR)
    except Exception as e:
        print(f"   ⚠️ Intraday FDI skipped: {e}")
        intraday_fdi = None
    
    # Step 5e: Strategy Integration
    signals = compute_strategy_signals(combined, sectoral_fdi)
    signals.to_csv("strategy_signals.csv")
    print(f"   ✅ Exported: strategy_signals.csv")
    create_strategy_diagnostics(signals, OUTPUT_DIR)
    
    # =========================================================================
    # PHASE 4: ANTICIPATORY FRAMEWORK
    # =========================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4: ANTICIPATORY REGIME FRAMEWORK")
    print("=" * 60)
    
    # Phase 4A: Regime Transition Dynamics
    transition_data = compute_regime_transitions(combined)
    create_transition_diagnostics(transition_data, OUTPUT_DIR)
    
    # Phase 4B: Sector Lead-Lag Analysis
    if sectoral_fdi is not None:
        lead_lag_df, optimal_leads = compute_sector_lead_lag(sectoral_fdi, combined)
        lead_lag_df.to_csv("sector_lead_lag.csv")
        print(f"   ✅ Exported: sector_lead_lag.csv")
        create_lead_lag_diagnostics(lead_lag_df, optimal_leads, OUTPUT_DIR)
        
        # Phase 4C: Portfolio Stress Routing
        routing_data = compute_stress_routing(sectoral_fdi, combined)
        routing_data['normalized_weights'].to_csv("stress_routing_weights.csv")
        print(f"   ✅ Exported: stress_routing_weights.csv")
        create_stress_routing_diagnostics(routing_data, OUTPUT_DIR)
    
    # Step 6: Create diagnostics
    create_diagnostics(fdi_df, rv, OUTPUT_DIR)
    create_sii_diagnostics(combined, rv, OUTPUT_DIR)
    
    # Step 7: Validate and export
    output = validate_and_export_2x2(combined, FDI_OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)
    
    return output


def create_sii_diagnostics(combined: pd.DataFrame, rv: pd.DataFrame, output_dir: str):
    """
    Create diagnostic plots for SII and 2x2 regime classification.
    
    Args:
        combined: DataFrame with FDI, SII, and regime_2x2
        rv: Daily realized volatility DataFrame
        output_dir: Directory to save plots
    """
    print("\n📊 Creating SII diagnostic plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- PLOT: 2x2 Regime Scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for regimes
    color_map = {
        'healthy': '#2ecc71',          # Green
        'absorbed_shock': '#f1c40f',   # Yellow
        'hidden_instability': '#e67e22', # Orange
        'reflexive_crash': '#e74c3c',  # Red
        'unknown': '#95a5a6'           # Gray
    }
    
    for regime in ['healthy', 'absorbed_shock', 'hidden_instability', 'reflexive_crash']:
        mask = combined['regime_2x2'] == regime
        if mask.sum() > 0:
            ax.scatter(
                combined.loc[mask, 'SII_zscore'],
                combined.loc[mask, 'FDI_zscore'],
                c=color_map[regime],
                label=regime.replace('_', ' ').title(),
                alpha=0.5,
                s=20
            )
    
    # Add quadrant lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.axvline(x=SII_HIGH_THRESHOLD, color='k', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add quadrant labels
    ax.text(0.5, 2.5, 'Hidden\nInstability', fontsize=11, ha='center', color='#e67e22', weight='bold')
    ax.text(3.0, 2.5, 'Reflexive\nCrash', fontsize=11, ha='center', color='#e74c3c', weight='bold')
    ax.text(0.5, -2.5, 'Healthy\nMarket', fontsize=11, ha='center', color='#2ecc71', weight='bold')
    ax.text(3.0, -2.5, 'Absorbed\nShock', fontsize=11, ha='center', color='#f1c40f', weight='bold')
    
    ax.set_xlabel('SII Z-score (Shock Magnitude)', fontsize=12)
    ax.set_ylabel('FDI Z-score (Feedback Direction)', fontsize=12)
    ax.set_title('2x2 Regime Classification: FDI vs SII', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fdi_sii_scatter.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # --- PLOT: Time series with regimes ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # SII time series
    ax1 = axes[0]
    ax1.plot(combined.index, combined['SII'], 'purple', linewidth=1.2, label='SII (Smoothed)')
    ax1.axhline(y=combined['SII'].mean(), color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.set_ylabel('SII', fontsize=12)
    ax1.set_title('Shock Intensity Index (SII) and FDI Z-score', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Highlight stress periods
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= combined.index.min() and start_dt <= combined.index.max():
            ax1.axvspan(start_dt, end_dt, alpha=0.3, color='red', label=label)
    
    # FDI z-score time series
    ax2 = axes[1]
    ax2.plot(combined.index, combined['FDI_zscore'], 'b-', linewidth=1.2, label='FDI Z-score')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.axhline(y=FDI_DESTAB_ZSCORE, color='r', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(y=FDI_STAB_ZSCORE, color='g', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('FDI Z-score', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= combined.index.min() and start_dt <= combined.index.max():
            ax2.axvspan(start_dt, end_dt, alpha=0.3, color='red')
    
    # Regime coloring
    ax3 = axes[2]
    regime_numeric = combined['regime_2x2'].map({
        'healthy': 0, 'absorbed_shock': 1, 'hidden_instability': 2, 'reflexive_crash': 3, 'unknown': -1
    })
    colors = [color_map.get(r, '#95a5a6') for r in combined['regime_2x2']]
    ax3.scatter(combined.index, regime_numeric, c=colors, s=5, alpha=0.7)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Healthy', 'Absorbed', 'Hidden', 'Reflexive'])
    ax3.set_ylabel('Regime', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if start_dt >= combined.index.min() and start_dt <= combined.index.max():
            ax3.axvspan(start_dt, end_dt, alpha=0.3, color='red')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'fdi_sii_timeseries.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")


def validate_and_export_2x2(combined: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Perform validation and export 2x2 regime classification.
    
    Args:
        combined: DataFrame with FDI, SII, and regime_2x2
        output_file: Path to save CSV
    
    Returns:
        Output DataFrame
    """
    print("\n✅ Performing validation and exporting results...")
    
    output = combined[['FDI', 'FDI_zscore', 'SII', 'SII_zscore', 'regime_2x2']].copy()
    output = output.reset_index()
    output.columns = ['date', 'FDI', 'FDI_zscore', 'SII', 'SII_zscore', 'regime']
    
    # Print regime statistics
    regime_counts = output['regime'].value_counts()
    print("\n   2x2 Regime Distribution:")
    emoji_map = {'healthy': '🟢', 'absorbed_shock': '🟡', 'hidden_instability': '🟠', 'reflexive_crash': '🔴', 'unknown': '⚪'}
    for regime in ['healthy', 'absorbed_shock', 'hidden_instability', 'reflexive_crash', 'unknown']:
        if regime in regime_counts:
            count = regime_counts[regime]
            pct = count / len(output) * 100
            print(f"      {emoji_map.get(regime, '')} {regime}: {count} days ({pct:.1f}%)")
    
    # Check stress periods
    print("\n   Stress Period Analysis:")
    for start, end, label in STRESS_PERIODS:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        mask = (output['date'] >= start_dt) & (output['date'] <= end_dt)
        if mask.sum() > 0:
            stress_data = output.loc[mask]
            print(f"      {label}:")
            print(f"         Mean FDI Z-score: {stress_data['FDI_zscore'].mean():.2f}")
            print(f"         Mean SII Z-score: {stress_data['SII_zscore'].mean():.2f}")
            
            regime_during = stress_data['regime'].value_counts()
            print(f"         Regime breakdown:")
            for r, c in regime_during.items():
                print(f"            {emoji_map.get(r, '')} {r}: {c} days")
    
    # Export
    output.to_csv(output_file, index=False)
    print(f"\n   ✅ Exported: {output_file}")
    print(f"      Total rows: {len(output)}")
    
    return output


if __name__ == "__main__":
    main()

