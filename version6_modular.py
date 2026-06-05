"""
Modular Momentum Strategy (Version 6 Modular) with ML Regime Filter
-------------------------------------------------------------------------------------
Based on 'version6.py' with specific active configurations:
- Universe: Nifty 100
- Weighting: HRP (Hierarchical Risk Parity)
- Stops: ATR-Based (Dynamic 1.0/2.0/3.0)
- Sector Caps: 25% Max
- Momentum: 70% (12m) + 30% (3m)
- Benchmark: Nifty 100 Buy & Hold
- Crash Predictor: LightGBM + HMM regime confirmation

Generates:
1. Cumulative Returns Plot
2. Monthly Returns Heatmap
3. Drawdown Plot
4. Performance Metrics vs Nifty 100
5. Crash Predictor Diagnostics (confusion matrix, ROC, feature importance)
"""

from numpy import False_
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Crash Predictor imports
try:
    import lightgbm as lgb
    from hmmlearn.hmm import GaussianHMM
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: lightgbm/hmmlearn/sklearn not installed. ML filter disabled.")
    print("Install with: pip install lightgbm hmmlearn scikit-learn")

plt.style.use("seaborn-v0_8-darkgrid")

# ==================== CONFIGURATION ====================

@dataclass
class StrategyConfig:
    # Universe & Portfolio
    universe_file_path: str = r"c:\Users\DELL\Desktop\PYTHON\Historical stock composition of Nifty 50 and Nifty Next 50.xlsx"
    portfolio_size: int =  10
    rebalance_freq: int = 30     # Bi-weekly (balanced)

    
    # Momentum Params
    lookback_12m: int = 252
    lookback_3m: int = 63
    skip_recent: int = 21
    weight_12m: float = 0.70
    weight_3m: float = 0.30
    
    # Trend Filter
    ma_short: int = 50
    ma_long: int = 200
    
    # Risk Management (ATR Stops)
    use_atr_stops: bool = True
    atr_window: int = 20
    atr_k_bull: float = 1.0    # Balanced
    atr_k_neutral: float = 2.0  # Balanced
    atr_k_bear: float = 3.0  # Balanced
    
    # Drawdown Brake
    enable_dd_brake = False
    dd_brake_thresholds = (0.08, 0.15, 0.22)
    dd_brake_exposures  = (0.75, 0.50, 0.25)

    
    # Crowded Trade Exit
    enable_crowding_exit: bool = False
    crowding_lookback_years: int = 3
    crowding_exit_percentile: float = 0.75
    
    # Allocation & Constraints
    use_hrp: bool = True
    use_schur: bool = False  # Use Schur Allocator instead of HRP
    schur_gamma: float = 0.8          # Gamma parameter (0=HRP, 1=MVP)
    hrp_lookback: int = 126
    enable_sector_caps: bool = True
    sector_cap_limit: float = 0.20
    # Costs
    transaction_cost: float = 0.001
    slippage: float = 0.000
    
    # Dates
    start_date: str = "2014-01-01"  # Skip 2020 (COVID black swan)
    end_date: str = "2026-01-01"
    
    # Benchmarks
    benchmark_ticker: str = "^NSEI"  # Nifty 50 (More reliable than CNX100)
    risk_free_rate: float = 0.05
    
    # Nearhigh Filter (Crash Protection)
    enable_nearhigh_filter: bool = False  # Set to False - using weighted scoring instead
    nearhigh_min_threshold: float = 0.85  # Only used if filter enabled
    
    # Nearhigh Weighted Scoring (Prefer stocks near 52-week highs)
    use_nearhigh_scoring: bool = False
    nearhigh_weight: float = 0.50  # Weight for nearhigh in final score (0.0 to disable)
    
    # ML Regime Filter Configuration
    enable_ml_filter: bool = False  # Disabled - using baseline strategy
    
    # LightGBM Regime Filter Params
    enable_lgbm_regime_filter: bool = False  # Enable LightGBM Filter
    lgbm_train_pct: float = 0.70            # Train on first 70%
    lgbm_warmup_days: int = 504             # ~2 years
    lgbm_target_horizon: int = 20           # 20 days forward
    
    ml_train_pct: float = 0.70          # Train on first 70%
    ml_target_horizon: int = 20         # Forward return horizon (trading days)
    ml_warmup_days: int = 504           # Min days before ML can predict (~2 years)
    ml_hmm_states: int = 3              # Number of HMM hidden states
    ml_high_vol_state: int = 2          # Which state is high-vol/reversal (determined post-training)
    
    # FDI Regime Filter Configuration (from volatility_diagnostic.py research)
    enable_fdi_filter: bool = False      # Enable FDI-based regime filtering
    fdi_data_path: str = r"c:\Users\DELL\Desktop\project_nifty_liquid\fdi_output.csv"
    sectoral_fdi_path: str = r"c:\Users\DELL\Desktop\project_nifty_liquid\sectoral_fdi_output.csv"
    enable_fdi_sector_tilts: bool = True  # Apply sector tilts based on sectoral FDI

# ==================== CONSTANTS ====================

TICKER_REMAP = {
    "ABIRLANUVO.NS": "GRASIM.NS", "ADANITRANS.NS": "ADANIENSOL.NS", "AVENTIS.NS": "SANOFI.NS",
    "ANDHRABANK.NS": "UNIONBANK.NS", "BAJAJAUTO.NS": "BAJAJ-AUTO.NS", "CAIRN.NS": "VEDL.NS",
    "CADILAHC.NS": "ZYDUSLIFE.NS", "CORPBANK.NS": "UNIONBANK.NS", "CROMPGREAV.NS": "CGPOWER.NS",
    "GMRINFRA.NS": "GMRAIRPORT.NS", "GSKCONS.NS": "HINDUNILVR.NS", "HEROHONDA.NS": "HEROMOTOCO.NS",
    "HDFC.NS": "HDFCBANK.NS", "I-FLEX.NS": "OFSS.NS", "IDFC.NS": "IDFCFIRSTB.NS",
    "INFOSYSTCH.NS": "INFY.NS", "INFRATEL.NS": "INDUSTOWER.NS", "INGVYSYABK.NS": "KOTAKBANK.NS",
    "L&TFH.NS": "LTF.NS", "LTI.NS": "LTIM.NS", "MCDOWELL-N.NS": "UNITDSPR.NS",
    "MINDTREE.NS": "LTIM.NS", "MOTHERSUMI.NS": "MOTHERSON.NS", "MUNDRAPORT.NS": "ADANIPORTS.NS",
    "RANBAXY.NS": "SUNPHARMA.NS", "PUNJABTRAC.NS": "M&M.NS", "RNRL.NS": "RPOWER.NS",
    "REL.NS": "RINFRA.NS", "RPL.NS": "RELIANCE.NS", "SATYAMCOMP.NS": "TECHM.NS",
    "SESAGOA.NS": "VEDL.NS", "STER.NS": "VEDL.NS", "SSLT.NS": "VEDL.NS",
    "SRTRANSFIN.NS": "SHRIRAMFIN.NS", "TATAGLOBAL.NS": "TATACONSUM.NS", "SYNDIBANK.NS": "CANBK.NS",
    "TATAMTRDVR.NS": "TATAMOTORS.NS", "TMPV.NS": "TATAMOTORS.NS", "UNIPHOS.NS": "UPL.NS",
    "VSNL.NS": "TATACOMM.NS", "VIJAYABANK.NS": "BANKBARODA.NS",
    "ENRIN.NS": None, "HYUNDAI.NS": None, "IBULHSGFIN.NS": None, "NICOLASPIR.NS": None,
    "NIRMA.NS": None, "PATNI.NS": None, "PEL.NS": None, "SWIGGY.NS": None, "ZOMATO.NS": None
}

# ==================== DATA & UNIVERSE ====================

class UniverseManager:
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.universe_map = {}
        self.all_tickers = []
        self._load_universe()

    def _load_universe(self):
        print("Loading Dynamic Universe...")
        try:
            xls = pd.ExcelFile(self.config.universe_file_path)
            sheets = ['Nifty 50 Data', 'Nifty Next 50 Data']
            temp_map = {}
            all_set = set()

            for sheet in sheets:
                if sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    for col in df.columns:
                        try:
                            date_obj = pd.to_datetime(col)
                            if pd.isna(date_obj): continue
                            
                            tickers = df[col].dropna().astype(str).tolist()
                            clean = []
                            for t in tickers:
                                t = t.strip().upper()
                                if t == "NAN": continue
                                if not t.endswith(".NS"): t += ".NS"
                                if t in TICKER_REMAP:
                                    t = TICKER_REMAP[t]
                                    if t is None: continue
                                clean.append(t)
                            
                            if date_obj not in temp_map: temp_map[date_obj] = []
                            temp_map[date_obj].extend(clean)
                        except: continue

            for d, t_list in temp_map.items():
                self.universe_map[d] = sorted(list(set(t_list)))
                all_set.update(t_list)
            
            self.all_tickers = sorted(list(all_set))
            # DEBUG: Limit tickers to speed up verification of RF integration
            # print("DEBUG: Limiting universe to 50 tickers for speed.")
            # self.all_tickers = self.all_tickers[:50]
            
            print(f"Loaded {len(self.universe_map)} months, {len(self.all_tickers)} unique tickers.")
        except Exception as e:
            print(f"Error loading universe: {e}")

    def get_universe(self, date):
        dates = sorted([d for d in self.universe_map.keys() if d <= date])
        return self.universe_map[dates[-1]] if dates else []

class DataManager:
    def __init__(self, config: StrategyConfig, universe_manager: UniverseManager):
        self.config = config
        self.um = universe_manager
        self.price_data = {}
        self.sectors = {}
        self.benchmark_data = None
        self.index_data = None # For regime check (using Nifty 50 usually, or user defined)
        self.macro_data = pd.DataFrame() # Store VIX, Yields etc.

    def fetch_data(self):
        print(f"Fetching data for {len(self.um.all_tickers)} tickers...")
        
        # Calculate start date with buffer
        # INCREASED BUFFER FOR CROWDING HISTORY (Requires ~3 years lookback + warm up)
        start_dt = datetime.strptime(self.config.start_date, "%Y-%m-%d") - timedelta(days=365*6)
        start_str = start_dt.strftime("%Y-%m-%d")

        def _fetch(t):
            try:
                tk = yf.Ticker(t)
                hist = tk.history(start=start_str, end=self.config.end_date, auto_adjust=True)
                if hist.empty or len(hist) < 200: return None
                
                # TZ conversion
                hist.index = pd.to_datetime(hist.index).tz_localize(None)
                
                # Sector
                sector = "Unknown"
                try: 
                    info = tk.get_info() if hasattr(tk, "get_info") else tk.info
                    sector = info.get("sector", "Unknown")
                except: pass
                
                return t, hist, sector
            except: return None

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(_fetch, t) for t in self.um.all_tickers]
            for f in as_completed(futures):
                res = f.result()
                if res:
                    self.price_data[res[0]] = res[1]
                    self.sectors[res[0]] = res[2]
        
        print(f"Successfully fetched {len(self.price_data)} tickers.")
        
        # Fetch Benchmark (Nifty 100 or NSEI)
        print(f"Fetching Benchmark: {self.config.benchmark_ticker}")
        try:
            bench = yf.Ticker(self.config.benchmark_ticker).history(start=start_str, end=self.config.end_date, auto_adjust=True)
            if bench.empty: raise ValueError("Empty benchmark data")
            bench.index = pd.to_datetime(bench.index).tz_localize(None)
            self.benchmark_data = bench["Close"]
        except Exception as e:
            print(f"Benchmark fetch failed ({self.config.benchmark_ticker}): {e}")
            print("Attempting fallback to ^NSEI...")
            try:
                bench = yf.Ticker("^NSEI").history(start=start_str, end=self.config.end_date, auto_adjust=True)
                if bench.empty: raise ValueError("Empty fallback benchmark")
                bench.index = pd.to_datetime(bench.index).tz_localize(None)
                self.benchmark_data = bench["Close"]
                print("Fallback to ^NSEI successful.")
            except Exception as e2:
                print(f"Fallback failed: {e2}")

        # Fetch Index for Regime (Nifty 50) - defaulted to ^NSEI logic from original
        try:
            idx = yf.Ticker("^NSEI").history(start=start_str, end=self.config.end_date, auto_adjust=True)
            idx.index = pd.to_datetime(idx.index).tz_localize(None)
            self.index_data = idx["Close"]
        except: pass
        
        # Fetch Macro Data (VIX, Yields)
        self.fetch_macro_data(start_str)

    def fetch_macro_data(self, start_date):
        print("Fetching Macro Data (VIX, Yields)...")
        macro_dict = {}
        
        # 1. India VIX (using ^INDIAVIX or proxy ^VIX if unavailable)
        # Using US VIX as proxy for global risk if India VIX fails or is short
        try:
            vix = yf.Ticker("^VIX").history(start=start_date, end=self.config.end_date, auto_adjust=True)["Close"]
            vix.index = pd.to_datetime(vix.index).tz_localize(None)
            macro_dict["VIX"] = vix
        except: print("Failed to fetch VIX")

        # 2. US Treasury Yields (10y and 2y) for Yield Curve
        try:
            tnx = yf.Ticker("^TNX").history(start=start_date, end=self.config.end_date, auto_adjust=True)["Close"] # 10y
            tnx.index = pd.to_datetime(tnx.index).tz_localize(None)
            macro_dict["US10Y"] = tnx
            
            # 2y often ^IRX is 13 week, ^FVX is 5 year. ^TwoYear? No. 
            # Using 5Y as proxy for short end if 2Y unavailable, or just 10Y level.
            # actually ^IRX is 3-month. Yield curve slope = 10Y - 3M is common.
            irx = yf.Ticker("^IRX").history(start=start_date, end=self.config.end_date, auto_adjust=True)["Close"]
            irx.index = pd.to_datetime(irx.index).tz_localize(None)
            macro_dict["US3M"] = irx
        except: print("Failed to fetch Yields")
        
        # 3. Credit Spread Proxy (LQD vs HYG) - US Corporate Inv Grade vs High Yield
        try:
            lqd = yf.Ticker("LQD").history(start=start_date, end=self.config.end_date, auto_adjust=True)["Close"]
            hyg = yf.Ticker("HYG").history(start=start_date, end=self.config.end_date, auto_adjust=True)["Close"]
            if not lqd.empty and not hyg.empty:
                # Align
                lqd.index = pd.to_datetime(lqd.index).tz_localize(None)
                hyg.index = pd.to_datetime(hyg.index).tz_localize(None)
                # Spread proxy: Ratio or difference. Rising LQD/HYG means stress (Flight to quality)
                # Or HYG/LQD dropping means stress. 
                # Credit spread usually Yield(JNK) - Yield(AAA). 
                # Price wise: LQD and HYG move differently. 
                # Let's save prices, compute spread feature later.
                macro_dict["LQD"] = lqd
                macro_dict["HYG"] = hyg
        except: print("Failed to fetch Credit Proxies")

        self.macro_data = pd.DataFrame(macro_dict).fillna(method='ffill')
        print(f"Macro Data fetched: {self.macro_data.shape}")

    def get_price(self, ticker): return self.price_data.get(ticker)
    def get_sector(self, ticker): return self.sectors.get(ticker, "Unknown")

# ==================== CALCULATORS ====================

class Calculator:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def momentum_score(self, prices):
        if len(prices) < self.cfg.lookback_12m: return np.nan, np.nan
        
        # 1. 12m Average Monthly Return (Excluding recent)
        end_idx_12 = -self.cfg.skip_recent - 1
        hist_12 = prices.iloc[-self.cfg.lookback_12m : end_idx_12]
        # Calculate mean daily return and annualize to monthly (x21)
        mom12 = hist_12.pct_change().mean() * 21

        # 2. 3m Average Monthly Return
        hist_3 = prices.iloc[-self.cfg.lookback_3m:]
        mom3 = hist_3.pct_change().mean() * 21
        
        return mom12, mom3

    def trend_ok(self, prices):
        if len(prices) < self.cfg.ma_long: return False
        ema_s = prices.ewm(span=self.cfg.ma_short, adjust=False).mean().iloc[-1]
        ema_l = prices.ewm(span=self.cfg.ma_long, adjust=False).mean().iloc[-1]
        return (prices.iloc[-1] > ema_l) and (ema_s > ema_l)
    
    def nearhigh_ratio(self, prices):
        """
        Calculate Nearhigh = Current Price / 52-Week High
        Returns value between 0 and 1.
        Higher = closer to 52-week high (safer for momentum strategies)
        """
        if len(prices) < 252:
            return np.nan
        high_52w = prices.iloc[-252:].max()
        current = prices.iloc[-1]
        return current / high_52w if high_52w > 0 else np.nan

class ATRManager:
    def __init__(self, config: StrategyConfig):
        self.cfg = config

    def get_stop_loss_pct(self, df, regime="NEUTRAL"):
        if not self.cfg.use_atr_stops: return -0.10 # Fallback
        
        # Compute ATR
        high, low, close = df["High"].values, df["Low"].values, df["Close"].values
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        atr = np.mean(tr[-self.cfg.atr_window:])
        
        # Determine K
        k = self.cfg.atr_k_neutral
        if regime == "BULL": k = self.cfg.atr_k_bull
        elif regime == "BEAR": k = self.cfg.atr_k_bear
        
        # Monthly scaling approx
        atr_monthly = atr * np.sqrt(21)
        sl_pct = -1.0 * (k * atr_monthly) / close[-1]
        return max(min(sl_pct, -0.02), -0.50)

# ==================== ALLOCATOR (HRP) ====================

class HRPAllocator:
    @staticmethod
    def get_weights(prices_df):
        returns = prices_df.pct_change().dropna()
        if returns.empty: return {}
        
        cov, corr = returns.cov(), returns.corr()
        dist = ssd.squareform(ssd.pdist(corr))
        link = sch.linkage(dist, 'single')
        sort_ix = HRPAllocator._get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        hrp = HRPAllocator._get_rec_bipart(cov, sort_ix)
        return hrp.to_dict()

    @staticmethod
    def _get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    @staticmethod
    def _get_rec_bipart(cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c0, c1 = c_items[i], c_items[i+1]
                v0 = HRPAllocator._cluster_var(cov, c0)
                v1 = HRPAllocator._cluster_var(cov, c1)
                alpha = 1 - v0 / (v0 + v1)
                w[c0] *= alpha
                w[c1] *= 1 - alpha
        return w

    @staticmethod
    def _cluster_var(cov, items):
        cov_slice = cov.loc[items, items]
        w = (1. / np.diag(cov_slice))
        w /= w.sum()
        w = w.reshape(-1, 1)
        return np.dot(np.dot(w.T, cov_slice), w)[0, 0]


# ==================== SCHUR ALLOCATOR (arxiv 2411.05807) ====================

class SchurAllocator:
    """
    Schur Complementary Allocation (arxiv 2411.05807 by Peter Cotton)
    Unifies HRP and MVP via gamma parameter:
        - gamma=0: Equivalent to HRP (ignores off-diagonal covariance)
        - gamma=1: Converges to MVP (full covariance)
        - gamma=0.5: Balanced interpolation (default)
    """
    
    @staticmethod
    def get_weights(prices_df, gamma: float = 0.5):
        """
        Compute portfolio weights using Schur Complementary Allocation.
        
        Args:
            prices_df: DataFrame of asset prices
            gamma: Regularization parameter (0=HRP, 1=MVP)
        
        Returns:
            dict: Asset weights
        """
        returns = prices_df.pct_change().dropna()
        if returns.empty:
            return {}
        
        cov = returns.cov()
        corr = returns.corr()
        
        # 1. Distance matrix and hierarchical clustering (same as HRP)
        dist = ssd.squareform(ssd.pdist(corr))
        link = sch.linkage(dist, 'single')
        
        # 2. Quasi-diagonalization (same as HRP)
        sort_ix = SchurAllocator._get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        # 3. Schur-augmented recursive bisection (differs from HRP)
        weights = SchurAllocator._get_rec_bipart_schur(cov, sort_ix, gamma)
        return weights.to_dict()
    
    @staticmethod
    def _get_quasi_diag(link):
        """Quasi-diagonalization via seriation (same as HRP)."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()
    
    @staticmethod
    def _get_rec_bipart_schur(cov, sort_ix, gamma):
        """
        Recursive bisection with Schur complement augmented variances.
        
        The key difference from HRP: Instead of using raw sub-covariance matrices,
        we augment them using the Schur complement to incorporate cross-cluster
        correlation information, scaled by gamma.
        """
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix]
        
        while len(c_items) > 0:
            # Split each cluster into two
            c_items = [i[j:k] for i in c_items 
                       for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) 
                       if len(i) > 1]
            
            for i in range(0, len(c_items), 2):
                c0, c1 = c_items[i], c_items[i + 1]
                
                # Compute Schur-augmented cluster variances
                v0 = SchurAllocator._cluster_var_schur(cov, c0, c1, gamma)
                v1 = SchurAllocator._cluster_var_schur(cov, c1, c0, gamma)
                
                # Inverse-variance allocation between clusters
                alpha = 1 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
                w[c0] *= alpha
                w[c1] *= 1 - alpha
        
        return w
    
    @staticmethod
    def _cluster_var_schur(cov, items_a, items_b, gamma):
        """
        Compute Schur-augmented cluster variance.
        
        For cluster A with respect to cluster B:
        A_aug = A + gamma * B_cov @ inv(D) @ B_cov.T
        
        Where:
            A = cov[items_a, items_a] (covariance of cluster A)
            D = cov[items_b, items_b] (covariance of cluster B)
            B_cov = cov[items_a, items_b] (cross-covariance)
        
        gamma=0: Uses raw A (like HRP)
        gamma=1: Full Schur complement augmentation (approaches MVP)
        """
        # Extract sub-matrices
        A = cov.loc[items_a, items_a].values
        D = cov.loc[items_b, items_b].values
        B = cov.loc[items_a, items_b].values
        
        # Compute augmented covariance if gamma > 0
        if gamma > 0 and len(items_b) > 0:
            try:
                # Add small regularization for numerical stability
                D_reg = D + np.eye(len(items_b)) * 1e-8
                D_inv = np.linalg.inv(D_reg)
                
                # Schur complement term: B @ D^-1 @ B.T
                schur_term = B @ D_inv @ B.T
                
                # Augmented covariance: A + gamma * schur_term
                A_aug = A + gamma * schur_term
            except np.linalg.LinAlgError:
                # Fallback to non-augmented if inversion fails
                A_aug = A
        else:
            A_aug = A
        
        # Compute cluster variance using inverse-variance weights
        diag = np.diag(A_aug)
        diag = np.maximum(diag, 1e-10)  # Ensure positive
        w = 1.0 / diag
        w = w / w.sum()
        w = w.reshape(-1, 1)
        
        return np.dot(np.dot(w.T, A_aug), w)[0, 0]

# ==================== CRASH PREDICTOR V2 ====================

class CrashPredictor:
    """
    Simplified ML filter V2 - predicts momentum crashes.
    
    Key improvements over V1:
    - Predicts CRASHES (next_20d_return < -8%) not general returns
    - Uses Logistic Regression (less overfitting)
    - Only 5 important features
    - Linear exposure mapping: exposure = 1 - crash_probability
    - Rule-based circuit breakers for extreme cases
    """
    
    def __init__(self, config: StrategyConfig, dm, um):
        self.cfg = config
        self.dm = dm
        self.um = um
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = []
        self.factor_equity = None
        self.test_predictions = []
    
    def set_factor_equity(self, equity_series: pd.Series):
        """Set the momentum basket equity curve for factor calculations."""
        self.factor_equity = equity_series
    
    def _compute_features(self, date, lookback_data=None) -> Dict:
        """
        Compute 5 simplified features using ONLY past data.
        """
        features = {}
        
        # Get benchmark history
        bench = self.dm.benchmark_data
        if bench is None or bench.empty:
            return None
        
        bench_hist = bench[bench.index < date]
        if len(bench_hist) < 126:
            return None
        
        # 1. realized_vol_60 - 60-day realized volatility
        log_ret = np.log(bench_hist / bench_hist.shift(1)).dropna()
        if len(log_ret) >= 60:
            features["realized_vol_60"] = log_ret.iloc[-60:].std() * np.sqrt(252)
        else:
            return None
        
        # 2. vol_regime - vol spike detection (vol_20 / vol_60)
        vol_20 = log_ret.iloc[-20:].std() * np.sqrt(252) if len(log_ret) >= 20 else features["realized_vol_60"]
        features["vol_regime"] = vol_20 / features["realized_vol_60"] if features["realized_vol_60"] > 0 else 1.0
        
        # 3. market_trend - Nifty distance from 200MA (%)
        if len(bench_hist) >= 200:
            ma_200 = bench_hist.iloc[-200:].mean()
            features["market_trend"] = (bench_hist.iloc[-1] / ma_200) - 1.0
        else:
            features["market_trend"] = 0.0
        
        # 4. factor_drawdown - current drawdown of momentum factor
        if self.factor_equity is not None:
            factor_hist = self.factor_equity[self.factor_equity.index < date]
            if len(factor_hist) > 20:
                peak = factor_hist.cummax()
                dd = (factor_hist / peak) - 1.0
                features["factor_drawdown"] = dd.iloc[-1] if len(dd) > 0 else 0.0
            else:
                features["factor_drawdown"] = 0.0
        else:
            features["factor_drawdown"] = 0.0
        
        # 5. factor_momentum - recent factor return (20d)
        if self.factor_equity is not None:
            factor_hist = self.factor_equity[self.factor_equity.index < date]
            if len(factor_hist) > 20:
                features["factor_momentum"] = (factor_hist.iloc[-1] / factor_hist.iloc[-20]) - 1.0
            else:
                features["factor_momentum"] = 0.0
        else:
            features["factor_momentum"] = 0.0
        
        return features
    
    def train(self, equity_curve: pd.Series, dates: pd.DatetimeIndex):
        """
        Train Logistic Regression to predict crashes.
        Target: 1 if next_20d_return < -8% (crash), else 0
        """
        if not ML_AVAILABLE:
            print("ML libraries not available. Skipping training.")
            return
        
        print("Training Crash Predictor V2...")
        self.set_factor_equity(equity_curve)
        
        X_list = []
        y_list = []
        date_list = []
        
        crash_threshold = -0.08  # -8% defines a "crash"
        
        # Build training data
        for i in range(self.cfg.ml_warmup_days, len(dates) - self.cfg.ml_target_horizon):
            date = dates[i]
            
            features = self._compute_features(date)
            if features is None:
                continue
            
            # Compute target: is there a crash in the next 20 days?
            future_date = dates[i + self.cfg.ml_target_horizon]
            if date not in equity_curve.index or future_date not in equity_curve.index:
                continue
            
            current_val = equity_curve.loc[date]
            future_val = equity_curve.loc[future_date]
            forward_return = (future_val / current_val) - 1.0
            
            # Target = 1 if crash, 0 if not
            target = 1 if forward_return < crash_threshold else 0
            
            X_list.append(features)
            y_list.append(target)
            date_list.append(date)
        
        if len(X_list) < 100:
            print(f"Insufficient samples for training: {len(X_list)}")
            return
        
        X_df = pd.DataFrame(X_list)
        y = np.array(y_list)
        self.feature_names = list(X_df.columns)
        
        # Chronological split
        split_idx = int(len(X_df) * self.cfg.ml_train_pct)
        X_train, X_test = X_df.iloc[:split_idx], X_df.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Crash rate in train: {y_train.mean():.1%}, in test: {y_test.mean():.1%}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression with balanced classes
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(
            class_weight='balanced',  # Handle imbalanced data
            C=0.1,                     # Regularization
            max_iter=1000,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\n=== Crash Predictor V2 Results ===")
        print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.3f}")
        print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.3f}")
        
        if y_test.sum() > 0:  # Only if there are actual crashes
            print(f"Test Precision: {precision_score(y_test, test_pred, zero_division=0):.3f}")
            print(f"Test Recall: {recall_score(y_test, test_pred, zero_division=0):.3f}")
            try:
                print(f"Test ROC-AUC: {roc_auc_score(y_test, test_proba):.3f}")
            except:
                print("Test ROC-AUC: N/A (single class)")
        else:
            print("No crashes in test period to evaluate.")
        
        # Store test predictions for diagnostics
        test_dates = [date_list[split_idx + i] for i in range(len(X_test))]
        self.test_predictions = list(zip(test_dates, y_test.tolist(), test_proba.tolist()))
        
        # Store feature coefficients
        self.feature_importances = dict(zip(self.feature_names, 
                                            np.abs(self.model.coef_[0]).tolist()))
        
        self.is_trained = True
        print("Crash Predictor V2 training complete.")
    
    def get_exposure(self, date) -> float:
        """
        Get exposure scaling for a given date.
        Returns value in [0, 1].
        
        Uses:
        1. Rule-based circuit breakers for extreme cases
        2. ML probability for nuanced scaling
        """
        if not self.is_trained or not ML_AVAILABLE:
            return 1.0
        
        # Compute features
        features = self._compute_features(date)
        if features is None:
            return 1.0
        
        # ==== RULE-BASED CIRCUIT BREAKERS (LAYER 1) ====
        factor_dd = features.get("factor_drawdown", 0.0)
        vol_60 = features.get("realized_vol_60", 0.15)
        vol_regime = features.get("vol_regime", 1.0)
        
        # Hard exit if factor in severe drawdown + vol spike
        if factor_dd < -0.12 and vol_regime > 1.3:
            return 0.20
        
        # Defensive if factor struggling or vol extreme
        if factor_dd < -0.08:
            return 0.40
        
        if vol_60 > 0.30:
            return 0.35
        
        # ==== ML-BASED SCALING (LAYER 2) ====
        # Predict crash probability
        X = pd.DataFrame([features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        crash_prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # Linear mapping: exposure = 1 - crash_probability
        # But with floor of 0.40 to avoid over-cutting
        exposure = 1.0 - crash_prob
        exposure = max(0.40, exposure)
        
        # Mild vol adjustment
        if vol_60 > 0.22:
            exposure = min(exposure, 0.70)
        
        return exposure
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostics for reporting."""
        if not self.is_trained:
            return {}
        
        return {
            "feature_importances": self.feature_importances,
            "test_predictions": self.test_predictions,
        }



class LightGBMRegimeFilter:
    """
    LightGBM-based Regime Filter.
    Classifies market state as 'Safe' (1) or 'Dangerous' (0).
    """
    def __init__(self, config: StrategyConfig, dm, um):
        self.cfg = config
        self.dm = dm
        self.um = um
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = []
        self.factor_equity = None
        self.test_predictions = []
        self.feature_importances = {}
        
    def set_factor_equity(self, equity_series: pd.Series):
        """Set the momentum basket equity curve for factor calculations."""
        self.factor_equity = equity_series
    
    def _compute_features(self, date) -> Dict:
        """
        Compute features for RF model using ONLY past data.
        """
        features = {}
        
        # 1. Macro Features (VIX, Yields)
        macro = self.dm.macro_data
        if macro.empty: return None
        
        # Use data strictly < date
        macro_hist = macro[macro.index < date]
        if macro_hist.empty: return None
        
        # VIX Features
        if "VIX" in macro_hist.columns:
            vix_curr = macro_hist["VIX"].iloc[-1]
            vix_ma20 = macro_hist["VIX"].iloc[-20:].mean() if len(macro_hist) >= 20 else vix_curr
            vix_ma60 = macro_hist["VIX"].iloc[-60:].mean() if len(macro_hist) >= 60 else vix_curr
            
            features["VIX_level"] = vix_curr
            features["VIX_term_structure"] = vix_curr - vix_ma60 # Proxy
            features["vol_regime"] = vix_ma20 / vix_ma60 if vix_ma60 > 0 else 1.0
            
        # Yield Curve
        if "US10Y" in macro_hist.columns and "US3M" in macro_hist.columns:
            y10 = macro_hist["US10Y"].iloc[-1]
            y3m = macro_hist["US3M"].iloc[-1]
            features["yield_curve_slope"] = y10 - y3m
            
        # Credit Spread Proxy (LQD vs HYG) -> Ratio
        if "LQD" in macro_hist.columns and "HYG" in macro_hist.columns:
            lqd = macro_hist["LQD"].iloc[-1]
            hyg = macro_hist["HYG"].iloc[-1]
            features["credit_spread_proxy"] = lqd / hyg if hyg > 0 else 0
            
        # 2. Market Features (Benchmark)
        bench = self.dm.benchmark_data
        bench_hist = bench[bench.index < date]
        if len(bench_hist) < 200: return None
        
        ma200 = bench_hist.iloc[-200:].mean()
        features["breadth"] = (bench_hist.iloc[-1] / ma200) - 1.0
        
        ret = bench_hist.pct_change()
        features["realized_vol_20"] = ret.iloc[-20:].std() * np.sqrt(252)
        features["realized_vol_60"] = ret.iloc[-60:].std() * np.sqrt(252)
        
        # 3. Momentum Factor Diagnostics
        if self.factor_equity is not None:
            fac_hist = self.factor_equity[self.factor_equity.index < date]
            if len(fac_hist) > 60:
                peak = fac_hist.cummax()
                dd = (fac_hist / peak) - 1.0
                features["momentum_factor_dd"] = dd.iloc[-1]
                features["momentum_factor_return_20d"] = (fac_hist.iloc[-1] / fac_hist.iloc[-20]) - 1.0
                features["momentum_factor_volatility"] = fac_hist.pct_change().iloc[-20:].std() * np.sqrt(252)
            else:
                features["momentum_factor_dd"] = 0.0
                features["momentum_factor_return_20d"] = 0.0
                features["momentum_factor_volatility"] = 0.15
        else:
            features.update({
                "momentum_factor_dd": 0.0,
                "momentum_factor_return_20d": 0.0,
                "momentum_factor_volatility": 0.0
            })
            
        # 4. Cross-Sectional (Beta Reversal Proxy)
        if self.factor_equity is not None and "momentum_factor_return_20d" in features:
            bench_ret_20 = (bench_hist.iloc[-1] / bench_hist.iloc[-20]) - 1.0
            features["beta_reversal"] = bench_ret_20 - features["momentum_factor_return_20d"]
            
        return features

    def train(self, equity_curve: pd.Series, dates: pd.DatetimeIndex):
        """
        Train Random Forest.
        Label = 1 (Safe) if Forward 20d > 0 and DD < 10% inside horizon.
        Label = 0 (Dangerous) if Forward 20d < -5% OR DD > 15%.
        """
        if not ML_AVAILABLE: return
        
        print("Training Random Forest Regime Filter...")
        self.set_factor_equity(equity_curve)
        
        X_list, y_list, date_list = [], [], []
        
        horizon = self.cfg.lgbm_target_horizon
        
        # Build Dataset
        for i in range(self.cfg.lgbm_warmup_days, len(dates) - horizon):
            d = dates[i]
            feat = self._compute_features(d)
            if feat is None: continue
            
            future_d = dates[i + horizon]
            if d not in equity_curve.index or future_d not in equity_curve.index: continue
            
            p_curr = equity_curve.loc[d]
            p_future = equity_curve.loc[future_d]
            fwd_ret = (p_future / p_curr) - 1.0
            
            path = equity_curve.loc[d:future_d]
            peak = path.cummax()
            dd = (path / peak) - 1.0
            max_dd_horizon = dd.min()
            
            label = -1
            if fwd_ret > 0.0 and max_dd_horizon > -0.10:
                label = 1 # Safe
            elif fwd_ret < -0.05 or max_dd_horizon < -0.15:
                label = 0 # Dangerous
                
            if label != -1:
                X_list.append(feat)
                y_list.append(label)
                date_list.append(d)
                
        if len(X_list) < 100:
            print(f"RF: Insufficient samples {len(X_list)}")
            return

        X_df = pd.DataFrame(X_list).fillna(0)
        y = np.array(y_list)
        self.feature_names = list(X_df.columns)
        
        split = int(len(X_df) * self.cfg.lgbm_train_pct)
        X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"RF Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"Class Balance (Safe=1): Train {y_train.mean():.2f}, Test {y_test.mean():.2f}")
        
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # LightGBM Classifier
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            class_weight={0: 2.0, 1: 1.0}, # Moderate Penalty
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        self.model.fit(X_train_s, y_train)
        
        tr_acc = accuracy_score(y_train, self.model.predict(X_train_s))
        te_pred = self.model.predict(X_test_s)
        te_prob = self.model.predict_proba(X_test_s)[:, 1]
        te_acc = accuracy_score(y_test, te_pred)
        
        print(f"LGBM Results: Train Acc {tr_acc:.2f}, Test Acc {te_acc:.2f}")
        
        if len(set(y_test)) > 1:
            print(f"Test ROC-AUC: {roc_auc_score(y_test, te_prob):.2f}")
            
        self.feature_importances = dict(zip(self.feature_names, self.model.feature_importances_))
        
        test_dates = [date_list[split + i] for i in range(len(X_test))]
        self.test_predictions = list(zip(test_dates, y_test.tolist(), te_prob.tolist()))
        
        self.is_trained = True

    def get_exposure(self, date) -> float:
        if not self.is_trained: return 1.0
        
        feat = self._compute_features(date)
        if feat is None: return 1.0
        
        X = pd.DataFrame([feat], columns=self.feature_names).fillna(0)
        X_s = self.scaler.transform(X)
        
        safe_prob = self.model.predict_proba(X_s)[0, 1]
        exposure = max(0.25, min(safe_prob, 1.0))
        
        return exposure
    
    def get_diagnostics(self):
        return {
            "feature_importances": self.feature_importances,
            "test_predictions": self.test_predictions
        }

# ==================== FDI REGIME FILTER ====================

class FDIRegimeFilter:
    """
    FDI (Feedback Dominance Index) based Regime Filter.
    
    Uses pre-computed FDI data from volatility_diagnostic.py:
    - fdi_output.csv: Market-level FDI z-scores and regime classifications
    - sectoral_fdi_output.csv: Sector-level FDI for early warning (NBFCs lead 9-10 days)
    
    Key insights from research:
    - Use FDI as RISK OVERLAY, not trading signal
    - NBFCs/Banks FDI leads market by 9-10 days → early warning
    - Absorbed shock → Reflexive transition is most dangerous (35.6%)
    - Don't trade crossings; use for position sizing
    """
    
    def __init__(self, config: StrategyConfig, fdi_data_path: str = None, sectoral_path: str = None):
        self.cfg = config
        self.fdi_data = None
        self.sectoral_fdi = None
        self.is_loaded = False
        
        # Default paths relative to project
        self.fdi_path = fdi_data_path or r"c:\Users\DELL\Desktop\project_nifty_liquid\fdi_output.csv"
        self.sectoral_path = sectoral_path or r"c:\Users\DELL\Desktop\project_nifty_liquid\sectoral_fdi_output.csv"
        
        # Load data
        self._load_fdi_data()
    
    def _load_fdi_data(self):
        """Load pre-computed FDI data from CSV files."""
        import os
        
        try:
            if os.path.exists(self.fdi_path):
                self.fdi_data = pd.read_csv(self.fdi_path, parse_dates=['date'], index_col='date')
                print(f"FDI Filter: Loaded market FDI ({len(self.fdi_data)} days)")
            else:
                print(f"FDI Filter: File not found: {self.fdi_path}")
                return
                
            if os.path.exists(self.sectoral_path):
                self.sectoral_fdi = pd.read_csv(self.sectoral_path, parse_dates=['date'], index_col='date')
                print(f"FDI Filter: Loaded sectoral FDI ({len(self.sectoral_fdi)} days, {len(self.sectoral_fdi.columns)} sectors)")
            else:
                print(f"FDI Filter: Sectoral file not found: {self.sectoral_path}")
                
            self.is_loaded = True
            
        except Exception as e:
            print(f"FDI Filter: Error loading data: {e}")
            self.is_loaded = False
    
    def get_fdi_metrics(self, date) -> dict:
        """
        Get FDI metrics for a given date.
        Returns dict with FDI_zscore, regime, NBFC warning, etc.
        """
        if not self.is_loaded or self.fdi_data is None:
            return None
        
        # Find closest date on or before the request date
        available_dates = self.fdi_data.index[self.fdi_data.index <= date]
        if len(available_dates) == 0:
            return None
        
        closest_date = available_dates[-1]
        row = self.fdi_data.loc[closest_date]
        
        metrics = {
            'date': closest_date,
            'fdi_zscore': row.get('FDI_zscore', 0),
            'sii_zscore': row.get('SII_zscore', 0),
            'regime': row.get('regime', 'healthy'),
        }
        
        # Check NBFC early warning (if sectoral data available)
        if self.sectoral_fdi is not None:
            sect_dates = self.sectoral_fdi.index[self.sectoral_fdi.index <= date]
            if len(sect_dates) > 0:
                sect_date = sect_dates[-1]
                sect_row = self.sectoral_fdi.loc[sect_date]
                
                # NBFC early warning: 3+ days above 1.5
                if 'NBFCs' in self.sectoral_fdi.columns:
                    nbfc_fdi = sect_row.get('NBFCs', 0)
                    metrics['nbfc_fdi'] = nbfc_fdi
                    
                    # Check if NBFCs have been elevated for 3+ days
                    if len(sect_dates) >= 3:
                        recent_nbfc = self.sectoral_fdi.loc[sect_dates[-3:], 'NBFCs']
                        metrics['nbfc_warning'] = (recent_nbfc > 1.5).all()
                    else:
                        metrics['nbfc_warning'] = False
                        
                # Banks early warning
                if 'Banks' in self.sectoral_fdi.columns:
                    metrics['banks_fdi'] = sect_row.get('Banks', 0)
        
        return metrics
    
    def get_exposure(self, date) -> float:
        """
        Get exposure scaling based on FDI regime.
        Returns value in [0.25, 1.25].
        
        Key rules from research:
        - FDI z-score > 2.0: 25% exposure (severe reduce)
        - FDI z-score > 1.5: 50% exposure (reduce)
        - FDI z-score < -1.5: 125% exposure (stabilizing, can overweight)
        - FDI z-score < -1.0: 110% exposure (slightly overweight)
        - NBFC warning active: Max 60% exposure
        - Absorbed shock regime + FDI rising: 30% exposure (danger zone!)
        """
        if not self.is_loaded:
            return 1.0
        
        metrics = self.get_fdi_metrics(date)
        if metrics is None:
            return 1.0
        
        fdi_z = metrics.get('fdi_zscore', 0)
        regime = metrics.get('regime', 'healthy')
        nbfc_warning = metrics.get('nbfc_warning', False)
        
        # Base exposure from FDI z-score
        if fdi_z > 2.0:
            exposure = 0.25
        elif fdi_z > 1.5:
            exposure = 0.50
        elif fdi_z > 0.5:
            exposure = 0.75
        elif fdi_z < -1.5:
            exposure = 1.25  # Stabilizing regime → can increase
        elif fdi_z < -1.0:
            exposure = 1.10
        else:
            exposure = 1.0  # Neutral
        
        # Regime-specific adjustments
        if regime == 'reflexive_crash':
            # High persistence (80.6%) → be very defensive
            exposure = min(exposure, 0.25)
        elif regime == 'absorbed_shock':
            # If FDI is also rising, this is the DANGER ZONE (35.6% crash probability)
            if fdi_z > 0.5:
                exposure = min(exposure, 0.30)
            else:
                exposure = min(exposure, 0.60)
        elif regime == 'hidden_instability':
            # Less dangerous than expected (75.7% self-loops)
            exposure = min(exposure, 0.70)
        
        # NBFC early warning override (9-10 day lead)
        if nbfc_warning:
            exposure = min(exposure, 0.60)
        
        return exposure
    
    def get_sector_tilt(self, date) -> dict:
        """
        Get sector tilts based on sectoral FDI.
        Returns dict of sector -> weight multiplier.
        
        When financial sector (NBFCs/Banks) FDI is elevated:
        - Reduce: Banks, NBFCs, Metals (leading/cyclical)
        - Increase: IT, Pharma, FMCG (defensive/lagging)
        """
        if not self.is_loaded or self.sectoral_fdi is None:
            return {}
        
        metrics = self.get_fdi_metrics(date)
        if metrics is None:
            return {}
        
        tilts = {
            'Banks': 1.0,
            'NBFCs': 1.0,
            'IT': 1.0,
            'Pharma': 1.0,
            'FMCG': 1.0,
            'Metals': 1.0,
            'Auto': 1.0,
            'Infrastructure': 1.0,
            'Power_Utilities': 1.0,
        }
        
        # Check if financials are stressed
        nbfc_fdi = metrics.get('nbfc_fdi', 0)
        banks_fdi = metrics.get('banks_fdi', 0)
        
        if nbfc_fdi > 1.5 or banks_fdi > 1.5:
            # Rotate OUT of financials and cyclicals
            tilts['Banks'] = 0.50
            tilts['NBFCs'] = 0.50
            tilts['Metals'] = 0.70
            tilts['Auto'] = 0.80
            
            # Rotate INTO defensives
            tilts['IT'] = 1.25
            tilts['Pharma'] = 1.25
            tilts['FMCG'] = 1.25
        
        return tilts
    
    def get_diagnostics(self) -> dict:
        """Return diagnostic information."""
        if not self.is_loaded:
            return {"error": "FDI data not loaded"}
        
        return {
            "fdi_data_shape": self.fdi_data.shape if self.fdi_data is not None else None,
            "sectoral_fdi_shape": self.sectoral_fdi.shape if self.sectoral_fdi is not None else None,
            "date_range": (
                str(self.fdi_data.index.min().date()) if self.fdi_data is not None else None,
                str(self.fdi_data.index.max().date()) if self.fdi_data is not None else None,
            )
        }

class BacktestEngine:
    def __init__(self, config: StrategyConfig, dm: DataManager, um: UniverseManager):
        self.cfg = config
        self.dm = dm
        self.um = um
        self.calc = Calculator(config)
        self.atr_mgr = ATRManager(config)
        
        self.results = []
        self.holdings = {}
        self.entry_info = {} # ticker: (price, date, regime)
        self.cash = 100_000.0
        self.initial_capital = 100_000.0
        self.peak_equity = 100_000.0
        self.allowed_exposure = 1.0
        
        # Crowding Vars
        self.crowding_history = [] # Stores (date, rel_perf_36m)
        self.is_crowded_exit = False
        
        # Crash Predictor (initialized later after first pass)
        # Crash Predictor (LightGBM)
        self.ml_filter = None
        self._ml_filter_active = False
        
        # RF Regime Filter
        self.rf_filter = None 
        self._rf_filter_active = False
        
        # LGBM Regime Filter
        self.lgbm_regime_filter = None
        self._lgbm_regime_filter_active = False
        
        # FDI Regime Filter (from volatility_diagnostic.py research)
        self.fdi_filter = None
        if self.cfg.enable_fdi_filter:
            self.fdi_filter = FDIRegimeFilter(
                config, 
                fdi_data_path=self.cfg.fdi_data_path,
                sectoral_path=self.cfg.sectoral_fdi_path
            )
        
    def run(self):
        print("Starting Backtest...")
        
        # Timeline
        if self.dm.index_data is None or self.dm.index_data.empty:
            print("CRITICAL ERROR: No Index Data found (Fetch Failed). Aborting run.")
            return pd.DataFrame() # Return empty DF
            
        dates = self.dm.index_data.loc[self.cfg.start_date:self.cfg.end_date].index
        
        # 0. Warm-up Crowding History (if enabled)
        if self.cfg.enable_crowding_exit:
            self._warm_up_crowding_history(dates[0])
            
        rebalance_dates = []

        

        # Crash Predictor: Initialize but dont train yet

        if self.cfg.enable_ml_filter and ML_AVAILABLE:
            self.ml_filter = CrashPredictor(self.cfg, self.dm, self.um)
            ml_train_idx = int(len(dates) * self.cfg.ml_train_pct)
            print(f"Crash Predictor (LightGBM) will train after {dates[ml_train_idx].date()}")
        else:
            ml_train_idx = len(dates) + 1  # Never train
            
        # RF Filter Init (Legacy - Disabled by config usually)
        # RF Filter Init (Legacy - Removed)
        # if self.cfg.enable_rf_filter and ML_AVAILABLE:
        #      pass 
             
        # LightGBM Regime Filter Init
        if self.cfg.enable_lgbm_regime_filter and ML_AVAILABLE:
            self.lgbm_regime_filter = LightGBMRegimeFilter(self.cfg, self.dm, self.um)
            lgbm_train_idx = int(len(dates) * self.cfg.lgbm_train_pct)
            print(f"LGBM Regime Filter will train after {dates[lgbm_train_idx].date()}")
        else:
            lgbm_train_idx = len(dates) + 1


        

        curr = dates[0]
        while curr <= dates[-1]:
            rebalance_dates.append(curr)
            curr += timedelta(days=self.cfg.rebalance_freq)
            # Find next valid trading day
            next_valid = dates[dates >= curr]
            if len(next_valid) > 0: curr = next_valid[0]
            else: break
            
        ema200 = self.dm.index_data.ewm(span=200).mean()

        for i, date in enumerate(dates):
            # Check if we should train ML filter now
            if i == ml_train_idx and self.ml_filter is not None and not self._ml_filter_active:
                print(f"\n=== Training Crash Predictor at {date.date()} ===")
                # Build equity curve from results so far
                if self.results:
                    equity_df = pd.DataFrame(self.results).set_index("date")
                    equity_series = equity_df["value"]
                    # Train ML filter
                    self.ml_filter.train(equity_series, dates[:i])
                    if self.ml_filter.is_trained:
                        self._ml_filter_active = True
                        print("Crash Predictor activated for remaining backtest.\n")
                    else:
                        print("Crash Predictor training failed. Continuing without.\n")

            # Check for RF Filter Training
            # Check for LightGBM Filter Training
            if i == lgbm_train_idx and self.lgbm_regime_filter is not None and not self._lgbm_regime_filter_active:
                print(f"\n=== Training LightGBM Regime Filter at {date.date()} ===")
                if self.results:
                    equity_df = pd.DataFrame(self.results).set_index("date")
                    equity_series = equity_df["value"]
                    self.lgbm_regime_filter.train(equity_series, dates[:i])
                    if self.lgbm_regime_filter.is_trained:
                        self._lgbm_regime_filter_active = True
                        print("LGBM Filter activated for remaining backtest.\n")
                    else:
                        print("LGBM Filter training failed.\n")

            
            # 1. Check Stops (Using Previous Close Data)
            self._check_stops(date)
            
            # 2. Drawdown Brake & Crowding Exit (Daily Check)
            self._apply_risk_overlays(date)
            
            # 3. Rebalance
            if date in rebalance_dates:
                # Regime Check using T-1
                if date in ema200.index:
                    # Look strictly at the day before "date"
                    past_idx = self.dm.index_data[self.dm.index_data.index < date]
                    past_ema = ema200[ema200.index < date]
                    
                    if not past_idx.empty and not past_ema.empty:
                        # Compare last available close with last available EMA
                        if past_idx.iloc[-1] > past_ema.iloc[-1]: regime = "BULL"
                        else: regime = "BEAR"
                    else: regime = "NEUTRAL" # Init
                
                self._rebalance(date, regime)
            
            # 4. Record Value
            val = self.cash + sum(self.holdings[t] * self.dm.get_price(t).loc[date, "Close"] 
                                  for t in self.holdings if date in self.dm.get_price(t).index)
            self.results.append({"date": date, "value": val})
            if val > self.peak_equity: self.peak_equity = val
            
            # Update Crowding History Daily (for signal continuity)
            if self.cfg.enable_crowding_exit:
                self._update_crowding_metric(date, val)

        print("Backtest Complete.")
        return pd.DataFrame(self.results).set_index("date")

    def _apply_risk_overlays(self, date):
        # 1. Reset
        self.allowed_exposure = 1.0
        
        # 2. Crowding Check (Priority High -> Force 0% if crowded)
        if self.cfg.enable_crowding_exit:
            if self._check_is_crowded(date):
                self.allowed_exposure = 0.0
                self.is_crowded_exit = True
                # print(f"[{date.date()}] CROWDED EXIT TRIGGERED!")
            else:
                self.is_crowded_exit = False
        
        # 3. Crash Predictor Exposure (after regime check, before DD brake)
        # 3. Crash Predictor Exposure (after regime check, before DD brake)
        if self.cfg.enable_ml_filter and self._ml_filter_active and self.ml_filter is not None:
            ml_exposure = self.ml_filter.get_exposure(date)
            self.allowed_exposure = min(self.allowed_exposure, ml_exposure)
            
        # 3a. LGBM Regime Filter Exposure
        if self.cfg.enable_lgbm_regime_filter and self._lgbm_regime_filter_active and self.lgbm_regime_filter is not None:
            lgbm_exposure = self.lgbm_regime_filter.get_exposure(date)
            # Apply IF safer than current
            self.allowed_exposure = min(self.allowed_exposure, lgbm_exposure)

        # 3b. FDI Regime Filter Exposure (Research Integrated)
        if self.cfg.enable_fdi_filter and self.fdi_filter is not None:
            fdi_exposure = self.fdi_filter.get_exposure(date)
            self.allowed_exposure = min(self.allowed_exposure, fdi_exposure)
        
        # 3c. Extra safety: Check current portfolio drawdown and cut if needed
        if self.results and len(self.results) > 0:
            curr_val = self.results[-1]["value"]
            curr_dd = (curr_val / self.peak_equity) - 1.0
            if curr_dd < -0.16:
                # Portfolio in 16%+ drawdown - reduce significantly
                self.allowed_exposure = min(self.allowed_exposure, 0.30)
            elif curr_dd < -0.12:
                # Portfolio in 12%+ drawdown - get defensive
                self.allowed_exposure = min(self.allowed_exposure, 0.50)
        
        # 4. DD Brake (Only if not already 0 from crowding or ML)
        if self.cfg.enable_dd_brake and self.allowed_exposure > 0:
            self._apply_drawdown_brake_logic(date)
            
        # 5. Execute Reduction if needed
        self._enforce_exposure_limit(date)

    def _warm_up_crowding_history(self, start_date):
        print("Warming up Crowding History (2008-2013)...")
        # Simulates a simplified monthly momentum portfolio to build history
        # We need historical Strat vs Bench performance
        
        # Get historical dates prior to start_date
        all_dates = self.dm.index_data.index
        warmup_dates = all_dates[all_dates < start_date]
        if warmup_dates.empty: return
        
        # We need at least 3 years
        # Logic: Rebalance monthly, hold top 10 equal weight. Track INDEX vs STRAT.
        
        # Simplified simulation
        hist_cash = 100.0
        hist_holdings = {}
        hist_vals = []
        
        # We'll sample monthly to save time
        monthly_dates = []
        curr = warmup_dates[0]
        while curr < warmup_dates[-1]:
            if curr in warmup_dates: monthly_dates.append(curr)
            curr += timedelta(days=30)
            
        for d in monthly_dates:
            # Calc Val
            curr_val = hist_cash + sum(hist_holdings[t] * self.dm.get_price(t).loc[d, "Close"] 
                                       for t in hist_holdings if d in self.dm.get_price(t).index)
            hist_vals.append({"date": d, "val": curr_val})
            
            # Rebalance Logic (Simplified - Top 10 by 12m, Equal Weight)
            univ = self.um.get_universe(d)
            if not univ: continue
            
            scores = []
            for t in univ:
                df = self.dm.get_price(t)
                if df is None: continue
                hist = df[df.index < d]
                if len(hist) < 260: continue
                
                # Fast 12m calc
                closes = hist["Close"]
                ret12 = (closes.iloc[-1] / closes.iloc[-252]) - 1.0 if len(closes) >= 252 else -99
                scores.append((t, ret12))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            top_n = [x[0] for x in scores[:10]]
            
            if not top_n: continue
            
            # Sell all
            hist_cash = curr_val
            hist_holdings = {}
            w = 1.0 / len(top_n)
            for t in top_n:
                df = self.dm.get_price(t)
                if d in df.index:
                    p = df.loc[d, "Close"] # Approx exec at Close
                    shares = (curr_val * w) / p
                    hist_holdings[t] = shares
                    hist_cash -= (shares * p)
        
        # Now Compute 36m Rolling Diff
        tdf = pd.DataFrame(hist_vals).set_index("date")
        
        # Align with Benchmark
        b = self.dm.benchmark_data.reindex(tdf.index, method='ffill')
        
        # Calculate monthly returns then rolling 36m cumulative
        # Actually easier: (Val_t / Val_t-36m) - 1
        
        # We just store the history of RELATIVE 36m Cumulative Returns
        # But we need it daily? No, monthly is fine for distribution.
        
        # Compute for each month in history
        for i in range(len(tdf)):
            if i < 36: continue
            date = tdf.index[i]
            
            # Strat 36m
            s_now = tdf.iloc[i]["val"]
            s_old = tdf.iloc[i-36]["val"]
            s_ret = (s_now/s_old) - 1.0
            
            # Bench 36m
            b_now = b.loc[date]
            b_old = b.iloc[i-36]
            b_ret = (b_now/b_old) - 1.0
            
            rel_perf = s_ret - b_ret
            self.crowding_history.append(rel_perf)
            
        print(f"Warm-up Complete. History Points: {len(self.crowding_history)}")

    def _update_crowding_metric(self, date, curr_strat_val):
        # We need exactly 36m ago. 
        # For simplicity, we can use the 'results' list if long enough, else skip
        # Or simpler: Just append current Metric to history if valid?
        
        # To compute current 36m metric:
        # We need Val t-36m.
        
        lookback_days = 365 * self.cfg.crowding_lookback_years
        target_date = date - timedelta(days=lookback_days)
        
        # Find closest date in results
        # This is expensive to search daily.
        # Optimization: Only calculate periodically or map by date
        # But since we run daily loop, we can just peek back approx index
        
        # Approximation: 252 * 3 = 756 trading days
        if len(self.results) < 756: return
        
        # However, at start of backtest, 'results' is empty/short.
        # But we want to use the Strat 36m return.
        # Since we just started, we DON'T have a 36m return for the REAL strategy yet.
        # We only have it from the Warm-up proxy.
        # PAPER says: "monitor... past performance". 
        # Ideally we stitch the warm-up equity curve with live curve.
        # Or we rely on Warm-up history to provide the distribution, but we need the CURRENT metric.
        
        # IMPT: We cannot compute 36m return of the strategy on Day 1 of backtest.
        # So we must use the Warm-Up proxy's latest values as the "Current Momentum Performance" 
        # until the live strategy builds its own history?
        # Actually, usually 'Momentum Factor' is continuous. 
        # For this implementation, we will skip updating the METRIC until we have live history,
        # OR we need to accept that we can't check crowding for first 3 years of live backtest?
        # That's bad.
        
        # Better: continue the 'warm up' simulation in background? No.
        
        # Hack: The 'Momentum Score' itself is a proxy.
        # But paper specifies 'Accumulated return of the strategy'.
        # We will assume safely that if we don't have 36m history, we are not crowded?
        # Or use the last value from warm-up?
        pass

    def _check_is_crowded(self, date):
        # We need Current 36m Rel Perf.
        # We need History of 36m Rel Perf.
        
        # 1. Get Current 36m Rel Perf
        # Issue: See above. We need 'past returns of the momentum portfolio'.
        # Solution: Use the Universe Manager or Mock Portfolio to calc ' Theoretical Momentum Index'
        # OR just use Nifty 100 as proxy? No.
        
        # Compromise: We use the 'warm up' logic to get a 'Theoretical Momentum Index' that spans 
        # from 2008 to Today. This 'Index' runs in background or is pre-calculated.
        # For now, let's rely on 'self.results' if available, otherwise return False (Safe).
        
        if len(self.results) < 756: return False
        
        old_idx = -756
        s_now = self.results[-1]["value"]
        s_old = self.results[old_idx]["value"]
        s_ret = (s_now/s_old) - 1.0
        
        # Use asof for safer lookup
        b_now = self.dm.benchmark_data.asof(date)
        if pd.isna(b_now): return False # Should not happen if data aligned, but safety first
        
        b_old_date = self.results[old_idx]["date"]
        b_old = self.dm.benchmark_data.asof(b_old_date) # safe lookup
        
        if pd.isna(b_old) or b_old == 0: return False
        b_ret = (b_now/b_old) - 1.0
        
        curr_metric = s_ret - b_ret
        
        # 2. Compare to History
        # We need to maintain the distribution.
        # Add current to history temporarily for ranking?
        # Or just rank current vs all past.
        
        # Percentile
        # count how many historic values are < curr_metric
        n_below = sum(1 for x in self.crowding_history if x < curr_metric)
        total = len(self.crowding_history)
        if total < 10: return False
        
        pct = n_below / total
        
        # Update history for next time (rolling forward)
        # We should add it sparingly (e.g. monthly) to avoid skewing with autocorrelation
        # For now, we don't add strictly daily to 'history distribution' to keep it stable
        # or we add it. Let's not add it daily. The paper uses 'growing window'.
        # We'll stick to static warm-up history + slowly growing? 
        # Let's just use warm-up history + static for stability in this simple v1.
        
        return pct > self.cfg.crowding_exit_percentile

    def _apply_drawdown_brake_logic(self, date):
        # ... [Existing Logic moved here] ...
        # (Same content as previous _apply_drawdown_brake, but using self.allowed_exposure min())
        # Calc DD
        if not self.results: return
        last_val = self.results[-1]["value"]
        dd = (last_val / self.peak_equity) - 1.0
        
        brake_exp = 1.0
        for thresh, exp_limit in zip(self.cfg.dd_brake_thresholds, self.cfg.dd_brake_exposures):
            if dd < -thresh:
                brake_exp = min(brake_exp, exp_limit)
        
        self.allowed_exposure = min(self.allowed_exposure, brake_exp)

    def _enforce_exposure_limit(self, date):
        # Reduce positions if Exposure > Allowed
        current_val = self.cash
        # Calc holdings val
        h_vals = {}
        for t, shares in self.holdings.items():
            df = self.dm.get_price(t)
            # Use prev close
            hist = df[df.index < date]
            if not hist.empty:
               h_vals[t] = shares * hist.iloc[-1]["Close"]
        
        invested_val = sum(h_vals.values())
        total_equity = self.cash + invested_val
        if total_equity == 0: return
        
        curr_exp = invested_val / total_equity
        
        if curr_exp > self.allowed_exposure + 0.02: # Buffer
             # Reduce logic (same as before)
             target_invested = total_equity * self.allowed_exposure
             to_sell_val = invested_val - target_invested
             
             if to_sell_val > 0:
                 # Pro-rata sell
                 for t, val in h_vals.items():
                     proportion = val / invested_val
                     sell_amt = to_sell_val * proportion
                     
                     df = self.dm.get_price(t)
                     if date in df.index:
                         p = df.loc[date, "Open"]
                         shares = int(sell_amt / p)
                         if shares > 0:
                             if shares > self.holdings[t]: shares = self.holdings[t]
                             self.holdings[t] -= shares
                             
                             proc = shares * p * (1 - self.cfg.transaction_cost - self.cfg.slippage)
                             self.cash += proc
                             
                             if self.holdings[t] <= 0:
                                 del self.holdings[t]
                                 if t in self.entry_info: del self.entry_info[t]

    def _check_stops(self, date):
        # Uses T-1 data for check, executes at T Open
        to_sell = []
        for t, shares in self.holdings.items():
            df = self.dm.get_price(t)
            # Need history BEFORE today
            hist = df[df.index < date]
            if hist.empty: continue
            
            curr_price = hist.iloc[-1]["Close"]
            entry_p, _, entry_reg = self.entry_info[t]
            
            sl_pct = self.atr_mgr.get_stop_loss_pct(hist, entry_reg)
            ret = (curr_price / entry_p) - 1.0
            
            if ret < sl_pct:
                if date in df.index:
                    exit_p = df.loc[date, "Open"] # Execute at Open
                    proceeds = shares * exit_p * (1 - self.cfg.transaction_cost - self.cfg.slippage)
                    self.cash += proceeds
                    to_sell.append(t)
        
        for t in to_sell:
            del self.holdings[t]
            del self.entry_info[t]

    def _rebalance(self, date, regime):
        # Universe
        univ = self.um.get_universe(date)
        if not univ: return

        # Scoring
        scores = []
        for t in univ:
            df = self.dm.get_price(t)
            if df is None: continue
            hist = df[df.index < date] # Strictly past data
            if len(hist) < 260: continue
            
            m12, m3 = self.calc.momentum_score(hist["Close"])
            if np.isnan(m12): continue
            
            if not self.calc.trend_ok(hist["Close"]): continue
            
            # Nearhigh Filter: Skip stocks far from their 52-week highs
            # These stocks are prone to causing momentum crashes during market rebounds
            nearhigh = self.calc.nearhigh_ratio(hist["Close"])
            
            if self.cfg.enable_nearhigh_filter:
                if np.isnan(nearhigh) or nearhigh < self.cfg.nearhigh_min_threshold:
                    continue
            
            scores.append({
                "ticker": t, 
                "m12": m12, 
                "m3": m3,
                "nearhigh": nearhigh if not np.isnan(nearhigh) else 0.5,  # Default to 0.5 if no data
                "sector": self.dm.get_sector(t)
            })
            
        if not scores: return
        sdf = pd.DataFrame(scores)
        
        # Calculate composite score with optional Nearhigh weighting
        if self.cfg.use_nearhigh_scoring and self.cfg.nearhigh_weight > 0:
            # Normalize weights to sum to 1.0
            total_mom_weight = self.cfg.weight_12m + self.cfg.weight_3m
            mom12_w = self.cfg.weight_12m / total_mom_weight * (1 - self.cfg.nearhigh_weight)
            mom3_w = self.cfg.weight_3m / total_mom_weight * (1 - self.cfg.nearhigh_weight)
            nh_w = self.cfg.nearhigh_weight
            
            sdf["score"] = (sdf["m12"].rank(pct=True) * mom12_w + 
                          sdf["m3"].rank(pct=True) * mom3_w +
                          sdf["nearhigh"].rank(pct=True) * nh_w)
        else:
            sdf["score"] = (sdf["m12"].rank(pct=True) * self.cfg.weight_12m + 
                          sdf["m3"].rank(pct=True) * self.cfg.weight_3m)
        
        sdf = sdf.sort_values("score", ascending=False)
        
        # Selection & Holding Buffer
        current_tickers = list(self.holdings.keys())
        target_n = self.cfg.portfolio_size
        
        # Select Top N with buffer preference
        selected = []
        # Keep existing if rank is high enough (e.g. top N+5)
        for t in current_tickers:
            rank_row = sdf[sdf["ticker"] == t]
            if not rank_row.empty:
                # Approximate Rank Check (simplified)
                if rank_row.index[0] < target_n + 5: # If in top N+5
                    selected.append(t)
        
        # Fill rest
        for _, row in sdf.iterrows():
            if len(selected) >= target_n: break
            if row["ticker"] not in selected:
                selected.append(row["ticker"])
                
        # Sector Limits
        final_sel = []
        sector_counts = {}
        for t in selected:
            sec = self.dm.get_sector(t)
            w = 1.0/target_n # Approx weight check
            curr_sec_w = sector_counts.get(sec, 0.0)
            if self.cfg.enable_sector_caps and (curr_sec_w + w > self.cfg.sector_cap_limit + 0.05):
                continue
            final_sel.append(t)
            sector_counts[sec] = curr_sec_w + w
            if len(final_sel) >= target_n: break
            
        if not final_sel: return

        # Allocations (HRP or Schur)
        if self.cfg.use_schur:
            # Schur Complementary Allocation (arxiv 2411.05807)
            alloc_data = {}
            for t in final_sel:
                hist = self.dm.get_price(t)
                hist = hist.loc[hist.index < date].iloc[-self.cfg.hrp_lookback:]["Close"]
                alloc_data[t] = hist
            if alloc_data:
                weights = SchurAllocator.get_weights(
                    pd.DataFrame(alloc_data).dropna(), 
                    gamma=self.cfg.schur_gamma
                )
            else:
                weights = {t: 1.0/len(final_sel) for t in final_sel}
        elif self.cfg.use_hrp:
            hrp_data = {}
            for t in final_sel:
                # Use data strictly BEFORE rebalance date
                hist = self.dm.get_price(t)
                hist = hist.loc[hist.index < date].iloc[-self.cfg.hrp_lookback:]["Close"]
                hrp_data[t] = hist
            if hrp_data:
                weights = HRPAllocator.get_weights(pd.DataFrame(hrp_data).dropna())
            else:
                weights = {t: 1.0/len(final_sel) for t in final_sel}
        else:
            weights = {t: 1.0/len(final_sel) for t in final_sel}

        # Apply Sector Tilts (FDI Research)
        if self.cfg.enable_fdi_filter and self.cfg.enable_fdi_sector_tilts and self.fdi_filter is not None:
            tilts = self.fdi_filter.get_sector_tilt(date)
            if tilts:
                # Mapping Yahoo Sectors to FDI Sectors
                sec_map = {
                    "Financial Services": "Banks", # Proxy
                    "Technology": "IT",
                    "Healthcare": "Pharma",
                    "Consumer Defensive": "FMCG",
                    "Basic Materials": "Metals",
                    "Consumer Cyclical": "Auto",
                    "Utilities": "Power_Utilities",
                    "Industrials": "Infrastructure",
                    "Energy": "Power_Utilities"
                }
                
                for t in weights:
                    y_sec = self.dm.get_sector(t)
                    fdi_sec = sec_map.get(y_sec, "Unknown")
                    multiplier = tilts.get(fdi_sec, 1.0)
                    
                    # If Financial Services, check if we can distinguish NBFCs
                    # (Simple heuristic based on name if needed, or just apply Banks tilt)
                    if y_sec == "Financial Services":
                        # If Banks are stressed (tilt < 1), apply to all Fin Services
                        # If NBFCs are stressed (tilt < 1), apply to all Fin Services
                        # This is conservative (de-risk financials if either key sub-sector is stressed)
                        bank_tilt = tilts.get('Banks', 1.0)
                        nbfc_tilt = tilts.get('NBFCs', 1.0)
                        multiplier = min(bank_tilt, nbfc_tilt)
                    
                    weights[t] *= multiplier
                
                # Renormalize to 1.0
                total_w = sum(weights.values())
                if total_w > 0:
                    for t in weights: weights[t] /= total_w

        # Apply Exposure Limit (Drawdown Brake)
        # Scale weights so they sum to self.allowed_exposure instead of 1.0
        # If allowed_exposure is 0.7, sum(weights) should be 0.7
        current_weight_sum = sum(weights.values())
        if current_weight_sum > 0:
            scale_factor = self.allowed_exposure / current_weight_sum
            for t in weights:
                weights[t] *= scale_factor
        
        # Execute
        # Sell unwanted
        total_val = self.results[-1]["value"] if self.results else self.cash
        
        for t in list(self.holdings.keys()):
            if t not in weights:
                df = self.dm.get_price(t)
                if date in df.index:
                    p = df.loc[date, "Open"]
                    proc = self.holdings[t] * p * (1 - self.cfg.transaction_cost - self.cfg.slippage)
                    self.cash += proc
                    del self.holdings[t]
                    del self.entry_info[t]

        # Rebalance Existing & Buy New
        for t, w in weights.items():
            df = self.dm.get_price(t)
            if df is None or date not in df.index: continue
            
            p = df.loc[date, "Open"]
            target_amt = total_val * w
            curr_amt = self.holdings.get(t, 0) * p
            
            diff = target_amt - curr_amt
            if diff > p: # Buy
                shares = int(diff / p)
                cost = shares * p * (1 + self.cfg.transaction_cost + self.cfg.slippage)
                if self.cash >= cost:
                    self.cash -= cost
                    self.holdings[t] = self.holdings.get(t, 0) + shares
                    # Update entry info (weighted avg for price? Keep simple: reset if new buy)
                    # For ATR stops, usually track last entry or avg. 
                    # Here we update entry price/regime on rebalance buy
                    self.entry_info[t] = (p, date, regime)
            
            elif diff < -p: # Sell trim
                shares = int(abs(diff) / p)
                if shares > self.holdings.get(t, 0): shares = self.holdings.get(t, 0)
                proc = shares * p * (1 - self.cfg.transaction_cost - self.cfg.slippage)
                self.cash += proc
                self.holdings[t] -= shares
                if self.holdings[t] <= 0:
                    del self.holdings[t]
                    del self.entry_info[t]


# ==================== ANALYSIS & PLOTTING ====================

def generate_output(equity_df, dm, config, bt=None):
    print("Generating Analysis...")
    strat = equity_df["value"]
    strat_ret = strat.pct_change().fillna(0)
    
    # Bench
    bench = dm.benchmark_data
    bench = bench.reindex(equity_df.index).ffill().bfill()  # bfill handles start NaNs
    bench_ret = bench.pct_change().fillna(0)
    
    # Metrics
    def calc_metrics(ser, name):
        if ser.empty: return {}
        ret = ser.pct_change().fillna(0)
        
        # 1. Total Return & CAGR
        total = (ser.iloc[-1] / ser.iloc[0]) - 1
        days = (ser.index[-1] - ser.index[0]).days
        years = days / 365.25
        cagr = (ser.iloc[-1] / ser.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # 2. Volatility
        ann_vol = ret.std() * np.sqrt(252)
        
        # 3. Sharpe Ratio (Rf=0)
        sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
        
        # 4. Sortino Ratio (Rf=0, Target=0)
        downside = ret[ret < 0]
        downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(252)
        sortino = (ret.mean() * 252) / downside_dev if downside_dev > 0 else 0
        
        # 5. Drawdown & Calmar
        dd = ser / ser.cummax() - 1
        mdd = dd.min()
        calmar = cagr / abs(mdd) if mdd != 0 else 0
        
        # 6. Win Rate
        non_zero_ret = ret[ret != 0]
        win_rate = (non_zero_ret > 0).sum() / len(non_zero_ret) if len(non_zero_ret) > 0 else 0
        
        return {
            "Total Return": f"{total*100:.1f}%",
            "CAGR": f"{cagr*100:.1f}%",
            "Ann. Volatility": f"{ann_vol*100:.1f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Calmar Ratio": f"{calmar:.2f}",
            "Max Drawdown": f"{mdd*100:.1f}%",
            "Win Rate": f"{win_rate*100:.1f}%"
        }

    m_strat = calc_metrics(strat, "Strategy")
    m_bench = calc_metrics(bench, "Nifty 100")
    
    metrics_df = pd.DataFrame([m_strat, m_bench], index=["Strategy", "Benchmark"])
    print("\nPerformance Metrics:")
    print(metrics_df)
    
    # Plots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 1. Cum Returns
    ax1 = fig.add_subplot(gs[0, :])
    (strat / strat.iloc[0]).plot(ax=ax1, label="Strategy", color="blue", linewidth=2)
    (bench / bench.iloc[0]).plot(ax=ax1, label="Nifty 100", color="gray", alpha=0.7)
    ax1.set_title("Cumulative Returns")
    ax1.legend()
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    dd = strat / strat.cummax() - 1
    dd.plot(ax=ax2, color="red", alpha=0.6, fillstyle="bottom")
    ax2.fill_between(dd.index, dd, 0, color="red", alpha=0.3)
    ax2.set_title("Drawdown")
    
    # 3. Monthly Heatmap
    ax3 = fig.add_subplot(gs[2, :])
    m_ret = strat.resample("ME").last().pct_change()
    hm_data = pd.DataFrame({"Year": m_ret.index.year, "Month": m_ret.index.month, "Ret": m_ret.values})
    hm_pivot = hm_data.pivot(index="Year", columns="Month", values="Ret")
    sns.heatmap(hm_pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax3, cbar=False)
    ax3.set_title("Monthly Returns (%)")
    
    plt.tight_layout()
    plt.savefig("modular_strategy_report.png")
    print("\nSaved plot to 'modular_strategy_report.png'")
    
    # Crash Predictor Diagnostics
    if bt is not None and hasattr(bt, 'ml_filter') and bt.ml_filter is not None and bt.ml_filter.is_trained:
        print("\n=== Crash Predictor Diagnostics ===")
        diag = bt.ml_filter.get_diagnostics()
        
        # Feature Importance
        if diag.get("feature_importances"):
            print("\nFeature Importances:")
            sorted_imp = sorted(diag["feature_importances"].items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp:
                print(f"  {feat}: {imp}")
            
            # Feature importance plot
            fig2, ax = plt.subplots(figsize=(10, 6))
            feats = [x[0] for x in sorted_imp]
            imps = [x[1] for x in sorted_imp]
            ax.barh(feats, imps, color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title('Crash Predictor Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig("ml_filter_feature_importance.png")
            print("Saved feature importance plot to 'ml_filter_feature_importance.png'")
        
        # Test predictions analysis
        if diag.get("test_predictions") and ML_AVAILABLE:
            preds = diag["test_predictions"]
            if len(preds) > 0:
                y_true = [p[1] for p in preds]
                y_prob = [p[2] for p in preds]
                y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                print(f"\nConfusion Matrix:")
                print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
                print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
                
                # ROC Curve
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auc = roc_auc_score(y_true, y_prob)
                    
                    fig3, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Crash Predictor ROC Curve')
                    ax.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig("ml_filter_roc_curve.png")
                    print(f"Saved ROC curve to 'ml_filter_roc_curve.png'")
                except Exception as e:
                    print(f"Could not generate ROC curve: {e}")
                    
    # === LGBM Regime Filter Diagnostics ===
    if bt is not None and hasattr(bt, 'lgbm_regime_filter') and bt.lgbm_regime_filter is not None and bt.lgbm_regime_filter.is_trained:
        print("\n=== LGBM Regime Filter Diagnostics ===")
        diag = bt.lgbm_regime_filter.get_diagnostics()
        
        # Feature Importance
        if diag.get("feature_importances"):
            print("\nLGBM Feature Importances:")
            sorted_imp = sorted(diag["feature_importances"].items(), key=lambda x: x[1], reverse=True)
            for feat, imp in sorted_imp:
                print(f"  {feat}: {imp:.4f}")
            
            # Feature importance plot
            fig3, ax = plt.subplots(figsize=(10, 6))
            feats = [x[0] for x in sorted_imp]
            imps = [x[1] for x in sorted_imp]
            ax.barh(feats, imps, color='purple')
            ax.set_xlabel('Importance')
            ax.set_title('LGBM Regime Filter Feature Importance')
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig("lgbm_feature_importance.png")
            print("Saved lgbm_feature_importance.png")
            
        # Test predictions analysis
        if diag.get("test_predictions"):
            preds = diag["test_predictions"]
            if len(preds) > 0:
                y_true = np.array([p[1] for p in preds])
                y_prob = np.array([p[2] for p in preds])
                y_pred = (y_prob > 0.5).astype(int)
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                print(f"\nLGBM Confusion Matrix:")
                print(cm)
                
                # Signal Time Series
                dates_test = [p[0] for p in preds]
                exp_series = [bt.lgbm_regime_filter.get_exposure(d) for d in dates_test]
                
                fig4, ax = plt.subplots(figsize=(12, 4))
                ax.plot(dates_test, exp_series, label="LGBM Exposure", color="purple")
                ax.set_title("LGBM Regime Filter Exposure Signal (Test Set)")
                ax.set_ylabel("Exposure (0-1)")
                plt.tight_layout()
                plt.savefig("lgbm_exposure_timeseries.png")
                print("Saved lgbm_exposure_timeseries.png")

                # ROC Curve
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_prob)
                    auc = roc_auc_score(y_true, y_prob)
                    
                    fig5, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('LGBM Regime Filter ROC Curve')
                    ax.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig("lgbm_roc_curve.png")
                    print(f"Saved lgbm_roc_curve.png")
                except Exception as e:
                    print(f"Could not generate ROC curve: {e}")
                    ax.plot([0,1],[0,1], 'k--')
                    ax.set_title('RF Filter ROC')
                    ax.legend()
                    plt.savefig("rf_roc_curve.png")
                    print("Saved rf_roc_curve.png")
                except: pass
                
                # Plot Exposure Time Series
                dates = [p[0] for p in preds]
                # Re-calc exposure logic roughly for visualization
                # This is an approximation as actual exposure involves min/max caps
                exposures = [max(0.25, min(p, 1.0)) for p in y_prob] 
                
                fig6, ax = plt.subplots(figsize=(12, 4))
                ax.plot(dates, exposures, color='orange', label='RF Exposure')
                ax.set_title('RF Exposure Signal (OOS)')
                ax.set_ylabel('Allowed Exposure')
                plt.tight_layout()
                plt.savefig("rf_exposure_timeseries.png")
                print("Saved rf_exposure_timeseries.png")

# ==================== MAIN ====================

def main():
    cfg = StrategyConfig()
    
    # Optional: Override with 'found' parameters if desired
    # cfg.weight_12m = ...
    # cfg.rebalance_freq = ...
    
    um = UniverseManager(cfg)
    dm = DataManager(cfg, um)
    dm.fetch_data()
    
    if not dm.price_data:
        print("No data fetched. Check internet or tickers.")
        return

    bt = BacktestEngine(cfg, dm, um)
    res = bt.run()
    
    if res is not None and not res.empty:
        generate_output(res, dm, cfg, bt)

if __name__ == "__main__":
    main()
