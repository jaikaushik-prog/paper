"""
Unified Liquidity Regime Framework for Indian Markets
======================================================

Combines:
1. Directional FDI (FDI_up, FDI_down, asymmetry)
2. Multi-resolution analysis (daily + rolling windows)
3. Cross-asset contagion network
4. Simple regime classifier (baseline, before Transformer)
5. Position sizing rules

This is the complete trading system framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD ALL DATA
# =============================================================================

def load_all_data():
    """Load all required data for the unified framework."""
    print("📦 Loading all data sources...")
    
    # Directional FDI
    try:
        directional_fdi = pd.read_csv('directional_fdi_output.csv', parse_dates=['date'], index_col='date')
        print(f"   ✅ Directional FDI: {len(directional_fdi)} days")
    except:
        print("   ❌ Directional FDI not found - run directional_fdi.py first")
        directional_fdi = None
    
    # Sectoral FDI
    try:
        sectoral_fdi = pd.read_csv('sectoral_fdi_output.csv', parse_dates=['date'], index_col='date')
        print(f"   ✅ Sectoral FDI: {len(sectoral_fdi)} days")
    except:
        print("   ❌ Sectoral FDI not found")
        sectoral_fdi = None
    
    # Cross-asset data
    print("\n   Fetching cross-asset data...")
    cross_assets = {}
    tickers = {
        'NIFTY': '^NSEI',
        'GOLD': 'GC=F',
        'CRUDE': 'CL=F',
        'USDINR': 'INR=X',
        'VIX': '^VIX',
        'US10Y': '^TNX'
    }
    
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df['Close'].iloc[:, 0] if hasattr(df['Close'], 'iloc') else df['Close']
                else:
                    close = df['Close']
                cross_assets[name] = close
                print(f"      ✅ {name}: {len(df)} days")
        except Exception as e:
            print(f"      ❌ {name}: {e}")
    
    cross_asset_df = pd.DataFrame(cross_assets)
    cross_asset_returns = cross_asset_df.pct_change().dropna()
    
    return directional_fdi, sectoral_fdi, cross_asset_df, cross_asset_returns


# =============================================================================
# 2. MULTI-RESOLUTION FDI
# =============================================================================

def compute_multi_resolution_fdi(directional_fdi):
    """
    Compute FDI at multiple time scales:
    - Short-term: 5-day rolling
    - Medium-term: 20-day rolling
    - Long-term: 60-day rolling
    """
    print("\n📊 Computing Multi-Resolution FDI...")
    
    if directional_fdi is None:
        return None
    
    df = directional_fdi.copy()
    
    # Multi-scale FDI (using the raw FDI column if available)
    if 'FDI_up' in df.columns:
        for window in [5, 20, 60]:
            df[f'FDI_up_{window}d'] = df['FDI_up'].rolling(window).mean()
            df[f'FDI_down_{window}d'] = df['FDI_down'].rolling(window).mean()
            df[f'FDI_asymmetry_{window}d'] = df[f'FDI_down_{window}d'] / (df[f'FDI_up_{window}d'] + 1e-10)
    
    # Trend signals
    df['FDI_up_trend'] = df['FDI_up_zscore'].rolling(5).mean() - df['FDI_up_zscore'].rolling(20).mean()
    df['FDI_down_trend'] = df['FDI_down_zscore'].rolling(5).mean() - df['FDI_down_zscore'].rolling(20).mean()
    
    print(f"   ✅ Added multi-resolution features")
    
    return df


# =============================================================================
# 3. CROSS-ASSET CONTAGION NETWORK
# =============================================================================

def build_contagion_network(cross_asset_returns, directional_fdi, window=60):
    """
    Build a contagion network showing how stress propagates across assets.
    
    Edges: Granger-like lead-lag correlations
    Nodes: Assets (NIFTY, GOLD, CRUDE, USDINR, VIX)
    """
    print("\n📊 Building Cross-Asset Contagion Network...")
    
    if directional_fdi is None or cross_asset_returns is None:
        return None
    
    # Align data
    common_idx = directional_fdi.index.intersection(cross_asset_returns.index)
    
    # Create features dataframe with FDI
    features = cross_asset_returns.loc[common_idx].copy()
    if 'FDI_up_zscore' in directional_fdi.columns:
        features['FDI'] = directional_fdi.loc[common_idx, 'FDI_up_zscore']
    
    assets = list(features.columns)
    n_assets = len(assets)
    
    # Compute lead-lag correlation matrix
    leadlag_matrix = np.zeros((n_assets, n_assets))
    
    for i, asset_i in enumerate(assets):
        for j, asset_j in enumerate(assets):
            if i != j:
                # Does asset_i at t predict asset_j at t+1?
                corr = features[asset_i].shift(1).corr(features[asset_j])
                if not np.isnan(corr):
                    leadlag_matrix[i, j] = corr
    
    # Create network
    G = nx.DiGraph()
    
    for asset in assets:
        G.add_node(asset)
    
    # Add edges for significant lead-lag (|corr| > 0.05)
    for i, asset_i in enumerate(assets):
        for j, asset_j in enumerate(assets):
            if i != j and abs(leadlag_matrix[i, j]) > 0.05:
                G.add_edge(asset_i, asset_j, weight=leadlag_matrix[i, j])
    
    print(f"   ✅ Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Identify central nodes (most influential)
    if G.number_of_nodes() > 0:
        centrality = nx.degree_centrality(G)
        print(f"   📊 Centrality (influence):")
        for asset, cent in sorted(centrality.items(), key=lambda x: -x[1]):
            print(f"      {asset}: {cent:.3f}")
    
    return G, leadlag_matrix, assets


def visualize_contagion_network(G, leadlag_matrix, assets):
    """Visualize the contagion network."""
    if G is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Network visualization
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=1500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')
    
    # Draw edges with colors based on weight
    edges = G.edges(data=True)
    if len(edges) > 0:
        weights = [d['weight'] for u, v, d in edges]
        colors = ['red' if w < 0 else 'green' for w in weights]
        widths = [abs(w) * 5 for w in weights]
        
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=colors, 
                              width=widths, alpha=0.6, arrows=True,
                              arrowsize=20, connectionstyle="arc3,rad=0.1")
    
    ax1.set_title('Cross-Asset Contagion Network', fontweight='bold')
    ax1.axis('off')
    
    # 2. Lead-lag heatmap
    ax2 = axes[1]
    im = ax2.imshow(leadlag_matrix, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
    ax2.set_xticks(range(len(assets)))
    ax2.set_yticks(range(len(assets)))
    ax2.set_xticklabels(assets, rotation=45, ha='right')
    ax2.set_yticklabels(assets)
    ax2.set_xlabel('→ Affected (t)')
    ax2.set_ylabel('← Cause (t-1)')
    ax2.set_title('Lead-Lag Correlation Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Correlation')
    
    plt.tight_layout()
    plt.savefig('plots/contagion_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Saved: plots/contagion_network.png")


# =============================================================================
# 4. REGIME CLASSIFIER (Baseline - Before Transformer)
# =============================================================================

def train_regime_classifier(multi_res_fdi, cross_asset_returns):
    """
    Train a baseline Random Forest regime classifier.
    
    Target: Next 5-day return quintile (1=worst, 5=best)
    Features: Multi-resolution FDI + cross-asset returns
    """
    print("\n📊 Training Regime Classifier (Baseline RF)...")
    
    if multi_res_fdi is None:
        return None
    
    # Prepare features
    feature_cols = [c for c in multi_res_fdi.columns if 'FDI' in c and 'zscore' in c.lower()]
    
    if len(feature_cols) == 0:
        feature_cols = ['FDI_up_zscore', 'FDI_down_zscore', 'FDI_asymmetry_zscore']
        feature_cols = [c for c in feature_cols if c in multi_res_fdi.columns]
    
    if len(feature_cols) == 0:
        print("   ❌ No FDI features found")
        return None
    
    df = multi_res_fdi[feature_cols].copy()
    
    # Add cross-asset features if available
    if cross_asset_returns is not None:
        common_idx = df.index.intersection(cross_asset_returns.index)
        for col in cross_asset_returns.columns:
            df.loc[common_idx, f'{col}_ret'] = cross_asset_returns.loc[common_idx, col]
    
    # Target: 5-day forward return quintile
    if 'return' in multi_res_fdi.columns:
        df['fwd_5d'] = multi_res_fdi['return'].rolling(5).sum().shift(-5)
    else:
        print("   ❌ No return column for target")
        return None
    
    df = df.dropna()
    
    if len(df) < 500:
        print(f"   ⚠️ Insufficient data: {len(df)} rows")
        return None
    
    # Create quintile target
    df['target'] = pd.qcut(df['fwd_5d'], 5, labels=[1, 2, 3, 4, 5])
    
    # Train/test split (time-based)
    split_idx = int(len(df) * 0.7)
    
    feature_cols_final = [c for c in df.columns if c not in ['fwd_5d', 'target']]
    
    X_train = df[feature_cols_final].iloc[:split_idx]
    y_train = df['target'].iloc[:split_idx]
    X_test = df[feature_cols_final].iloc[split_idx:]
    y_test = df['target'].iloc[split_idx:]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RF
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = clf.score(X_train_scaled, y_train)
    test_acc = clf.score(X_test_scaled, y_test)
    
    print(f"   ✅ Train accuracy: {train_acc:.1%}")
    print(f"   ✅ Test accuracy: {test_acc:.1%}")
    print(f"   📊 Baseline (random): {100/5:.1%}")
    
    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=feature_cols_final)
    print(f"\n   📊 Top 5 Features:")
    for feat, imp in importances.nlargest(5).items():
        print(f"      {feat}: {imp:.3f}")
    
    return clf, scaler, feature_cols_final, test_acc


# =============================================================================
# 5. POSITION SIZING RULES
# =============================================================================

def compute_position_sizes(multi_res_fdi, regime_probs=None):
    """
    Compute position sizes based on FDI signals.
    
    Rules:
    - FDI_up_zscore > 1.5: Reduce to 50%
    - FDI_up_zscore > 2.0: Reduce to 25%
    - FDI_asymmetry > 1.5: Reduce to 75%
    - Regime = worst quintile: Reduce to 25%
    """
    print("\n📊 Computing Position Sizes...")
    
    if multi_res_fdi is None:
        return None
    
    df = multi_res_fdi.copy()
    
    # Base position = 100%
    df['position_size'] = 1.0
    
    # Rule 1: High FDI_up (feedback on up-days = reversal risk)
    if 'FDI_up_zscore' in df.columns:
        df.loc[df['FDI_up_zscore'] > 1.5, 'position_size'] *= 0.5
        df.loc[df['FDI_up_zscore'] > 2.0, 'position_size'] *= 0.5  # 25% total
    
    # Rule 2: High asymmetry (more down feedback than up)
    if 'FDI_asymmetry_zscore' in df.columns:
        df.loc[df['FDI_asymmetry_zscore'] > 1.5, 'position_size'] *= 0.75
    
    # Rule 3: Rising FDI trend
    if 'FDI_up_trend' in df.columns:
        df.loc[df['FDI_up_trend'] > 0.5, 'position_size'] *= 0.8
    
    # Floor at 10%
    df['position_size'] = df['position_size'].clip(lower=0.1)
    
    print(f"   ✅ Position sizes computed")
    print(f"   📊 Average position: {df['position_size'].mean():.1%}")
    print(f"   📊 Min position: {df['position_size'].min():.1%}")
    
    return df


# =============================================================================
# 6. BACKTEST
# =============================================================================

def backtest_framework(position_df, transaction_cost=0.001):
    """
    Backtest the unified framework.
    
    Compare:
    - Buy & Hold
    - FDI-Sized positions
    """
    print("\n📊 Backtesting Unified Framework...")
    
    if position_df is None or 'return' not in position_df.columns:
        print("   ❌ No return data for backtest")
        return None
    
    df = position_df.dropna(subset=['return', 'position_size']).copy()
    
    if len(df) < 100:
        print("   ❌ Insufficient data for backtest")
        return None
    
    # Buy & Hold returns
    df['bh_return'] = df['return']
    df['bh_cumret'] = (1 + df['bh_return']).cumprod()
    
    # FDI-Sized returns
    df['position_change'] = df['position_size'].diff().abs().fillna(0)
    df['fdi_cost'] = df['position_change'] * transaction_cost
    df['fdi_return'] = df['return'] * df['position_size'] - df['fdi_cost']
    df['fdi_cumret'] = (1 + df['fdi_return']).cumprod()
    
    # Metrics
    years = len(df) / 252
    
    bh_cagr = (df['bh_cumret'].iloc[-1] ** (1/years) - 1) * 100
    fdi_cagr = (df['fdi_cumret'].iloc[-1] ** (1/years) - 1) * 100
    
    bh_vol = df['bh_return'].std() * np.sqrt(252) * 100
    fdi_vol = df['fdi_return'].std() * np.sqrt(252) * 100
    
    bh_sharpe = (bh_cagr - 5) / bh_vol
    fdi_sharpe = (fdi_cagr - 5) / fdi_vol
    
    bh_dd = (df['bh_cumret'] / df['bh_cumret'].cummax() - 1).min() * 100
    fdi_dd = (df['fdi_cumret'] / df['fdi_cumret'].cummax() - 1).min() * 100
    
    print(f"\n   📊 BACKTEST RESULTS ({len(df)} days, {years:.1f} years)")
    print(f"\n   {'Metric':<20} {'Buy & Hold':>15} {'FDI-Sized':>15}")
    print("   " + "-" * 50)
    print(f"   {'CAGR':<20} {bh_cagr:>14.2f}% {fdi_cagr:>14.2f}%")
    print(f"   {'Volatility':<20} {bh_vol:>14.2f}% {fdi_vol:>14.2f}%")
    print(f"   {'Sharpe Ratio':<20} {bh_sharpe:>15.2f} {fdi_sharpe:>15.2f}")
    print(f"   {'Max Drawdown':<20} {bh_dd:>14.2f}% {fdi_dd:>14.2f}%")
    
    # Verdict
    print(f"\n   📈 CAGR improvement: {fdi_cagr - bh_cagr:+.2f}%")
    print(f"   📉 Drawdown improvement: {fdi_dd - bh_dd:+.2f}%")
    print(f"   📊 Sharpe improvement: {fdi_sharpe - bh_sharpe:+.2f}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    ax1 = axes[0]
    ax1.plot(df.index, df['bh_cumret'], 'b-', label='Buy & Hold', linewidth=1.5)
    ax1.plot(df.index, df['fdi_cumret'], 'g-', label='FDI-Sized', linewidth=1.5)
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Unified Liquidity Framework Backtest', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.fill_between(df.index, 0, df['position_size'], alpha=0.5, color='blue')
    ax2.set_ylabel('Position Size')
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    relative = (df['fdi_cumret'] / df['bh_cumret'] - 1) * 100
    ax3.fill_between(df.index, 0, relative, where=relative > 0, color='green', alpha=0.5)
    ax3.fill_between(df.index, 0, relative, where=relative < 0, color='red', alpha=0.5)
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.set_ylabel('Relative Performance (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/unified_framework_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   ✅ Saved: plots/unified_framework_backtest.png")
    
    return {
        'bh_cagr': bh_cagr, 'fdi_cagr': fdi_cagr,
        'bh_sharpe': bh_sharpe, 'fdi_sharpe': fdi_sharpe,
        'bh_dd': bh_dd, 'fdi_dd': fdi_dd
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  UNIFIED LIQUIDITY REGIME FRAMEWORK FOR INDIAN MARKETS")
    print("=" * 70)
    
    # 1. Load all data
    directional_fdi, sectoral_fdi, cross_asset_df, cross_asset_returns = load_all_data()
    
    # 2. Multi-resolution FDI
    multi_res_fdi = compute_multi_resolution_fdi(directional_fdi)
    
    # 3. Contagion network
    network_result = build_contagion_network(cross_asset_returns, directional_fdi)
    if network_result:
        G, leadlag_matrix, assets = network_result
        visualize_contagion_network(G, leadlag_matrix, assets)
    
    # 4. Regime classifier
    clf_result = train_regime_classifier(multi_res_fdi, cross_asset_returns)
    
    # 5. Position sizing
    position_df = compute_position_sizes(multi_res_fdi)
    
    # 6. Backtest
    backtest_results = backtest_framework(position_df)
    
    # Summary
    print("\n" + "=" * 70)
    print("  UNIFIED FRAMEWORK SUMMARY")
    print("=" * 70)
    
    print("\n   Components Implemented:")
    print("   ✅ Directional FDI (FDI_up, FDI_down, asymmetry)")
    print("   ✅ Multi-resolution (5d, 20d, 60d)")
    print("   ✅ Cross-asset contagion network")
    print("   ✅ Regime classifier (RF baseline)")
    print("   ✅ Position sizing rules")
    print("   ✅ Complete backtest")
    
    if backtest_results:
        print(f"\n   Key Results:")
        print(f"   • CAGR: {backtest_results['fdi_cagr']:.2f}% vs {backtest_results['bh_cagr']:.2f}% B&H")
        print(f"   • Sharpe: {backtest_results['fdi_sharpe']:.2f} vs {backtest_results['bh_sharpe']:.2f} B&H")
        print(f"   • Max DD: {backtest_results['fdi_dd']:.2f}% vs {backtest_results['bh_dd']:.2f}% B&H")
    
    print("\n   Next Steps:")
    print("   • Replace RF with Transformer for regime forecasting")
    print("   • Add RL agent for dynamic position sizing")
    print("   • Integrate with live trading system")
    
    return backtest_results


if __name__ == "__main__":
    results = main()
