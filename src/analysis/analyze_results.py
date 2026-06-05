import pandas as pd
import numpy as np
import os

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"

def load_surface(group_name):
    path = os.path.join(RESULTS_DIR, f"surface_{group_name}.csv")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return pd.read_csv(path)

def calculate_metrics(df, group_name):
    """
    Calculate M(L) and rho(L) as per the paper.
    
    Conditionals:
    M(L) = Weighted Mean Absolute Response strength.
    rho(L) = Asymmetry dominance (Antisymmetry vs Symmetry).
    
    The paper definitions (simplified for our binned data):
    S(L, |j|) = 0.5 * (z_r(L, +j) + z_r(L, -j))
    A(L, |j|) = 0.5 * (z_r(L, +j) - z_r(L, -j))
    
    Aggregating over |j| (magnitude bins):
    We pair +bin and -bin. 
    """
    if df is None or df.empty:
        return
        
    print(f"\nAnalysis for: {group_name.upper()}")
    
    # Ensure we use push_bin as float
    df['push_bin'] = df['push_bin'].astype(float).round(2)
    
    # Filter for valid pairs
    # Create a Pivot Table: Index=Lag, Columns=PushBin, Values=AvgResponse
    pivot = df.pivot(index='lag', columns='push_bin', values='avg_response')
    
    lags = pivot.index.tolist()
    bins = pivot.columns.tolist()
    
    # Find symmetric bin pairs
    # Bins are centered around 0. e.g. -3.95 and 3.95
    # We look for pairs b and -b
    
    positive_bins = [b for b in bins if b > 0]
    
    m_L_stats = []
    rho_L_stats = []
    
    for lag in lags:
        row = pivot.loc[lag]
        
        sum_abs_A = 0
        sum_abs_S = 0
        sum_strength = 0
        count = 0
        
        for p_bin in positive_bins:
            n_bin = -p_bin # Negative counterpart
            
            # Check availability (using approx match for float safety if needed, but direct look up should work if grid is perfect)
            # Find closest negative bin if exact match fails?
            # Our grid is clean: -4.0 to 4.0 step 0.1 centered.
            # let's try direct look up
            
            if p_bin in row and n_bin in row:
                if pd.isna(row[p_bin]) or pd.isna(row[n_bin]):
                    continue
                    
                resp_pos = row[p_bin]
                resp_neg = row[n_bin]
                
                # Eqs 3.17, 3.18
                S = 0.5 * (resp_pos + resp_neg)
                A = 0.5 * (resp_pos - resp_neg)
                
                # Weighting: Paper uses support count. We don't have count here (aggregated mean only).
                # Assuming equal weight for now or uniform support (which is roughly true for center bins).
                # For robust metric, just sum magnitudes.
                
                sum_abs_A += abs(A)
                sum_abs_S += abs(S)
                sum_strength += (abs(resp_pos) + abs(resp_neg)) / 2
                count += 1
        
        if count > 0:
            # Eq 3.21: rho(L) = (Sum |A| - Sum |S|) / (Sum |A| + Sum |S|)
            total_mag = sum_abs_A + sum_abs_S
            if total_mag > 0:
                rho = (sum_abs_A - sum_abs_S) / total_mag
            else:
                rho = 0
                
            # M(L) strength
            M_L = sum_strength / count # Average strength per bin pair
            
            m_L_stats.append({'lag': lag, 'M(L)': M_L})
            rho_L_stats.append({'lag': lag, 'rho(L)': rho})
            
    m_df = pd.DataFrame(m_L_stats)
    rho_df = pd.DataFrame(rho_L_stats)
    
    # Print summary stats
    if not m_df.empty:
        print(f"Average Response Strength (M): {m_df['M(L)'].mean():.4f}")
        print(f"Max Response Strength (M): {m_df['M(L)'].max():.4f}")
        
    if not rho_df.empty:
        print(f"Average Asymmetry Dominance (rho): {rho_df['rho(L)'].mean():.4f}")
        # Positive rho -> Directional (Predictable drift)
        # Negative rho -> Symmetric (Reversion/Volatility clustering without drift)
        
    return m_df, rho_df

    return m_df, rho_df

def calculate_panic_premium(surface_minnows_df, surface_whales_df):
    """
    Quantifies the asymmetry between negative shocks (panic) and positive shocks (greed).
    """
    results = {}
    
    # Pre-process: Create Mean Profile (Average over all lags)
    # Series index = push_bin, value = avg_response
    w_profile = surface_whales_df.groupby('push_bin')['avg_response'].mean()
    m_profile = surface_minnows_df.groupby('push_bin')['avg_response'].mean()
    
    for name, surface in [("Minnows", m_profile), ("Whales", w_profile)]:
        # 1. Isolate Panic vs Greed
        # |z_p| > 1.0
        panic_zone = surface[surface.index < -1.0]
        greed_zone = surface[surface.index > 1.0]
        
        # 2. Calculate Energy (Area/Sum of absolute values)
        # We take absolute values to measure magnitude of reaction
        panic_energy = np.abs(panic_zone).sum()
        greed_energy = np.abs(greed_zone).sum()
        
        # 3. Ratio
        ratio = panic_energy / greed_energy if greed_energy != 0 else np.nan
        
        results[name] = {
            "Panic Energy": panic_energy,
            "Greed Energy": greed_energy,
            "Asymmetry Ratio": ratio
        }
        
    return pd.DataFrame(results).T

def plot_asymmetry(df_results):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plotting the Ratio
    colors = ['firebrick' if x > 1.1 else 'gray' for x in df_results['Asymmetry Ratio']]
    bars = ax.bar(df_results.index, df_results['Asymmetry Ratio'], color=colors, alpha=0.8, width=0.5)
    
    # Reference line at 1.0 (Perfect Symmetry)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label="Symmetry (1.0)")
    
    ax.set_title("The 'Panic Premium': Reaction to Negative vs. Positive Shocks")
    ax.set_ylabel("Asymmetry Ratio (Panic / Greed)")
    ax.set_ylim(0, max(df_results['Asymmetry Ratio']) * 1.3)
    
    # Labeling
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(r"c:\Users\DELL\Desktop\project_nifty_liquid\plots", "panic_premium.png")
    plt.savefig(output_path)
    print(f"Saved Panic Premium plot to {output_path}")

if __name__ == "__main__":
    whales = load_surface("whales")
    minnows = load_surface("minnows")
    
    # 1. Original Lag-based metrics
    m_w, r_w = calculate_metrics(whales, "whales")
    m_m, r_m = calculate_metrics(minnows, "minnows")
    
    print("\n--- LAG METRICS COMPARISON ---")
    if m_w is not None and m_m is not None:
        avg_w = m_w['M(L)'].mean()
        avg_m = m_m['M(L)'].mean()
        ratio = avg_m / avg_w if avg_w else 0
        print(f"Inefficiency Ratio (Minnows / Whales): {ratio:.2f}x")
    
    # 2. Panic Premium (Aggregated)
    print("\n--- PANIC PREMIUM ANALYSIS ---")
    pp_results = calculate_panic_premium(minnows, whales)
    print(pp_results)
    
    # 3. Visualize
    plot_asymmetry(pp_results)
