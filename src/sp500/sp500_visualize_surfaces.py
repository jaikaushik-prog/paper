"""
S&P 500 Visualize Surfaces
Generates interactive 3D Plotly visualizations of efficiency surfaces.
"""

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def load_surface(group_name):
    """Load surface data from CSV."""
    file_path = os.path.join(RESULTS_DIR, f"sp500_surface_{group_name}.csv")
    return pd.read_csv(file_path)

def create_surface_plot(df, group_name, title_suffix=""):
    """
    Create an interactive 3D surface plot.
    
    X-axis: Lag (days)
    Y-axis: Push Z-score
    Z-axis: Average Response Z-score
    """
    # Pivot data for surface
    pivot = df.pivot(index='push_bin', columns='lag', values='avg_response')
    
    # Get axes
    lags = pivot.columns.values
    z_bins = pivot.index.values
    responses = pivot.values
    
    # Create meshgrid
    X, Y = np.meshgrid(lags, z_bins)
    Z = responses
    
    # Create figure
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='RdBu_r',
        cmin=-0.5,
        cmax=0.5,
        colorbar=dict(title='Avg Response (Z)')
    ))
    
    # Layout
    display_name = "Whales (Large Cap)" if group_name == "whales" else "Minnows (Small Cap)"
    
    fig.update_layout(
        title=dict(
            text=f"S&P 500 {display_name} - Efficiency Surface{title_suffix}",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title="Lag (Trading Days)",
            yaxis_title="Push Z-score",
            zaxis_title="Response Z-score",
            xaxis=dict(tickmode='linear', dtick=5),
            yaxis=dict(range=[-4, 4]),
            zaxis=dict(range=[-0.5, 0.5]),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)
            )
        ),
        width=900,
        height=700
    )
    
    return fig

def create_2d_comparison(whales_df, minnows_df, lag=5):
    """
    Create a 2D line plot comparing Whales vs Minnows at a specific lag.
    """
    whales_lag = whales_df[whales_df['lag'] == lag]
    minnows_lag = minnows_df[minnows_df['lag'] == lag]
    
    fig = go.Figure()
    
    # Whales
    fig.add_trace(go.Scatter(
        x=whales_lag['push_bin'],
        y=whales_lag['avg_response'],
        mode='lines+markers',
        name='Whales (Large Cap)',
        line=dict(color='blue', width=2)
    ))
    
    # Minnows
    fig.add_trace(go.Scatter(
        x=minnows_lag['push_bin'],
        y=minnows_lag['avg_response'],
        mode='lines+markers',
        name='Minnows (Small Cap)',
        line=dict(color='red', width=2)
    ))
    
    # Reference line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"S&P 500 Push-Response Curve (Lag = {lag} days)",
        xaxis_title="Push Z-score",
        yaxis_title="Average Response Z-score",
        legend=dict(x=0.02, y=0.98),
        width=800,
        height=500
    )
    
    return fig

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load surfaces
    print("Loading surface data...")
    whales = load_surface("whales")
    minnows = load_surface("minnows")
    
    print(f"Whales: {len(whales)} data points")
    print(f"Minnows: {len(minnows)} data points")
    
    # Create 3D surface plots
    print("\nGenerating 3D surface plots...")
    
    whales_fig = create_surface_plot(whales, "whales")
    whales_file = os.path.join(PLOTS_DIR, "sp500_surface_whales.html")
    whales_fig.write_html(whales_file)
    print(f"  Saved: {whales_file}")
    
    minnows_fig = create_surface_plot(minnows, "minnows")
    minnows_file = os.path.join(PLOTS_DIR, "sp500_surface_minnows.html")
    minnows_fig.write_html(minnows_file)
    print(f"  Saved: {minnows_file}")
    
    # Create 2D comparison plots for key lags
    print("\nGenerating 2D comparison plots...")
    
    for lag in [5, 10, 20]:
        try:
            compare_fig = create_2d_comparison(whales, minnows, lag)
            compare_file = os.path.join(PLOTS_DIR, f"sp500_compare_lag{lag}.html")
            compare_fig.write_html(compare_file)
            print(f"  Saved: {compare_file}")
        except Exception as e:
            print(f"  Skipped lag {lag}: {e}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
