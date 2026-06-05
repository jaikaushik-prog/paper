import pandas as pd
import plotly.graph_objects as go
import os
import glob
from plotly.subplots import make_subplots

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_refined_surfaces(group):
    print(f"Plotting refined surfaces for {group}...")
    file_path = os.path.join(RESULTS_DIR, f"surface_refined_{group}.csv")
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    
    # We want 3D surface of Probability and Risk
    # Pivot
    p_prob = df.pivot(index='push_bin', columns='lag', values='prob_positive')
    p_risk = df.pivot(index='push_bin', columns='lag', values='risk_std')
    
    # 1. Probability Map
    fig_prob = go.Figure(data=[go.Surface(
        z=p_prob.values,
        x=p_prob.columns,
        y=p_prob.index,
        colorscale='RdBu', 
        cmin=0.4, cmax=0.6,
        colorbar=dict(title="Win Rate")
    )])
    fig_prob.update_layout(title=f"<b>{group.capitalize()} Win Rate Surface</b> (P(z_r > 0))", 
                          scene=dict(xaxis_title='Lag', yaxis_title='Push Z', zaxis_title='Prob > 0'))
    fig_prob.write_html(os.path.join(PLOTS_DIR, f"surface_prob_{group}.html"))
    
    # 2. Risk Surface
    fig_risk = go.Figure(data=[go.Surface(
        z=p_risk.values,
        x=p_risk.columns,
        y=p_risk.index,
        colorscale='Magma',
        colorbar=dict(title="Volatility")
    )])
    fig_risk.update_layout(title=f"<b>{group.capitalize()} Risk Surface</b> (StdDev(z_r))", 
                          scene=dict(xaxis_title='Lag', yaxis_title='Push Z', zaxis_title='Response Vol'))
    fig_risk.write_html(os.path.join(PLOTS_DIR, f"surface_risk_{group}.html"))

def plot_lead_lag():
    print("Plotting Lead-Lag...")
    file_path = os.path.join(RESULTS_DIR, "surface_leadlag_whales_lead_minnows.csv")
    if not os.path.exists(file_path): 
        print("Lead-Lag file not found.")
        return
    
    df = pd.read_csv(file_path)
    p_surf = df.pivot(index='push_bin', columns='lag', values='avg_response')
    
    fig = go.Figure(data=[go.Surface(
        z=p_surf.values,
        x=p_surf.columns,
        y=p_surf.index,
        colorscale='Viridis',
        colorbar=dict(title="Minnow Resp Z")
    )])
    
    fig.update_layout(
        title="<b>Lead-Lag Surface</b>: Whales(t) -> Minnows(t+Lag)",
        scene=dict(
            xaxis_title='Lag (5-min Units)',
            yaxis_title='Whale Push Z',
            zaxis_title='Minnow Response Z'
        ),
        width=900, height=700
    )
    
    fig.write_html(os.path.join(PLOTS_DIR, "surface_lead_lag.html"))
    print("Saved Lead-Lag plot.")

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_refined_surfaces("whales")
    plot_refined_surfaces("minnows")
    plot_lead_lag()
