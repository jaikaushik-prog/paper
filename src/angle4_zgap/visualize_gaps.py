import pandas as pd
import plotly.graph_objects as go
import os

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_combined_gaps():
    print("Generating Gap Response Plot...")
    
    # Load Data
    w_path = os.path.join(RESULTS_DIR, "gap_curve_whales.csv")
    m_path = os.path.join(RESULTS_DIR, "gap_curve_minnows.csv")
    
    if not os.path.exists(w_path) or not os.path.exists(m_path):
        print("Missing gap curve data.")
        return
        
    df_w = pd.read_csv(w_path)
    df_m = pd.read_csv(m_path)
    
    fig = go.Figure()
    
    # Whales Trace (Green/Red? Blue)
    fig.add_trace(go.Scatter(
        x=df_w['bin_mid'], 
        y=df_w['response_30m'],
        mode='lines+markers',
        name='Whales (Liquid)',
        line=dict(color='blue', width=3)
    ))
    
    # Minnows Trace (Red)
    fig.add_trace(go.Scatter(
        x=df_m['bin_mid'], 
        y=df_m['response_30m'],
        mode='lines+markers',
        name='Minnows (Illiquid)',
        line=dict(color='red', width=3)
    ))
    
    # Reference Lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # Annotations
    fig.update_layout(
        title="<b>Z-Gap Response</b>: Fade vs. Follow (First 30 Mins)",
        xaxis_title="Standardized Overnight Gap (z_gap)",
        yaxis_title="30-min Intraday Return (Response)",
        width=900,
        height=600,
        template='plotly_white'
    )
    
    # Save
    out_file = os.path.join(PLOTS_DIR, "z_gap_response.html")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.write_html(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    plot_combined_gaps()
