import pandas as pd
import plotly.graph_objects as go
import os

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_2d_comparison():
    print("Generating Volume Comparison 2D Plot...")
    
    # Load High/Low Vol Surface Data
    hf = os.path.join(RESULTS_DIR, "surface_minnows_high_vol.csv")
    lf = os.path.join(RESULTS_DIR, "surface_minnows_low_vol.csv")
    
    if not (os.path.exists(hf) and os.path.exists(lf)):
        print("Missing surface data.")
        return
        
    df_h = pd.read_csv(hf)
    df_l = pd.read_csv(lf)
    
    # Filter for Lag 10
    h_lag = df_h[df_h['lag'] == 10].sort_values('push_bin')
    l_lag = df_l[df_l['lag'] == 10].sort_values('push_bin')
    
    fig = go.Figure()
    
    # High Volume Trace
    fig.add_trace(go.Scatter(
        x=h_lag['push_bin'], y=h_lag['avg_response'],
        mode='lines+markers', name='High Volume',
        line=dict(color='blue', width=3)
    ))
    
    # Low Volume Trace
    fig.add_trace(go.Scatter(
        x=l_lag['push_bin'], y=l_lag['avg_response'],
        mode='lines+markers', name='Low Volume',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    fig.add_vline(x=0, line_dash="solid", line_color="gray")
    
    fig.update_layout(
        title="<b>Volume Physics: High vs Low Volume (Lag 10)</b>",
        xaxis_title="Price Push (Z-Score)",
        yaxis_title="Response (Return)",
        template="plotly_white",
        width=900, height=600
    )
    
    out_file = os.path.join(PLOTS_DIR, "volume_comparison_2d.html")
    fig.write_html(out_file)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_2d_comparison()
