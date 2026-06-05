import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_wick_divergence():
    print("Generating Wick Divergence Map...")
    file_path = os.path.join(RESULTS_DIR, "wick_heatmap_data.csv")
    if not os.path.exists(file_path): 
        print("Data missing.")
        return
    
    df = pd.read_csv(file_path)
    
    # Pivot for Heatmap
    # X: z_ret_val, Y: z_wick_val, Z: response
    
    # We need a matrix
    heatmap_data = df.pivot(index='z_wick_val', columns='z_ret_val', values='response')
    
    # Interactive Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="Response (Log Ret)"),
    ))
    
    fig.update_layout(
        title="<b>Wick Divergence Map (Trap Zone)</b>",
        xaxis_title="Price Momentum (z_ret)",
        yaxis_title="Candle Quality (z_wick)",
        width=900, height=700
    )
    
    # Annotations for Zones
    fig.add_annotation(x=3, y=3, text="High Quality Breakout", showarrow=False, font=dict(color="white"))
    fig.add_annotation(x=3, y=-3, text="Bull Trap (Fakeout)", showarrow=False, font=dict(color="white"))
    
    out_file = os.path.join(PLOTS_DIR, "wick_divergence_map.html")
    fig.write_html(out_file)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_wick_divergence()
