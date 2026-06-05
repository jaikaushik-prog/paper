import pandas as pd
import plotly.graph_objects as go
import os

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_volume_surface(regime):
    print(f"Plotting Surface: {regime}...")
    file_path = os.path.join(RESULTS_DIR, f"surface_minnows_{regime}.csv")
    if not os.path.exists(file_path): 
        print(f"File missing: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    p_surf = df.pivot(index='push_bin', columns='lag', values='avg_response')
    
    title_map = {
        'high_vol': "<b>Minnows: High Volume (Z > 1.5)</b>",
        'low_vol': "<b>Minnows: Low Volume (Z < -0.5)</b>"
    }
    
    fig = go.Figure(data=[go.Surface(
        z=p_surf.values,
        x=p_surf.columns,
        y=p_surf.index,
        colorscale='Viridis',
        colorbar=dict(title="Response Z")
    )])
    
    fig.update_layout(
        title=title_map.get(regime, regime),
        scene=dict(
            xaxis_title='Lag',
            yaxis_title='Push Z',
            zaxis_title='Response Z'
        ),
        width=900, height=700
    )
    
    out_file = os.path.join(PLOTS_DIR, f"surface_minnows_{regime}.html")
    fig.write_html(out_file)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_volume_surface("high_vol")
    plot_volume_surface("low_vol")
