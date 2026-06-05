import pandas as pd
import plotly.graph_objects as go
import os
import glob

# Configuration
RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def create_3d_surface(df, group_name):
    print(f"Generating plot for {group_name}...")
    
    # Pivot to create a grid: Index=Lag, Columns=PushBin, Values=Response
    # Ensure sorted index and columns
    pivot_df = df.pivot(index='lag', columns='push_bin', values='avg_response')
    pivot_df = pivot_df.sort_index(ascending=True) # Sort lags
    pivot_df = pivot_df.sort_index(axis=1, ascending=True) # Sort push bins
    
    # X: Push Bins (Columns)
    x = pivot_df.columns.values
    # Y: Lags (Index)
    y = pivot_df.index.values
    # Z: Response (Values)
    z = pivot_df.values
    
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    
    fig.update_layout(
        title=f'Push-Response Efficiency Surface: {group_name.upper()}',
        autosize=False,
        width=1000,
        height=800,
        scene=dict(
            xaxis_title='Standardized Push (z_p)',
            yaxis_title='Lag (L)',
            zaxis_title='Avg Standardized Response (z_r)'
        )
    )
    
    out_file = os.path.join(PLOTS_DIR, f"surface_{group_name}.html")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.write_html(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    files = glob.glob(os.path.join(RESULTS_DIR, "surface_*.csv"))
    
    if not files:
        print("No result files found.")
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            # filename like surface_whales.csv
            group_name = filename.replace("surface_", "").replace(".csv", "")
            
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Skipping empty results for {group_name}")
                continue
                
            create_3d_surface(df, group_name)
            
        except Exception as e:
            print(f"Error plotting {file_path}: {e}")
