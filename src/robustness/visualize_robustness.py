import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_amihud_scatter():
    print("Plotting Amihud vs Slope...")
    file_path = os.path.join(RESULTS_DIR, "robustness_amihud.csv")
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    # df has symbol, amihud, slope
    
    # Filter valid
    df = df[df['amihud'] > 0]
    df['log_amihud'] = np.log(df['amihud'])
    
    # Clip outliers for plot
    # Slope usually -1 to 1.
    df = df[(df['slope'] > -2) & (df['slope'] < 2)]
    
    fig = px.scatter(
        df, x='log_amihud', y='slope',
        hover_data=['symbol'],
        trendline='ols',
        trendline_color_override='red',
        title="<b>Mechanism Proof</b>: Illiquidity (Amihud) Drives Predictability (Slope)",
        labels={'log_amihud': 'Log Amihud Ratio (Illiquidity)', 'slope': 'Inefficiency Slope (Resp/Push)'}
    )
    
    fig.update_layout(width=800, height=600)
    fig.write_html(os.path.join(PLOTS_DIR, "robustness_amihud_scatter.html"))
    print("Saved Amihud scatter.")

def plot_subperiod_surface(period):
    print(f"Plotting Surface: {period}...")
    file_path = os.path.join(RESULTS_DIR, f"surface_minnows_{period}.csv")
    if not os.path.exists(file_path): return
    
    df = pd.read_csv(file_path)
    p_surf = df.pivot(index='push_bin', columns='lag', values='avg_response')
    
    fig = go.Figure(data=[go.Surface(
        z=p_surf.values,
        x=p_surf.columns,
        y=p_surf.index,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=f"<b>Minnows Surface: {period.upper()}</b>",
        scene=dict(
            xaxis_title='Lag',
            yaxis_title='Push Z',
            zaxis_title='Response Z'
        )
    )
    fig.write_html(os.path.join(PLOTS_DIR, f"surface_minnows_{period}.html"))

if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_amihud_scatter()
    plot_subperiod_surface("precovid")
    plot_subperiod_surface("postcovid")
