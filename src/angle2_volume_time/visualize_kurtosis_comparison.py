import pandas as pd
import plotly.graph_objects as go
import os

# Hardcoded results from analyze_kurtosis.py output
# Average Clock Kurtosis: 2746.65
# Average Volume Kurtosis: 3145.18

def plot_kurtosis_comparison():
    print("Generating Kurtosis Comparison Plot...")
    
    categories = ['Clock Time (5-min)', 'Volume Time (1/50th Daily)']
    values = [2746.65, 3145.18]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            text=values,
            textposition='auto',
            marker_color=['#1f77b4', '#d62728'], # Blue vs Red (Failure)
        )
    ])
    
    fig.update_layout(
        title="<b>The Failure of Volume Time Normalization</b><br>Average Kurtosis (Fat Tails) - Top 20 Whales",
        yaxis_title="Kurtosis (Higher = Fatter Tails)",
        template="plotly_white",
        width=800, height=600,
        annotations=[
            dict(
                x=1, y=3145,
                xref="x", yref="y",
                text="<b>+14.5% Worse</b>",
                showarrow=True,
                arrowhead=1,
                ax=0, ay=-40
            )
        ]
    )
    
    out_path = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots\kurtosis_comparison.html"
    fig.write_html(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_kurtosis_comparison()
