import pandas as pd
import plotly.graph_objects as go
import os

RESULTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\results"
PLOTS_DIR = r"c:\Users\DELL\Desktop\project_nifty_liquid\plots"

def plot_dynamic_comparison():
    print("Generating Dynamic Comparison Plot...")
    
    wf = os.path.join(RESULTS_DIR, "surface_dynamic_whales.csv")
    mf = os.path.join(RESULTS_DIR, "surface_dynamic_minnows.csv")
    
    if not (os.path.exists(wf) and os.path.exists(mf)):
        print("Missing dynamic data.")
        return
        
    df_w = pd.read_csv(wf)
    df_m = pd.read_csv(mf)
    
    # Lag 10 Slice
    w_lag = df_w[df_w['lag'] == 10].sort_values('push_bin')
    m_lag = df_m[df_m['lag'] == 10].sort_values('push_bin')
    
    fig = go.Figure()
    
    # Whales
    fig.add_trace(go.Scatter(
        x=w_lag['push_bin'], y=w_lag['avg_response'],
        mode='lines', name='Dynamic Whales',
        line=dict(color='blue', width=3)
    ))
    
    # Minnows
    fig.add_trace(go.Scatter(
        x=m_lag['push_bin'], y=m_lag['avg_response'],
        mode='lines', name='Dynamic Minnows',
        line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="<b>Dynamic Liquidity Stratification (Lag 10)</b><br>Rolling ADT Re-ranking",
        xaxis_title="Price Push (Z-Score)",
        yaxis_title="Response (Return)",
        template="plotly_white",
        width=900, height=600
    )
    
    fig.write_html(os.path.join(PLOTS_DIR, "defense_dynamic_liquidity.html"))
    print("Saved defense_dynamic_liquidity.html")

def plot_confidence_intervals():
    print("Generating Confidence Interval Plot...")
    
    mf = os.path.join(RESULTS_DIR, "surface_dynamic_minnows.csv")
    if not os.path.exists(mf): return
    
    df = pd.read_csv(mf)
    m_lag = df[df['lag'] == 10].sort_values('push_bin')
    
    # Calc CI
    m_lag['ci_upper'] = m_lag['avg_response'] + 1.96 * m_lag['sem_response']
    m_lag['ci_lower'] = m_lag['avg_response'] - 1.96 * m_lag['sem_response']
    
    fig = go.Figure()
    
    # Confidence Ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([m_lag['push_bin'], m_lag['push_bin'][::-1]]),
        y=pd.concat([m_lag['ci_upper'], m_lag['ci_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='95% Confidence Interval'
    ))
    
    # Mean Line
    fig.add_trace(go.Scatter(
        x=m_lag['push_bin'], y=m_lag['avg_response'],
        mode='lines+markers', name='Minnows Mean',
        line=dict(color='red', width=3)
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray")
    
    fig.update_layout(
        title="<b>Statistical Significance Test (Lag 10)</b><br>Minnows Response with 95% Confidence Bands",
        xaxis_title="Price Push (Z-Score)",
        yaxis_title="Response (Return)",
        template="plotly_white",
        width=900, height=600
    )
    
    fig.write_html(os.path.join(PLOTS_DIR, "defense_confidence_intervals.html"))
    print("Saved defense_confidence_intervals.html")

if __name__ == "__main__":
    plot_dynamic_comparison()
    plot_confidence_intervals()
