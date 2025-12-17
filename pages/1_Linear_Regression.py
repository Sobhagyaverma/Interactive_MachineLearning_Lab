import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
from models.linear_regression import LinearRegression

# Page Config
st.set_page_config(page_title="Linear Regression", layout="wide")
st.title("📏 Linear Regression Playground")

# --- SIDEBAR: GOD MODE (Data Generator) ---
with st.sidebar:
    st.header("🛠️ 1. Create Data")
    n_samples = st.slider("Samples", 10, 200, 50)
    noise = st.slider("Noise (Messiness)", 0, 50, 10)
    true_slope = st.slider("True Slope", 0.5, 5.0, 2.5)
    
    if st.button("🔄 Generate New Universe"):
        np.random.seed(int(time.time())) # True random
        X = 2 * np.random.rand(n_samples, 1)
        y = true_slope * X + 4 + np.random.randn(n_samples, 1) * (noise / 10)
        
        # Save to Session State (Memory)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['trained'] = False

# Initialize Data if not present
if 'X' not in st.session_state:
    st.session_state['X'] = 2 * np.random.rand(50, 1)
    st.session_state['y'] = 2.5 * st.session_state['X'] + 4 + np.random.randn(50, 1)

X = st.session_state['X']
y = st.session_state['y']

# --- MAIN SECTION: THE COCKPIT ---
col1, col2, col3 = st.columns(3)
with col1:
    lr = st.number_input("Learning Rate", value=0.01, step=0.001, format="%.4f")
with col2:
    epochs = st.slider("Iterations", 50, 500, 100)
with col3:
    st.write("##")
    train_btn = st.button("🚀 Train Model", type="primary")

# --- VISUALIZATION AREA ---
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

if train_btn:
    model = LinearRegression(learning_rate=lr, n_iterations=epochs)
    model.fit(X, y)
    st.session_state['trained'] = True
    
    # ANIMATION LOOP
    progress_bar = st.progress(0)
    
    # We skip frames to make animation faster if many epochs
    skip = max(1, epochs // 20) 
    
    for i in range(0, epochs, skip):
        # Retrieve weights from history
        weights, bias = model.train_history[i]
        current_cost = model.history[i]
        
        # Create Line
        x_range = np.linspace(0, 2, 100).reshape(-1, 1)
        y_pred_line = weights[0][0] * x_range + bias
        
        # Draw Plot
        fig = go.Figure()
        
        # 1. The Data Points
        fig.add_trace(go.Scatter(
            x=X.flatten(), y=y.flatten(), mode='markers', 
            name='Data', marker=dict(color='cyan', size=10)
        ))
        
        # 2. The Learning Line
        fig.add_trace(go.Scatter(
            x=x_range.flatten(), y=y_pred_line.flatten(), mode='lines', 
            name='Model', line=dict(color='red', width=4)
        ))

        fig.update_layout(
            title=f"Epoch: {i}/{epochs} | Error (Cost): {current_cost:.4f}",
            xaxis_title="X (Input)", yaxis_title="y (Output)",
            template="plotly_dark", height=500
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.01) # Small delay for smooth animation
        progress_bar.progress((i+1)/epochs)

    progress_bar.empty()
    st.success("Training Complete!")

# Show static chart if not training but data exists
elif not st.session_state.get('trained', False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y.flatten(), mode='markers', marker=dict(color='cyan')))
    fig.update_layout(template="plotly_dark", title="Raw Data (Waiting to Train)", height=500)
    chart_placeholder.plotly_chart(fig, use_container_width=True)