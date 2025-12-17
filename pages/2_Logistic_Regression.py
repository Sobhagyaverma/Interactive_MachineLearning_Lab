import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from models.logistic_regression import LogisticRegression
from utils import accuracy_score

st.set_page_config(page_title="Logistic Regression", layout="wide")
st.title("🧬 Logistic Regression: Cancer Detection")

# --- 1. SIDEBAR: DATA GENERATOR ---
with st.sidebar:
    st.header("🛠️ Create Tumor Data")
    n_samples = st.slider("Samples", 50, 500, 200)
    noise = st.slider("Overlap (Noise)", 0.0, 1.0, 0.2)
    
    if st.button("🔄 Generate Patient Data"):
        np.random.seed(int(time.time()))
        
        # Use integer division (e.g., 201 // 2 = 100)
        half_samples = n_samples // 2
        
        # Class 0: Centered at (2, 2)
        X0 = np.random.randn(half_samples, 2) + 2
        y0 = np.zeros((half_samples, 1))
        
        # Class 1: Centered at (4, 4)
        X1 = np.random.randn(half_samples, 2) + 4
        y1 = np.ones((half_samples, 1))
        
        # Combine
        X = np.vstack((X0, X1))
        y = np.vstack((y0, y1))
        
        # Add Noise
        X += np.random.randn(X.shape[0], 2) * noise
        
        st.session_state['log_X'] = X
        st.session_state['log_y'] = y
        st.session_state['log_trained'] = False

# Initialize Data
if 'log_X' not in st.session_state:
    st.session_state['log_X'] = np.random.randn(100, 2)
    st.session_state['log_y'] = np.random.randint(0, 2, (100, 1))

X = st.session_state['log_X']
y = st.session_state['log_y']

# --- 2. MAIN CONTROLS ---
col1, col2, col3 = st.columns(3)
with col1:
    lr = st.number_input("Learning Rate", value=0.1, step=0.01)
with col2:
    epochs = st.slider("Epochs", 50, 1000, 200)
with col3:
    st.write("##")
    train_btn = st.button("🚀 Train Diagnosis Model", type="primary")

chart_placeholder = st.empty()

# --- 3. VISUALIZATION HELPER ---
def plot_decision_boundary(X, y, w, b, epoch, cost):
    fig = go.Figure()
    
    # Plot Data Points (Benign vs Malignant)
    mask = y.flatten() == 0
    fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name='Benign (0)', marker=dict(color='blue', size=8, opacity=0.6)))
    fig.add_trace(go.Scatter(x=X[~mask, 0], y=X[~mask, 1], mode='markers', name='Malignant (1)', marker=dict(color='red', size=8, opacity=0.6)))

    # Plot Decision Boundary Line
    # The line is where w1*x1 + w2*x2 + b = 0
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_vals = np.linspace(x1_min, x1_max, 100)
    
    if w[1] != 0:
        x2_vals = -(w[0] * x1_vals + b) / w[1]
        # --- CHANGE IS HERE ---
        fig.add_trace(go.Scatter(x=x1_vals, y=x2_vals, mode='lines', name='Boundary', line=dict(color='green', width=3, dash='dash')))
        # ----------------------

    fig.update_layout(
        title=f"Epoch {epoch} | Log Loss: {cost:.4f}",
        xaxis_title="Feature 1 (e.g. Tumor Size)",
        yaxis_title="Feature 2 (e.g. Tumor Density)",
        template="plotly_white",
        height=600
    )
    return fig

# --- 4. TRAINING LOOP ---
if train_btn:
    try:
        model = LogisticRegression(learning_rate=lr, n_iterations=epochs)
        model.fit(X, y)
        st.session_state['log_trained'] = True
        
        progress_bar = st.progress(0)
        skip = max(1, epochs // 20)
        
        for i in range(0, epochs, skip):
            curr_w, curr_b = model.train_history[i]
            curr_cost = model.history[i]
            
            fig = plot_decision_boundary(X, y, curr_w, curr_b, i, curr_cost)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.01)
            progress_bar.progress((i+1)/epochs)
            
        progress_bar.empty()
        
        # Final Metrics
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        st.success(f"Training Complete! Final Accuracy: {acc*100:.2f}%")
        
    except Exception as e:
        st.error(f"⚠️ Math Error: {e}")
        st.info("Did you implement the 'sigmoid' and 'fit' methods in models/logistic_regression.py?")

elif not st.session_state.get('log_trained', False):
    # Initial static plot
    fig = plot_decision_boundary(X, y, np.array([[0],[1]]), 0, 0, 0)
    chart_placeholder.plotly_chart(fig, use_container_width=True)