import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from models.knn import KNN
import utils
from utils import accuracy_score

st.set_page_config(page_title="K-Nearest Neighbors", layout="wide")
utils.navbar()
st.title("📍 K-Nearest Neighbors (KNN)")

# --- TABS ---
tab1, tab2 = st.tabs(["🎮 Playground", "📖 Theory & Math"])

# =========================================
# TAB 1: PLAYGROUND
# =========================================
with tab1:
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🛠️ Data Generator")
        n_samples = st.slider("Samples per Class", 20, 100, 50)
        noise = st.slider("Cluster Spread (Noise)", 0.5, 3.0, 1.0)
        
        if st.button("🔄 Generate 3 Classes"):
            np.random.seed(int(time.time()))
            
            # Class 0: Bottom Left (Red)
            X0 = np.random.randn(n_samples, 2) + [2, 2]
            y0 = np.zeros((n_samples, 1))
            
            # Class 1: Top Right (Blue)
            X1 = np.random.randn(n_samples, 2) + [6, 6]
            y1 = np.ones((n_samples, 1))
            
            # Class 2: Bottom Right (Green) - The "3rd" class!
            X2 = np.random.randn(n_samples, 2) + [7, 1]
            y2 = np.full((n_samples, 1), 2)
            
            # Combine
            X = np.vstack((X0, X1, X2))
            y = np.vstack((y0, y1, y2))
            
            # Add extra noise
            X += np.random.randn(X.shape[0], 2) * (noise * 0.5)
            
            st.session_state['knn_X'] = X
            st.session_state['knn_y'] = y
            st.session_state['knn_trained'] = False

    # Default Data
    if 'knn_X' not in st.session_state:
        # Generate default data silently
        X0 = np.random.randn(30, 2) + [2, 2]
        X1 = np.random.randn(30, 2) + [6, 6]
        X2 = np.random.randn(30, 2) + [7, 1]
        st.session_state['knn_X'] = np.vstack((X0, X1, X2))
        st.session_state['knn_y'] = np.vstack((np.zeros((30,1)), np.ones((30,1)), np.full((30,1), 2)))

    X = st.session_state['knn_X']
    y = st.session_state['knn_y']

    # --- CONTROLS ---
    col1, col2 = st.columns([1, 3])
    with col1:
        k_value = st.slider("K (Neighbors)", 1, 25, 3)
        st.caption("Lower K = Complex/Jagged. Higher K = Smooth.")
        train_btn = st.button("🚀 Classify Space", type="primary")

    chart_placeholder = st.empty()

    # --- VISUALIZATION FUNCTION ---
    def plot_knn_boundary(model, X, y):
        # 1. Create a meshgrid (a net of points across the whole graph)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # We lower resolution (0.2) to keep it fast. High res = slow.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                             np.arange(y_min, y_max, 0.2))
        
        # 2. Predict every single point in the net
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 3. Plot
        fig = go.Figure()
        
        # Background Contours (The Decision Regions)
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.2),
            y=np.arange(y_min, y_max, 0.2),
            z=Z,
            showscale=False,
            opacity=0.4,
            colorscale=[[0, 'red'], [0.5, 'blue'], [1, 'green']], # Custom map
            contours=dict(start=0, end=2, size=1) # Forces discrete steps
        ))
        
        # Scatter Points (The Real Data)
        colors = ['red', 'blue', 'green']
        for i in range(3):
            mask = (y == i).flatten()
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1],
                mode='markers',
                marker=dict(color=colors[i], size=10, line=dict(color='black', width=1)),
                name=f'Class {i}'
            ))
            
        fig.update_layout(
            title=f"KNN Classification (k={model.k})",
            xaxis_title="Feature 1", yaxis_title="Feature 2",
            height=600,
            template="plotly_white"
        )
        return fig

    # --- RUN LOGIC ---
    if train_btn:
        with st.spinner("Calculating distances for entire grid..."):
            model = KNN(k=k_value)
            model.fit(X, y)
            
            # This step takes a second because it predicts thousands of background dots!
            fig = plot_knn_boundary(model, X, y)
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            st.session_state['knn_trained'] = True

    elif not st.session_state.get('knn_trained', False):
        # Show raw data before training
        fig = go.Figure()
        colors = ['red', 'blue', 'green']
        for i in range(3):
            mask = (y == i).flatten()
            fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', marker=dict(color=colors[i], size=10), name=f'Class {i}'))
        fig.update_layout(title="Raw Data (3 Classes)", height=600, template="plotly_white")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 2: THEORY
# =========================================
with tab2:
    st.markdown("## 🧠 How K-Nearest Neighbors Works")
    st.write("KNN is unlike the other models we built. It does not use Calculus or Gradient Descent. It uses **Geometry**.")
    
    st.info("### The Logic: 'Tell me who your friends are, and I'll tell you who you are.'")
    
    st.markdown("### 1. Euclidean Distance")
    st.write("To find the nearest neighbor, we measure the straight-line distance between points using Pythagoras' Theorem:")
    st.latex(r"d(p, q) = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2}")
    
    st.markdown("### 2. The Voting Process")
    st.write("1. **Measure:** Calculate distance from the new point to *every* known point.")
    st.write("2. **Sort:** Find the **K** closest points.")
    st.write("3. **Vote:** If 2 neighbors are Red and 1 is Blue, the new point becomes **Red**.")
    
    st.divider()
    
    st.markdown("### 3. The Effect of 'K'")
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("#### Small K (e.g., K=1)")
        st.write("The boundary is very jagged and complex. It tries to capture every single outlier. This is called **Overfitting**.")
    with col_k2:
        st.markdown("#### Large K (e.g., K=20)")
        st.write("The boundary becomes smooth. It ignores small details. This is called **Underfitting**.")