import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from models.kmeans import KMeans
import utils

st.set_page_config(page_title="K-Means Clustering", layout="wide")
utils.navbar()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .hero-container {
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f2937;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        font-style: italic;
    }
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #ffffff;
        border-radius: 8px;
        color: #4b5563;
        font-weight: 600;
        font-size: 16px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white !important;
        border-color: #ff4b4b;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✨ K-Means Clustering</div>
    <div class="hero-subtitle">"Finding patterns in chaos."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🎮 Playground", "📖 Theory & Math"])

# =========================================
# TAB 1: PLAYGROUND
# =========================================
with tab1:
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🛠️ Data Generator")
        n_samples = st.slider("Total Samples", 50, 500, 200)
        true_k = st.slider("True Clusters (Hidden)", 2, 6, 3)
        
        if st.button("🔄 Generate Unlabeled Data"):
            np.random.seed(int(time.time()))
            
            # Generate random centers for the "True" clusters
            centers = np.random.uniform(-8, 8, size=(true_k, 2))
            
            # Generate points around those centers
            X_list = []
            for c in centers:
                # Random count for each cluster
                count = n_samples // true_k
                # Random spread
                points = np.random.randn(count, 2) + c
                X_list.append(points)
            
            X = np.vstack(X_list)
            st.session_state['kmeans_X'] = X
            st.session_state['kmeans_trained'] = False

    # Default Data
    if 'kmeans_X' not in st.session_state:
        # Default: 3 clusters
        c1 = np.random.randn(50, 2) + [-5, -5]
        c2 = np.random.randn(50, 2) + [0, 5]
        c3 = np.random.randn(50, 2) + [5, -2]
        st.session_state['kmeans_X'] = np.vstack((c1, c2, c3))

    X = st.session_state['kmeans_X']

    # --- CONTROLS ---
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        k_user = st.number_input("How many clusters (K)?", 2, 8, 3)
    with col2:
        st.write("##") # Spacer
        run_btn = st.button("🚀 Find Clusters", type="primary")

    chart_placeholder = st.empty()

    # --- PLOT HELPER ---
    def plot_kmeans_step(X, centroids, labels, iteration, title):
        fig = go.Figure()
        
        # 1. Plot Data Points (Colored by assigned cluster)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        
        if labels is None:
            # Show all as gray if not yet assigned
            fig.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode='markers', marker=dict(color='lightgray', size=8), name='Unlabeled Data'))
        else:
            for i in range(len(centroids)):
                mask = (labels == i)
                fig.add_trace(go.Scatter(
                    x=X[mask, 0], y=X[mask, 1], 
                    mode='markers', 
                    marker=dict(color=colors[i % len(colors)], size=8),
                    name=f'Cluster {i}'
                ))
        
       
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1],
            mode='markers',
            marker=dict(
                symbol='circle',       # Changed from 'x' to 'circle' so it can be filled
                size=15, 
                color='white',         # Fill color
                line=dict(color='black', width=2) # Border color
            ),
            name='Centroids'
        ))

        fig.update_layout(
            title=f"{title}",
            xaxis_title="Feature 1", yaxis_title="Feature 2",
            height=600, template="plotly_white",
            xaxis=dict(range=[-12, 12]), yaxis=dict(range=[-12, 12]) # Fix axes to prevent jumping
        )
        return fig

    # --- RUN LOGIC ---
    if run_btn:
        model = KMeans(k=k_user)
        model.fit(X)
        st.session_state['kmeans_trained'] = True
        
        # ANIMATION LOOP
        # We replay the history saved in the model
        for i, historic_centroids in enumerate(model.history):
            # We must re-calculate labels for this specific frame to color correctly
            labels = model._assign_clusters(X, historic_centroids)
            
            fig = plot_kmeans_step(X, historic_centroids, labels, i, f"Iteration {i}: Moving Centroids...")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.5) # Pause to create animation effect
            
        # Final State
        st.success(f"Converged in {len(model.history)} iterations!")

    elif not st.session_state.get('kmeans_trained', False):
        # Show raw data (Random centroids just for visual placeholder)
        random_centroids = X[np.random.choice(X.shape[0], k_user, replace=False)]
        fig = plot_kmeans_step(X, random_centroids, None, 0, "Raw Unlabeled Data")
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 2: THEORY
# =========================================
with tab2:
    st.markdown("## 🧠 How K-Means Clustering Works")
    st.write("K-Means is an **Unsupervised Learning** algorithm. Unlike the previous models, it has no 'Teacher' (Labels). It must find structure in the data on its own.")
    
    st.divider()

    st.markdown("### 1. Initialization (The Random Start)")
    st.write("The algorithm begins by guessing. It randomly picks $K$ spots on the map to be the 'Centroids' (Cluster Centers).")
    st.info("⚠️ **The Butterfly Effect:** The final result depends heavily on these starting spots. If you start in a bad place, you might get a bad result! This is why we sometimes run K-Means multiple times.")

    st.divider()

    st.markdown("### 2. The Loop (Expectation-Maximization)")
    st.write("Once the centroids are placed, the model enters a loop of two repeating steps until it stabilizes.")
    
    st.markdown("#### Step A: Assignment (The 'E' Step)")
    st.write("Every single data point looks at the $K$ centroids and asks: *'Who is closest to me?'*")
    st.latex(r"Cluster(x) = \arg\min_j || x - \mu_j ||^2")
    st.caption("Each point $x$ is assigned to the cluster $j$ that minimizes the squared distance.")
    
    st.markdown("#### Step B: Update (The 'M' Step)")
    st.write("Now that every point has joined a team, the Centroid moves to the **center of gravity** of its team.")
    st.latex(r"\mu_j = \frac{1}{|C_j|} \sum_{x \in C_j} x")
    st.caption("The new position $\mu_j$ is simply the average (mean) of all points in that cluster.")

    st.divider()

    st.markdown("### 3. Convergence (The End)")
    st.write("The loop repeats: Assign -> Move -> Assign -> Move.")
    st.write("Eventually, the Centroids stop moving because they have found the optimal center of their groups. We call this **Convergence**.")
    
    st.success("""
    **Summary of the Algorithm:**
    1. **Pick** $K$ random points.
    2. **Assign** every dot to the nearest point.
    3. **Move** the point to the center of its dots.
    4. **Repeat** until movement stops.
    """)