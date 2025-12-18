import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import utils
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SVM - ML Lab",
    page_icon="⚔️",
    layout="wide"
)

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
    .theory-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
    }
    .math-box {
        background-color: #eef2f6;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER: DATA GEN ---
def generate_svm_data(n_samples, noise):
    # Generate 2 blobs
    X, y = make_blobs(n_samples=n_samples, centers=2, random_state=6, cluster_std=noise)
    # SVM requires labels to be -1 and 1 (not 0 and 1)
    y = np.where(y == 0, -1, 1)
    return X, y

# --- SVM CLASS (The Engine) ---
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
        self.loss_history = []

    def initialize(self, n_features):
        # Start with random weights
        self.w = np.random.randn(n_features)
        self.b = 0
        self.loss_history = []

    def train_step(self, X, y):
        # 1. Calculate Hinge Loss (Optional, for tracking)
        # Loss = max(0, 1 - y(wx - b)) + Regularization
        distances = 1 - y * (np.dot(X, self.w) - self.b)
        distances[distances < 0] = 0  # max(0, distance)
        hinge_loss = np.mean(distances) 
        self.loss_history.append(hinge_loss)

        # 2. Gradient Descent
        for idx, x_i in enumerate(X):
            # Check if point is "safe" (on the correct side of the margin)
            condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
            
            if condition:
                # If Safe: Only minimize weights (maximize margin)
                self.w -= self.lr * (2 * self.lambda_param * self.w)
            else:
                # If Violation: Minimize weights AND fix classification error
                self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                self.b -= self.lr * y[idx]

    def get_hyperplane(self, x, offset):
        """
        Returns the y-value of the hyperplane line for a given x.
        Equation: w0*x + w1*y - b = offset
        y = (offset + b - w0*x) / w1
        """
        return (offset + self.b - self.w[0] * x) / self.w[1]

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Generator")
    n_samples = st.slider("Samples", 20, 200, 50)
    noise = st.slider("Cluster Spread (Noise)", 0.5, 3.0, 1.2)
    
    if st.button("Generate New Data", type="primary") or 'svm_data' not in st.session_state:
        st.session_state['svm_data'] = generate_svm_data(n_samples, noise)

X, y = st.session_state['svm_data']

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚔️ Support Vector Machine (SVM)</div>
    <div class="hero-subtitle">"The art of drawing the widest possible street."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🛝 Playground", "📝 Theory & Math"])

# --- TAB 1: PLAYGROUND ---
with tab1:
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.subheader("1. Model Config")
        lr = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05], value=0.01)
        
        st.write("###")
        lambda_param = st.select_slider("Regularization (Lambda)", options=[0.001, 0.01, 0.1, 0.5], value=0.01)
        st.caption("Controls the 'Width' vs 'Errors' trade-off.")
        if lambda_param < 0.02:
            st.success("Hard Margin: Tries to classify everything perfectly (Strict).")
        else:
            st.warning("Soft Margin: Allows some errors to get a wider street.")
        
        epochs = st.slider("Training Steps", 100, 2000, 500)
        
        start_btn = st.button("▶️ Start Training", type="primary", use_container_width=True)
        
        st.write("###")
        progress_bar = st.progress(0)
        status_text = st.empty()

    with col_viz:
        plot_placeholder = st.empty()

    # --- TRAINING LOGIC ---
    if start_btn:
        svm = SVM(learning_rate=lr, lambda_param=lambda_param)
        svm.initialize(X.shape[1])
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        
        for i in range(epochs):
            svm.train_step(X, y)
            
            if i % (epochs // 20) == 0 or i == epochs - 1:
                progress_bar.progress((i + 1) / epochs)
                status_text.text(f"Step: {i}/{epochs}")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Scatter Data
                ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=60, edgecolors='k')
                
                # Plot Hyperplanes
                y_0 = svm.get_hyperplane(np.array([x_min, x_max]), 0) # Boundary
                y_pos = svm.get_hyperplane(np.array([x_min, x_max]), 1) # Margin +1
                y_neg = svm.get_hyperplane(np.array([x_min, x_max]), -1) # Margin -1
                
                ax.plot([x_min, x_max], y_0, 'k-', linewidth=2, label="Decision Boundary")
                ax.plot([x_min, x_max], y_pos, 'k--', linewidth=1, label="Margins")
                ax.plot([x_min, x_max], y_neg, 'k--', linewidth=1)
                
                ax.set_ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
                ax.legend()
                ax.set_title("Widening the Street...")
                ax.grid(True, linestyle='--', alpha=0.3)
                
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.01)

        st.session_state['svm_model'] = svm

    elif 'svm_model' in st.session_state:
        # Show final result
        svm = st.session_state['svm_model']
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=60, edgecolors='k')
        
        y_0 = svm.get_hyperplane(np.array([x_min, x_max]), 0)
        y_pos = svm.get_hyperplane(np.array([x_min, x_max]), 1)
        y_neg = svm.get_hyperplane(np.array([x_min, x_max]), -1)
        
        ax.plot([x_min, x_max], y_0, 'k-', linewidth=2, label="Decision Boundary")
        ax.plot([x_min, x_max], y_pos, 'k--', linewidth=1, label="Margins")
        ax.plot([x_min, x_max], y_neg, 'k--', linewidth=1)
        
        ax.set_ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        plot_placeholder.pyplot(fig)
        
    else:
        # Initial State
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=60, edgecolors='k')
        ax.set_title("Ready to Train")
        ax.grid(True, linestyle='--', alpha=0.3)
        plot_placeholder.pyplot(fig)

# --- TAB 2: THEORY ---
with tab2:
    st.markdown("## 🧠 Under the Hood: The Maximum Margin")
    
    st.markdown("""
    <div class="theory-box">
    <b>Analogy:</b> Imagine two warring villages (Blue vs Green). 
    SVM tries to build a road between them that is <b>as wide as possible</b> to prevent conflict.
    <br><br>
    The "Support Vectors" are the houses right on the edge of the road. 
    They are the only ones that matter! If you move the houses in the back, the road doesn't change.
    </div>
    """, unsafe_allow_html=True)
    
    # --- SECTION 1: GEOMETRY ---
    st.markdown("### 1. The Geometry of the Street")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**The Center Line (Decision Boundary)**")
        st.latex(r"w \cdot x - b = 0")
        st.write("This is the solid black line. Points here are 'neutral'.")
    with col2:
        st.markdown("**The Gutters (Margins)**")
        st.latex(r"w \cdot x - b = 1 \quad \text{and} \quad -1")
        st.write("These are the dashed lines. We want these to be far apart.")

    st.divider()

    # --- SECTION 2: OPTIMIZATION ---
    st.markdown("### 2. How do we make the street wider?")
    st.write("Here is the mind-bending part of SVM geometry:")
    
    st.markdown("""
    <div class="math-box">
    The width of the street is equal to: 
    $$ \text{Width} = \frac{2}{||w||} $$
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    To **Maximize Width**, we must **Minimize $||w||$** (the length of the weight vector).
    This is why we have `self.w -= ...` in the code. We are constantly trying to shrink the weights.
    """)

    st.divider()

    # --- SECTION 3: COST FUNCTION ---
    st.markdown("### 3. The Toll Booth (Hinge Loss)")
    st.write("The SVM drives a car down the road and checks every data point:")
    
    st.markdown("**The Rules:**")
    st.markdown("""
    1.  **Safe Zone:** If a point is on the correct side and off the road ($y(w \cdot x - b) \ge 1$), the **Cost is 0**.
    2.  **Danger Zone:** If a point is inside the street or on the wrong side, the **Cost increases** based on how bad the error is.
    """)
    
    st.latex(r"Loss = \max(0, 1 - y_i(w \cdot x_i - b))")
    
    st.info("""
    **Hard vs. Soft Margin (The Lambda Slider):**
    * **Low Lambda:** We care mostly about Hinge Loss (Classification). Result: Narrow street, but zero errors.
    * **High Lambda:** We care mostly about shrinking $||w||$ (Width). Result: Wide street, but we might run over a few points.
    """)