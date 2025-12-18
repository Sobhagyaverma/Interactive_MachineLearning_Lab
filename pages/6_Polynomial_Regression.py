import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import utils
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Polynomial Regression - ML Lab",
    page_icon="📈",
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
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    .theory-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
    }
    .underfit { background-color: #fff3cd; border-left: 5px solid #ffc107; color: #856404; }
    .goodfit { background-color: #d4edda; border-left: 5px solid #28a745; color: #155724; }
    .overfit { background-color: #f8d7da; border-left: 5px solid #dc3545; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def generate_nonlinear_data(n_samples, noise, type='Quadratic'):
    np.random.seed(42)
    X = 6 * np.random.rand(n_samples, 1) - 3  # Range -3 to 3
    X = np.sort(X, axis=0)
    
    if type == 'Quadratic (Parabola)':
        y_true = 0.5 * X**2 + X + 2
    elif type == 'Cubic (S-Curve)':
        y_true = 0.5 * X**3 - X**2 + 2
    elif type == 'W-Shape (Quartic)':
        y_true = 0.1 * X**4 - X**2 + 5
    elif type == 'Sine Wave':
        y_true = np.sin(X) * 2 + 2
    
    y = y_true + np.random.randn(n_samples, 1) * noise
    return X, y, y_true

# --- POLYNOMIAL REGRESSION CLASS ---
class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.lr = learning_rate
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.loss_history = []

    def _create_poly_features(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly

    def initialize(self, X):
        self.X_poly_raw = self._create_poly_features(X)
        self.mean = np.mean(self.X_poly_raw, axis=0)
        self.std = np.std(self.X_poly_raw, axis=0)
        self.X_poly = (self.X_poly_raw - self.mean) / (self.std + 1e-8)
        
        n_features = self.X_poly.shape[1]
        self.weights = np.random.randn(n_features, 1) * 0.1 
        self.bias = 0
        self.loss_history = []
        
    def train_step(self, y):
        n_samples = self.X_poly.shape[0]
        y_pred = np.dot(self.X_poly, self.weights) + self.bias
        error = y_pred - y
        mse = np.mean(error**2)
        self.loss_history.append(mse)
        
        dw = (1/n_samples) * np.dot(self.X_poly.T, error)
        db = (1/n_samples) * np.sum(error)
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def predict(self, X):
        if self.weights is None: return None
        X_poly = self._create_poly_features(X)
        X_poly = (X_poly - self.mean) / (self.std + 1e-8)
        return np.dot(X_poly, self.weights) + self.bias

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Generator")
    data_type = st.selectbox("Data Shape", ["Quadratic (Parabola)", "Cubic (S-Curve)", "W-Shape (Quartic)", "Sine Wave"])
    noise = st.slider("Noise Level", 0.0, 3.0, 1.0)
    n_samples = st.slider("Samples", 20, 200, 50)
    
    if st.button("Generate New Data", type="primary") or 'poly_data' not in st.session_state:
        st.session_state['poly_data'] = generate_nonlinear_data(n_samples, noise, data_type)
        st.session_state['data_type_name'] = data_type 

X, y, y_true = st.session_state['poly_data']
current_shape = st.session_state.get('data_type_name', 'Quadratic (Parabola)')

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📈 Polynomial Regression</div>
    <div class="hero-subtitle">"Watch the machine learn the curve."</div>
</div>
""", unsafe_allow_html=True)

# --- MAIN TABS ---
tab1, tab2 = st.tabs(["🛝 Playground", "📝 Theory & Math"])

# --- TAB 1: PLAYGROUND ---
with tab1:
    col_controls, col_viz = st.columns([1, 2])

    with col_controls:
        st.subheader("1. Config & Diagnosis")
        degree = st.slider("Polynomial Degree", 1, 15, 2)
        lr = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1, 0.5], value=0.05)
        epochs = st.slider("Training Epochs", 100, 2000, 500)
        
        # Diagnosis Logic
        is_complex_shape = current_shape in ["Cubic (S-Curve)", "W-Shape (Quartic)", "Sine Wave"]
        if degree == 1:
            status_class = "underfit"
            status_title = "⚠️ Underfitting"
            status_desc = "Model is too simple (Line). It can't bend."
        elif degree > 10:
            status_class = "overfit"
            status_title = "⚠️ Overfitting"
            status_desc = "Model is too complex. It chases noise."
        elif (is_complex_shape and degree < 3):
             status_class = "underfit"
             status_title = "⚠️ Underfitting"
             status_desc = "Need higher degree for this shape."
        else:
            status_class = "goodfit"
            status_title = "✅ Good Fit"
            status_desc = "Complexity matches the data pattern."

        st.markdown(f"""
        <div class="status-box {status_class}">
            <strong>{status_title}</strong><br>
            {status_desc}
        </div>
        """, unsafe_allow_html=True)
        
        start_btn = st.button("▶️ Start Training", type="primary", use_container_width=True)
        
        st.write("###")
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart_placeholder = st.empty()

    with col_viz:
        plot_placeholder = st.empty()

    # --- TRAINING LOOP ---
    if start_btn:
        model = PolynomialRegression(degree, lr)
        model.initialize(X)
        X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        
        for i in range(epochs):
            model.train_step(y)
            if i % (epochs // 20) == 0 or i == epochs - 1:
                progress_bar.progress((i + 1) / epochs)
                status_text.text(f"Epoch: {i}/{epochs} | MSE Error: {model.loss_history[-1]:.4f}")
                
                fig_loss, ax_loss = plt.subplots(figsize=(5, 2))
                ax_loss.plot(model.loss_history, color='#ff4b4b')
                ax_loss.set_title("Learning Curve")
                ax_loss.grid(True, alpha=0.3)
                loss_chart_placeholder.pyplot(fig_loss)
                plt.close(fig_loss)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(X, y, color='dodgerblue', alpha=0.5, label='Noisy Data')
                ax.plot(X, y_true, color='green', linestyle='--', linewidth=2, label='True Pattern', alpha=0.5)
                y_pred_line = model.predict(X_range)
                ax.plot(X_range, y_pred_line, color='#ff4b4b', linewidth=3, label=f'Model (Degree {degree})')
                ax.set_title(f"Training... Epoch {i}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                time.sleep(0.01)

        st.session_state['poly_model'] = model

    elif 'poly_model' in st.session_state:
        model = st.session_state['poly_model']
        X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred_line = model.predict(X_range)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, y, color='dodgerblue', alpha=0.5, label='Noisy Data')
        ax.plot(X, y_true, color='green', linestyle='--', linewidth=2, label='True Pattern', alpha=0.5)
        ax.plot(X_range, y_pred_line, color='#ff4b4b', linewidth=3, label=f'Model (Degree {degree})')
        ax.set_title(f"Final Result (MSE: {model.loss_history[-1]:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_placeholder.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X, y, color='dodgerblue', alpha=0.5, label='Noisy Data')
        ax.plot(X, y_true, color='green', linestyle='--', linewidth=2, label='True Pattern', alpha=0.5)
        ax.set_title("Ready to Train")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_placeholder.pyplot(fig)

# --- TAB 2: DETAILED THEORY ---
with tab2:
    st.markdown("## 🧠 Under the Hood: The Polynomial Trick")
    
    st.info("""
    Polynomial Regression is a master of disguise. Mathematically, **it is still Linear Regression**. 
    It just cheats by creating "fake" features.
    """)
    
    st.markdown("### 1. The 'Hack' (Feature Engineering)")
    st.write("""
    Imagine you have a single input $x$ (e.g., Temperature). 
    A linear model can only learn a straight line:
    """)
    st.latex(r"y = w_1 \cdot x + b")
    
    st.write("""
    But what if the relationship is curved? We can't change the algorithm (Linear Regression is stubborn), 
    so we **change the data**. We create new "imaginary" features by squaring or cubing $x$.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original Data**")
        st.code("[ 2 ]")
        st.caption("1 Feature: x")
    with col2:
        st.markdown("**Transformed Data (Degree 3)**")
        st.code("[ 2, 4, 8 ]")
        st.caption("3 Features: x, x², x³")
        
    st.write("Now the Linear Regression sees 3 features and learns a weight for each:")
    st.latex(r"y = w_1 (x) + w_2 (x^2) + w_3 (x^3) + b")
    
    st.divider()
    
    st.markdown("### 2. The Bias-Variance Tradeoff")
    st.write("This is one of the most important concepts in Machine Learning. You saw it in the playground:")

    # Bias
    st.markdown("#### 🔴 High Bias (Underfitting)")
    st.markdown("""
    * **What it is:** The model is **too simple**. It assumes the world is a straight line.
    * **Visual:** The model ignores the curve and cuts straight through.
    * **Cause:** Degree is too low (e.g., Degree 1 on a Parabola). 
    """)

    # Variance
    st.markdown("#### 🔵 High Variance (Overfitting)")
    st.markdown("""
    * **What it is:** The model is **too sensitive**. It tries to connect every single dot, including the noise.
    * **Visual:** The line wiggles aggressively. It hits the training dots perfectly but misses the "True Pattern" (Green Line).
    * **Cause:** Degree is too high (e.g., Degree 15). The model has too much freedom. 
    """)
    
    st.info("""
    **The Sweet Spot:** We want a model that is complex enough to capture the curve (Low Bias) 
    but simple enough to ignore the noise (Low Variance).
    """)

    st.divider()

    st.markdown("### 3. Why Feature Scaling?")
    st.write("You might notice `self.mean` and `self.std` in the code. Why?")
    st.code("X_poly = (X_poly - self.mean) / self.std")
    
    st.write("""
    If $x = 100$, then $x^2 = 10,000$ and $x^3 = 1,000,000$.
    Without scaling, the numbers get too huge, and Gradient Descent crashes (exploding gradients). 
    **Normalization** keeps everything small and manageable.
    """)