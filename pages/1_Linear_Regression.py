import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from models.linear_regression import LinearRegression
import utils

st.set_page_config(page_title="Linear Regression", layout="wide")
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 4px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📏 Linear Regression</div>
    <div class="hero-subtitle">"The foundation: fitting a line through the noise."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS FOR NAVIGATION ---
tab1, tab2 = st.tabs(["🎮 Playground", "📖 Theory & Math"])

# =========================================
# TAB 1: THE INTERACTIVE PLAYGROUND
# =========================================
with tab1:
    # --- HELPER FUNCTIONS ---
    def normalize(data):
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        if range_val == 0: range_val = 1
        return (data - min_val) / range_val, min_val, range_val

    def denormalize(norm_data, min_val, range_val):
        return norm_data * range_val + min_val

    def compute_loss_surface(X, y, w_history, b_history):
        w_min, w_max = min(w_history), max(w_history)
        b_min, b_max = min(b_history), max(b_history)
        padding = 0.5
        w_range = np.linspace(w_min - padding, w_max + padding, 50)
        b_range = np.linspace(b_min - padding, b_max + padding, 50)
        W, B = np.meshgrid(w_range, b_range)
        Z = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w_val = W[i, j]
                b_val = B[i, j]
                y_pred = w_val * X + b_val
                Z[i, j] = np.mean((y_pred - y) ** 2)
        return W, B, Z

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🛠️ Create Housing Market")
        n_samples = st.slider("Number of Houses", 10, 500, 100)
        base_price = st.slider("Base Price", 50000, 200000, 100000, step=5000)
        price_per_sqft = st.slider("Price per Sq Ft", 100, 500, 300)
        market_noise = st.slider("Market Noise", 0, 50000, 15000, step=1000)
        
        if st.button("🔄 Generate Market"):
            np.random.seed(int(time.time()))
            raw_X = np.random.randint(500, 3500, (n_samples, 1)).astype(float)
            noise = np.random.randn(n_samples, 1) * market_noise
            raw_y = (price_per_sqft * raw_X) + base_price + noise
            st.session_state['raw_X'] = raw_X
            st.session_state['raw_y'] = raw_y
            st.session_state['trained'] = False

    if 'raw_X' not in st.session_state:
        st.session_state['raw_X'] = np.random.randint(500, 3500, (50, 1)).astype(float)
        st.session_state['raw_y'] = (300 * st.session_state['raw_X']) + 100000 + (np.random.randn(50, 1) * 20000)

    raw_X = st.session_state['raw_X']
    raw_y = st.session_state['raw_y']
    X_norm, X_min, X_range = normalize(raw_X)
    y_norm, y_min, y_range = normalize(raw_y)

    # --- CONTROLS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        lr = st.number_input("Learning Rate (Step Size)", value=0.1, step=0.01, format="%.4f")
    with col2:
        epochs = st.slider("Training Epochs", 50, 500, 100)
    with col3:
        st.write("##")
        train_btn = st.button("🚀 Train Model", type="primary")

    viz_col1, viz_col2 = st.columns([1, 1])
    chart_placeholder = viz_col1.empty()
    contour_placeholder = viz_col2.empty()

    if train_btn:
        model = LinearRegression(learning_rate=lr, n_iterations=epochs)
        model.fit(X_norm, y_norm)
        st.session_state['trained'] = True
        st.session_state['model'] = model
        
        w_history = [step[0][0][0] for step in model.train_history]
        b_history = [step[1] for step in model.train_history]
        W, B, Z = compute_loss_surface(X_norm, y_norm, w_history, b_history)
        
        progress_bar = st.progress(0)
        skip = max(1, epochs // 20)
        
        for i in range(0, epochs, skip):
            curr_w, curr_b = w_history[i], b_history[i]
            curr_cost = model.history[i]
            
            # Real Units Plot
            x_line_norm = np.linspace(0, 1, 100).reshape(-1, 1)
            y_line_norm = curr_w * x_line_norm + curr_b
            x_line_real = denormalize(x_line_norm, X_min, X_range)
            y_line_real = denormalize(y_line_norm, y_min, y_range)
            
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=raw_X.flatten(), y=raw_y.flatten(), mode='markers', name='Data', marker=dict(color='cyan', opacity=0.5)))
            fig_reg.add_trace(go.Scatter(x=x_line_real.flatten(), y=y_line_real.flatten(), mode='lines', name='Prediction', line=dict(color='red', width=3)))
            fig_reg.update_layout(title=f"House Prices (Epoch {i})", xaxis_title="Sq Ft", yaxis_title="Price ($)", template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0))
            chart_placeholder.plotly_chart(fig_reg, use_container_width=True)

            # Contour Plot
            fig_cont = go.Figure()
            fig_cont.add_trace(go.Contour(z=Z, x=W[0], y=B[:, 0], colorscale='Viridis', contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z)-np.min(Z))/15)))
            fig_cont.add_trace(go.Scatter(x=w_history[:i+1], y=b_history[:i+1], mode='lines', line=dict(color='white', width=2)))
            fig_cont.add_trace(go.Scatter(x=[curr_w], y=[curr_b], mode='markers', marker=dict(color='red', size=10)))
            fig_cont.update_layout(title=f"Gradient Descent (Cost: {curr_cost:.4f})", xaxis_title="Weight", yaxis_title="Bias", template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0))
            contour_placeholder.plotly_chart(fig_cont, use_container_width=True)
            
            time.sleep(0.02)
            progress_bar.progress((i+1)/epochs)
        progress_bar.empty()

    if st.session_state.get('trained'):
        model = st.session_state['model']
        y_pred_norm = model.predict(X_norm)
        final_r2 = utils.r2_score(y_norm, y_pred_norm)
        
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Final Accuracy (R²)", f"{final_r2:.4f}")
        m2.metric("Final Loss (MSE)", f"{utils.mean_squared_error(y_norm, y_pred_norm):.5f}")
        
        st.subheader("💰 Estimate House Price")
        col_in, col_out = st.columns(2)
        with col_in:
            user_sqft = st.number_input("House Size (Sq Ft)", value=2000, step=100)
        with col_out:
            user_norm = (user_sqft - X_min) / X_range
            pred_norm = model.predict(np.array([[user_norm]]))
            price = denormalize(pred_norm, y_min, y_range)[0][0]
            st.success(f"Predicted Price: **${price:,.2f}**")

    elif not st.session_state.get('trained', False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw_X.flatten(), y=raw_y.flatten(), mode='markers', marker=dict(color='cyan')))
        fig.update_layout(title="Raw Data", xaxis_title="Sq Ft", yaxis_title="Price", template="plotly_dark", height=400)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 2: THEORY & MATH
# =========================================
with tab2:
    st.markdown("## 🧠 The Math Behind the Magic")
    
    st.markdown("### 1. The Goal (Linear Equation)")
    st.write("We are trying to find the best-fitting line through the data points. A line is defined by:")
    st.latex(r"y = w \cdot x + b")
    st.write("""
    * $y$: The prediction (House Price)
    * $x$: The input (Square Footage)
    * $w$: The **Weight** (Slope) - How much price increases per sq ft.
    * $b$: The **Bias** (Intercept) - The base price of a house (even with 0 sq ft).
    """)
    st.divider()

    st.markdown("### 2. The Cost Function (MSE)")
    st.write("How do we know if our line is 'good'? We calculate the **Error** (distance between the line and the dots).")
    st.latex(r"J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_{pred}^{(i)} - y_{actual}^{(i)})^2")
    st.write("""
    This is called **Mean Squared Error (MSE)**.
    * If the line touches all dots, MSE = 0 (Perfect).
    * If the line is far away, MSE is huge.
    * **Our Goal:** Minimize $J(w, b)$.
    """)
    st.divider()

    st.markdown("### 3. Gradient Descent (The Optimization)")
    st.write("We start with random $w$ and $b$, and we take small steps 'downhill' to reduce the error.")
    
    col_theory_1, col_theory_2 = st.columns(2)
    with col_theory_1:
        st.markdown("**Update Rule (How we move):**")
        st.latex(r"w_{new} = w_{old} - \ alpha \ cdot \ frac{\ partial J}{\ partial w}")
        st.latex(r"b_{new} = b_{old} - \ alpha \ cdot \ frac{\ partial J}{\ partial b}")
    with col_theory_2:
        st.info("""
        * $\ alpha$ (Alpha): **Learning Rate**.
        * $\ frac{\partial J}{\partial w}$: **Gradient** (The slope of the hill).
        """)
    
    st.markdown("""
    > **Analogy:** Imagine being on a mountain in the dark (high error). You feel the slope with your feet and take a step downwards. You repeat this until you reach the valley (minimum error).
    """)
    st.divider()

    st.markdown("### 4. The Learning Rate ($\ alpha$)")
    st.write("The **Learning Rate** controls how big of a step we take.")
    
    col_lr1, col_lr2, col_lr3 = st.columns(3)
    with col_lr1:
        st.markdown("#### Too Small 🐢")
        st.write("Steps are tiny. It takes forever to reach the bottom.")
    with col_lr2:
        st.markdown("#### Just Right ✅")
        st.write("Steps are efficient. We reach the minimum quickly.")
    with col_lr3:
        st.markdown("#### Too Big 💥")
        st.write("We overshoot the bottom and might even go *up* the other side (Exploding Gradient).")