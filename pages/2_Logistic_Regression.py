import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from models.logistic_regression import LogisticRegression
import utils

st.set_page_config(page_title="Logistic Regression", layout="wide")
utils.navbar()
st.title("🧬 Logistic Regression: The Classifier")

# --- TABS ---
tab1, tab2 = st.tabs(["🎮 Playground", "📖 Theory & Math"])

# =========================================
# TAB 1: THE INTERACTIVE PLAYGROUND
# =========================================
with tab1:
    # --- SIDEBAR MOVED HERE ---
    with st.sidebar:
        st.header("🛠️ Create Tumor Data")
        n_samples = st.slider("Samples", 50, 500, 200)
        noise = st.slider("Overlap (Noise)", 0.0, 2.0, 0.5) # Increased noise range for bigger scale
        
        if st.button("🔄 Generate Patient Data"):
            np.random.seed(int(time.time()))
            half_samples = n_samples // 2
            
            # --- REALISTIC SHIFT APPLIED HERE ---
            # Class 0 (Benign): Centered at 3.0 (Small/Low Density)
            X0 = np.random.randn(half_samples, 2) + 3
            y0 = np.zeros((half_samples, 1))
            
            # Class 1 (Malignant): Centered at 8.0 (Large/High Density)
            X1 = np.random.randn(half_samples, 2) + 8
            y1 = np.ones((half_samples, 1))
            
            X = np.vstack((X0, X1))
            y = np.vstack((y0, y1))
            
            # Add Noise (Safe because centers are far from 0)
            X += np.random.randn(X.shape[0], 2) * noise
            
            st.session_state['log_X'] = X
            st.session_state['log_y'] = y
            st.session_state['log_trained'] = False

    # Initialize Default Data (Also Shifted)
    if 'log_X' not in st.session_state:
        # Default start: Class 0 at 3, Class 1 at 8
        X0 = np.random.randn(50, 2) + 3
        X1 = np.random.randn(50, 2) + 8
        st.session_state['log_X'] = np.vstack((X0, X1))
        st.session_state['log_y'] = np.vstack((np.zeros((50,1)), np.ones((50,1))))

    X = st.session_state['log_X']
    y = st.session_state['log_y']

    # --- CONTROLS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        lr = st.number_input("Learning Rate", value=0.1, step=0.01)
    with col2:
        epochs = st.slider("Epochs", 50, 1000, 300)
    with col3:
        st.write("##")
        train_btn = st.button("🚀 Train Diagnosis Model", type="primary")

    chart_placeholder = st.empty()

    # --- PLOT HELPER ---
    def plot_decision_boundary(X, y, w, b, epoch, cost):
        fig = go.Figure()
        
        mask = y.flatten() == 0
        fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers', name='Benign (0)', marker=dict(color='blue', size=8, opacity=0.6)))
        fig.add_trace(go.Scatter(x=X[~mask, 0], y=X[~mask, 1], mode='markers', name='Malignant (1)', marker=dict(color='red', size=8, opacity=0.6)))

        # Adjust axes to fit the new positive data
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_vals = np.linspace(x1_min, x1_max, 100)
        
        if w[1] != 0:
            x2_vals = -(w[0] * x1_vals + b) / w[1]
            fig.add_trace(go.Scatter(x=x1_vals, y=x2_vals, mode='lines', name='Boundary', line=dict(color='green', width=3, dash='dash')))

        fig.update_layout(
            title=f"Epoch {epoch} | Log Loss: {cost:.4f}",
            xaxis_title="Tumor Size (cm)", yaxis_title="Tumor Density (Index)",
            template="plotly_white", height=600
        )
        return fig

    # --- TRAINING LOOP ---
    if train_btn:
        try:
            model = LogisticRegression(learning_rate=lr, n_iterations=epochs)
            model.fit(X, y)
            st.session_state['log_trained'] = True
            st.session_state['log_model'] = model
            
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
            y_pred = model.predict(X)
            acc = utils.accuracy_score(y, y_pred)
            st.success(f"Training Complete! Final Accuracy: {acc*100:.2f}%")
            
        except Exception as e:
            st.error(f"⚠️ Math Error: {e}")

    # --- DOCTOR'S OFFICE ---
    if st.session_state.get('log_trained'):
        model = st.session_state['log_model']
        
        st.divider()
        st.subheader("🩺 Doctor's Office: Test a Patient")
        st.info("Enter the patient's tumor measurements below:")
        
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            # Defaults now adapt to the new "Shifted" Mean
            feat1 = st.number_input("Tumor Size (cm)", value=float(X[:,0].mean()), format="%.2f")
        with p_col2:
            feat2 = st.number_input("Tumor Density (1-10)", value=float(X[:,1].mean()), format="%.2f")

        patient_data = np.array([[feat1, feat2]])
        linear_out = np.dot(patient_data, model.weights) + model.bias
        probability = model.sigmoid(linear_out)[0][0]
        prediction = 1 if probability > 0.5 else 0
        
        st.write("---")
        if prediction == 1:
            st.error(f"**Diagnosis: MALIGNANT** (Confidence: {probability:.2%})")
            st.write("⚠️ High risk detected. Recommend biopsy.")
        else:
            st.success(f"**Diagnosis: BENIGN** (Confidence: {1 - probability:.2%})")
            st.write("✅ Patient looks healthy.")

    elif not st.session_state.get('log_trained', False):
        fig = plot_decision_boundary(X, y, np.array([[0],[1]]), 0, 0, 0)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

# =========================================
# TAB 2: THEORY & MATH (FIXED)
# =========================================
with tab2:
    st.markdown("## 🧠 The Math of Classification")
    
    st.markdown("### 1. The Sigmoid Function (Squashing)")
    st.write("Linear Regression gives us numbers from $-\infty$ to $+\infty$. But for probability, we need **0 to 1**.")
    
    # 

    

    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    
    st.write("This function takes any number 'z' and squashes it.")
    # We use r"" strings here to fix the \rightarrow and \approx errors
    st.info(r"""
    * If $z$ is huge positive $\rightarrow \sigma(z) \approx 1$ (Malignant)
    * If $z$ is huge negative $\rightarrow \sigma(z) \approx 0$ (Benign)
    """)
    
    st.divider()

    st.markdown("### 2. The Decision Boundary")
    st.write("We draw a line where the probability is exactly **50% (0.5)**. This happens when $z = 0$.")
    
    # 
    st.latex(r"w_1 x_1 + w_2 x_2 + b = 0")
    
    st.write("Rearranging for $x_2$ (Equation of a line $y=mx+c$):")
    st.latex(r"x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}")
    st.caption("This is why the green boundary line is straight!")

    st.divider()

    st.markdown("### 3. Log Loss (Binary Cross Entropy)")
    st.write("We cannot use MSE because the Sigmoid function makes the error surface 'wavy'. We use **Log Loss**:")
    
    # 
    st.latex(r"J = - \frac{1}{n} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]")
    
    st.warning(r"""
    * If actual class is **1** and model predicts **0.9** (Confident Correct) $\rightarrow$ Low Cost.
    * If actual class is **1** and model predicts **0.1** (Confident Wrong) $\rightarrow$ **HUGE Cost**.
    """)