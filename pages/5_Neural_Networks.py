import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Neural Networks - ML Lab",
    page_icon="🧠",
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
        white-space: pre-wrap;
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
    /* Theory container styling */
    .theory-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# --- NEURAL NETWORK CLASS ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize Weights (Random) and Biases (Zeros)
        np.random.seed(42)
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
        self.loss_history = []

    def forward(self, X):
        # Layer 1 (Input -> Hidden)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1) # Activation
        
        # Layer 2 (Hidden -> Output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        return self.output

    def backward(self, X, y, output):
        # 1. Calculate Error
        self.error = y - output
        self.loss = np.mean(np.square(self.error))
        self.loss_history.append(self.loss)
        
        # 2. Backpropagation (Chain Rule)
        d_output = self.error * sigmoid_derivative(output)
        
        error_hidden_layer = d_output.dot(self.W2.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.a1)
        
        # 3. Update Weights and Biases (Gradient Ascent logic here since error is y-output)
        self.W2 += self.a1.T.dot(d_output) * self.learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(d_hidden_layer) * self.learning_rate
        self.b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.learning_rate

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Network Config")
    dataset_type = st.selectbox("Dataset", ["Circles", "Moons", "Blobs"])
    hidden_neurons = st.slider("Hidden Neurons", 2, 10, 4)
    learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.1, format="%.3f")
    epochs = st.slider("Training Epochs", 100, 5000, 1000, step=100)
    
    if st.button("Initialize / Reset Model", type="primary", use_container_width=True):
        st.session_state['nn_model'] = None
        st.session_state['training_complete'] = False

# --- DATA GENERATION ---
@st.cache_data
def get_data(type_name):
    if type_name == "Circles":
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    elif type_name == "Moons":
        X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    else:
        X, y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=42)
    y = y.reshape(-1, 1)
    return X, y

X, y = get_data(dataset_type)

if 'nn_model' not in st.session_state or st.session_state['nn_model'] is None:
    st.session_state['nn_model'] = NeuralNetwork(input_size=2, hidden_size=hidden_neurons, output_size=1, learning_rate=learning_rate)
nn = st.session_state['nn_model']

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 Neural Network (MLP)</div>
    <div class="hero-subtitle">"A web of neurons that learns to bend space."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🛝 Playground", "📝 Theory & Math"])

# --- TAB 1: PLAYGROUND ---
with tab1:
    st.info("""
    **🎯 The Goal:** The network wants to classify the points by color. 
    It is predicting the probability that a specific point $(x_1, x_2)$ belongs to the **Blue Class**.
    * **Blue Region:** The network is confident the point is Blue (Probability ≈ 1.0).
    * **Red Region:** The network is confident the point is Red (Probability ≈ 0.0).
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        start_btn = st.button("▶️ Start Training", use_container_width=True)

    with col2:
        st.subheader("2. Decision Boundary")
        viz_placeholder = st.empty()

    if start_btn:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        for i in range(epochs):
            output = nn.forward(X)
            nn.backward(X, y, output)
            
            if i % (epochs // 10) == 0 or i == epochs - 1:
                progress_bar.progress((i + 1) / epochs)
                status_text.text(f"Epoch: {i}/{epochs} | Loss: {nn.loss:.4f}")
                
                # Loss Chart
                fig_loss, ax_loss = plt.subplots(figsize=(5, 2))
                ax_loss.plot(nn.loss_history, color='#ff4b4b')
                ax_loss.set_title("Loss (Error) over Time", fontsize=10)
                ax_loss.set_xlabel("Epochs", fontsize=8)
                ax_loss.set_ylabel("Error", fontsize=8)
                ax_loss.grid(True, alpha=0.3)
                chart_placeholder.pyplot(fig_loss)
                plt.close(fig_loss)

                # Boundary Viz
                fig_viz, ax_viz = plt.subplots(figsize=(5, 4))
                mesh_data = np.c_[xx.ravel(), yy.ravel()]
                Z = nn.forward(mesh_data)
                Z = Z.reshape(xx.shape)
                ax_viz.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
                ax_viz.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=40, cmap=plt.cm.RdBu, edgecolors='k')
                ax_viz.set_title(f"Boundary at Epoch {i}")
                viz_placeholder.pyplot(fig_viz)
                plt.close(fig_viz)

        st.success("Training Complete!")
    else:
        fig_viz, ax_viz = plt.subplots(figsize=(5, 4))
        ax_viz.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=40, cmap=plt.cm.RdBu, edgecolors='k')
        ax_viz.set_title("Raw Data")
        viz_placeholder.pyplot(fig_viz)

# --- TAB 2: DETAILED THEORY ---
with tab2:
    st.markdown("## 📖 The Architecture of Thought")
    
    st.markdown("""
    A Neural Network is inspired by the human brain, but mathematically, it's just a **Function Approximator**. 
    It takes an input $X$ and tries to find a function $f(X)$ that outputs the correct label $y$.
    """)
    
    # --- SECTION 1: THE NEURON ---
    st.markdown("### 1. The Artificial Neuron (Perceptron)")
    
    st.info("""
    Every neuron is a tiny decision-making unit. It does three simple steps:
    1. **Weigh the Input:** How important is this input? (Multiply by Weight $W$)
    2. **Add Bias:** Should I be active by default? (Add Bias $b$)
    3. **Decide:** Is the signal strong enough to fire? (Apply Activation Function $\sigma$)
    """)
    
    col_math, col_text = st.columns([1, 2])
    with col_math:
        st.latex(r"Z = W \cdot X + b")
        st.latex(r"A = \sigma(Z)")
    with col_text:
        st.write("**The Linear Part ($Z$):** This is exactly like Linear Regression ($y=mx+c$). It creates straight lines.")
        st.write("**The Non-Linear Part ($A$):** This allows the network to bend the lines. Without this, a Neural Network is just a fancy Linear Regression.")

    st.divider()

    # --- SECTION 2: ACTIVATION FUNCTIONS ---
    st.markdown("### 2. The Activation Function (Sigmoid)")
    st.write("Why do we need `sigmoid(x)`? Why not just pass the numbers through?")
    
    st.markdown("""
    Real-world data is **messy**. You can't separate two crescent moons with a straight ruler. 
    Activation functions introduce **Curvature**.
    """)
    
    st.latex(r"\sigma(x) = \frac{1}{1 + e^{-x}}")
    st.write("The Sigmoid function squashes any number (from $-\infty$ to $+\infty$) into a probability between **0 and 1**.")

    st.divider()

    # --- SECTION 3: LEARNING ---
    st.markdown("### 3. How it Learns (Backpropagation)")
    st.write("At first, the network guesses randomly. To get smarter, it follows this loop:")
    
    st.markdown("""
    #### Step A: Forward Pass (The Guess)
    The data flows from Input $\to$ Hidden Layer $\to$ Output. The network makes a prediction $\hat{y}$.
    
    #### Step B: Loss Calculation (The Scorecard)
    We compare the guess $\hat{y}$ to the actual answer $y$.
    """)
    st.latex(r"Loss = \frac{1}{n} \sum (y - \hat{y})^2")
    
    st.markdown("""
    #### Step C: Backward Pass (The Blame Game) 
    This is the magic. We use **Calculus (The Chain Rule)** to go backward through the network.
    We ask: *"Who is responsible for this error?"*
    
    * Did the output neuron mess up?
    * Did the hidden neuron give bad info?
    * Was the weight too high?
    
    We calculate the **Gradient** ($\nabla$), which tells us which direction to push the weights to reduce the error.
    
    #### Step D: Optimization (Gradient Descent) 
    We update the weights slightly in the opposite direction of the gradient.
    """)
    st.latex(r"W_{new} = W_{old} - \text{LearningRate} \times \nabla Loss")
    
    st.info("""
    **Analogy:** Imagine you are hiking down a mountain (the Error) in thick fog. You can't see the bottom.
    * **Gradient:** You feel the slope under your feet. It points *uphill*.
    * **Descent:** You take a step *downhill* (opposite to gradient).
    * **Learning Rate:** How big is your step? Too big, and you might stumble over the valley. Too small, and it takes forever.
    """)