import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Naive Bayes - ML Lab",
    page_icon="🎲",
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
</style>
""", unsafe_allow_html=True)

# --- CLASS: GAUSSIAN NAIVE BAYES ---
class GaussianNB_Scratch:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for c in self.classes:
            # Filter data for this class
            X_c = X[y == c]
            
            # Calculate Mean & Variance (The "Shape" of the bell curve)
            self.mean[c] = np.mean(X_c)
            self.var[c] = np.var(X_c)
            
            # Prior Probability P(Class)
            self.priors[c] = len(X_c) / n_samples

    def get_probability(self, x, c):
        """
        Calculates P(x | c) using Gaussian PDF Formula.
        """
        mean = self.mean[c]
        var = self.var[c]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, x):
        posteriors = []
        for c in self.classes:
            prior = self.priors[c] # P(Class)
            likelihood = self.get_probability(x, c) # P(Data | Class)
            posteriors.append(prior * likelihood)
            
        # Normalize so they sum to 1
        return np.array(posteriors) / sum(posteriors)

# --- HELPER: DATA GEN ---
def generate_1d_data(n_samples, separation):
    np.random.seed(42)
    # Class 0: Centered at 2
    c0 = np.random.normal(loc=2.0, scale=1.0, size=n_samples)
    # Class 1: Centered at 2 + separation
    c1 = np.random.normal(loc=2.0 + separation, scale=1.0, size=n_samples)
    
    X = np.concatenate([c0, c1])
    y = np.array([0]*n_samples + [1]*n_samples)
    return X, y

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Generator")
    separation = st.slider("Separation (Difficulty)", 0.5, 5.0, 2.0)
    st.caption("Low Separation = The fruits are similar sizes (Hard). High Separation = Very different sizes (Easy).")
    
    if st.button("Generate New Data", type="primary") or 'nb_data' not in st.session_state:
        X, y = generate_1d_data(100, separation)
        st.session_state['nb_data'] = (X, y)

X, y = st.session_state['nb_data']

# Train Model Immediately (It's fast)
model = GaussianNB_Scratch()
model.fit(X, y)

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🎲 Naive Bayes</div>
    <div class="hero-subtitle">"Calculating the odds using Bell Curves."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🛝 Playground", "📝 Theory & Math"])

with tab1:
    # --- CONTEXT BLOCK ---
    st.info("""
    **🍎 The Scenario (Apples vs. Oranges):**
    Imagine we measured the size of 100 Apples (Blue) and 100 Oranges (Red).
    * **The Histogram:** Shows our real data.
    * **The Bell Curve:** The "Model" summarizing what a typical Apple or Orange looks like.
    
    **🎮 Your Mission:**
    Use the slider below to simulate finding a **Mystery Fruit**. 
    Based *only* on its size, the model will calculate the odds: **"Is this an Apple or an Orange?"**
    """)
    
    col_viz, col_info = st.columns([2, 1])
    
    with col_viz:
        st.subheader("1. Interactive Probability")
        
        # User Interaction: Pick a point X
        x_input = st.slider("Step 1: Measure the Mystery Fruit (Size X)", float(X.min()), float(X.max()), 3.0)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Histograms (Raw Data)
        ax.hist(X[y==0], bins=15, density=True, alpha=0.3, color='blue', label='Apples (Class 0)')
        ax.hist(X[y==1], bins=15, density=True, alpha=0.3, color='red', label='Oranges (Class 1)')
        
        # Plot Bell Curves (The "Model")
        x_range = np.linspace(X.min()-1, X.max()+1, 200)
        
        # Curve for Class 0
        pdf0 = stats.norm.pdf(x_range, model.mean[0], np.sqrt(model.var[0]))
        ax.plot(x_range, pdf0, 'b-', lw=2, label='Apple Model (Gaussian)')
        
        # Curve for Class 1
        pdf1 = stats.norm.pdf(x_range, model.mean[1], np.sqrt(model.var[1]))
        ax.plot(x_range, pdf1, 'r-', lw=2, label='Orange Model (Gaussian)')
        
        # Plot User Line
        ax.axvline(x_input, color='black', linestyle='--', linewidth=3, label=f'Your Fruit Size: {x_input:.2f}')
        
        # Visual Intersection Points
        prob_c0 = model.get_probability(x_input, 0)
        prob_c1 = model.get_probability(x_input, 1)
        ax.scatter([x_input], [prob_c0], color='blue', s=100, zorder=5)
        ax.scatter([x_input], [prob_c1], color='red', s=100, zorder=5)
        
        ax.set_title("Probability Density (Likelihood)")
        ax.legend()
        st.pyplot(fig)

    with col_info:
        st.subheader("2. The Odds")
        
        # Get final probabilities
        probs = model.predict_proba(x_input)
        
        st.markdown(f"**Likelihood (Height of Curve):**")
        st.write(f"Does it fit the Apple curve? `{prob_c0:.4f}`")
        st.write(f"Does it fit the Orange curve? `{prob_c1:.4f}`")
        
        st.divider()
        st.markdown("**Final Verdict (Normalized):**")
        
        # Simple Bar Chart for Result
        fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
        bars = ax_bar.bar(['Apple', 'Orange'], probs, color=['blue', 'red'])
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Confidence")
        st.pyplot(fig_bar)
        
        if probs[0] > probs[1]:
            st.success(f"It's an **Apple**! ({probs[0]*100:.1f}%)")
        else:
            st.error(f"It's an **Orange**! ({probs[1]*100:.1f}%)")

with tab2:
    st.markdown("## 🧠 Bayes' Theorem")
    
    st.markdown("""
    Naive Bayes is based on a simple question: 
    **"Given that we see feature X (Size), what are the odds it belongs to Class A?"**
    """)
    
    st.latex(r"P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}")
    
    st.write("In our specific case:")
    st.latex(r"P(Class | X) = \frac{P(X | Class) \cdot P(Class)}{P(X)}")
    
    st.divider()
    
    st.markdown("### 1. The Likelihood $P(X | Class)$")
    st.write("""
    This is the **Height of the Bell Curve** at point X.
    * Look at the playground graph.
    * The **Blue Dot** is the likelihood for Class 0.
    * The **Red Dot** is the likelihood for Class 1.
    * If the Blue Dot is higher, the point likely belongs to Class 0.
    """)
    
    st.markdown("### 2. The Prior $P(Class)$")
    st.write("""
    This is our "bias" before looking at data. 
    If 90% of our dataset is Class 0, we start with a 90% guess that a new point is Class 0.
    """)
    
    st.markdown("### 3. Why 'Naive'?")
    st.write("""
    It assumes that features are **independent**. 
    For example, it assumes 'Height' has nothing to do with 'Weight'. 
    This is usually wrong in real life, but the algorithm works surprisingly well anyway!
    """)