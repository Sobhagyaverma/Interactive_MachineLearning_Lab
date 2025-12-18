import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
import utils
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Decision Tree - ML Lab",
    page_icon="🌳",
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
    .math-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4f46e5;
        font-family: monospace;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- DECISION TREE CLASSES (FROM SCRATCH) ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature        # Index of feature to split on
        self.threshold = threshold    # Value to split at
        self.left = left              # Left Child Node
        self.right = right            # Right Child Node
        self.value = value            # Leaf node value (Class Label)

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeScratch:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping Criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)

        # Find the best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Build Children
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        # Parent Gini
        parent_gini = self._gini(y)

        # Generate split
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Weighted average child Gini
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l/n) * e_l + (n_r/n) * e_r

        # Information Gain is Gini Reduction
        ig = parent_gini - child_gini
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - sum(probabilities**2)

    def _most_common_label(self, y):
        # Fix for empty slice scenario
        if len(y) == 0: return 0 
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Generator")
    data_type = st.selectbox("Dataset", ["Blobs", "Moons", "Circles"])
    noise = st.slider("Noise", 0.0, 0.5, 0.2)
    n_samples = st.slider("Samples", 50, 500, 200)
    
    if st.button("Generate New Data", type="primary") or 'dt_data' not in st.session_state:
        if data_type == "Blobs":
            X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42, cluster_std=1.0 + noise)
        elif data_type == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        else:
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        st.session_state['dt_data'] = (X, y)

X, y = st.session_state['dt_data']

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌳 Decision Trees</div>
    <div class="hero-subtitle">"Chopping the world into simple boxes."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🛝 Playground", "📝 Theory & Math"])

with tab1:
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.subheader("1. Tree Config")
        max_depth = st.slider("Max Depth", 1, 15, 3)
        st.caption("How many times can the tree split? (1 = 1 Cut)")
        
        min_split = st.slider("Min Samples Split", 2, 20, 2)
        st.caption("Don't split if a box has fewer than X points.")
        
        if st.button("Train Tree", type="primary", use_container_width=True):
            clf = DecisionTreeScratch(max_depth=max_depth, min_samples_split=min_split)
            with st.spinner("Planting tree..."):
                clf.fit(X, y)
            st.session_state['dt_model'] = clf
            st.success(f"Tree grown to depth {max_depth}!")

    with col_viz:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Decision Boundaries if model exists
        if 'dt_model' in st.session_state:
            clf = st.session_state['dt_model']
            
            # Create Meshgrid
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot contours (The Boxes)
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
            ax.set_title(f"Decision Boundary (Depth {max_depth})")
        else:
            ax.set_title("Ready to Train")

        # Plot Data
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        st.pyplot(fig)

with tab2:
    st.markdown("## 🧠 How the Tree Thinks")
    
    st.markdown("""
    A Decision Tree is basically playing a game of **"20 Questions"** with the data. 
    It wants to separate the Red dots from the Blue dots using only **Straight Lines** (Vertical or Horizontal).
    """)
    
    st.markdown("### 1. The Strategy (Divide and Conquer)")
    st.write("""
    1.  **Start:** Look at all the data mixed together.
    2.  **Ask:** "What is the best single line I can draw to separate the colors?"
    3.  **Split:** Draw the line. Now we have two smaller boxes.
    4.  **Repeat:** Go into each box and repeat step 2 until the box is "Pure" (only one color) or we hit `Max Depth`.
    """)
    
    st.divider()
    
    st.markdown("### 2. The Math (Gini Impurity)")
    st.write("How does it decide *where* to draw the line? It wants to minimize **Impurity**.")
    
    st.markdown("""
    * **Impurity = 0:** Perfect! The box contains only Blue dots.
    * **Impurity = 0.5:** Worst case! The box is 50% Blue, 50% Red.
    """)
    
    st.latex(r"Gini = 1 - \sum (p_i)^2")
    
    st.info("""
    **Glass Box Check:** Look at the `_best_split` function in the code. It literally tries **every possible vertical and horizontal line**, calculates the Gini Impurity for each, and picks the winner.
    """)