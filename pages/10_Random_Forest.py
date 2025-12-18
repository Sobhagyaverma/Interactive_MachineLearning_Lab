
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from scipy.stats import mode
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Random Forest - ML Lab",
    page_icon="🌲",
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

# --- HELPER CLASSES (Simplified Decision Tree) ---
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
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
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None: # Safety check
            return Node(value=self._most_common_label(y))

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
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l/n) * e_l + (n_r/n) * e_r
        return parent_gini - child_gini

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - sum((counts / len(y))**2)

    def _most_common_label(self, y):
        if len(y) == 0: return 0
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

# --- RANDOM FOREST CLASS ---
class RandomForestScratch:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # BOOTSTRAPPING: Randomly sample with replacement
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # Gather predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority Vote (Mode) across columns
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        predictions = []
        for preds in tree_preds:
            # simple majority vote
            most_common = np.bincount(preds).argmax()
            predictions.append(most_common)
        return np.array(predictions)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Data Generator")
    data_type = st.selectbox("Dataset", ["Moons", "Circles", "Blobs"])
    noise = st.slider("Noise", 0.0, 0.5, 0.3)
    n_samples = st.slider("Samples", 50, 500, 200)
    
    if st.button("Generate New Data", type="primary") or 'rf_data' not in st.session_state:
        if data_type == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        elif data_type == "Circles":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        else:
            X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42, cluster_std=1.5)
        st.session_state['rf_data'] = (X, y)

X, y = st.session_state['rf_data']

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌲🌲 Random Forest</div>
    <div class="hero-subtitle">"Wisdom of the Crowd: Many weak trees make one strong forest."</div>
</div>
""", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["🛝 Playground", "📝 Theory & Math"])

with tab1:
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.subheader("1. Forest Config")
        n_trees = st.slider("Number of Trees", 1, 50, 5)
        st.caption("More trees = Smoother boundary (but slower).")
        
        max_depth = st.slider("Max Depth per Tree", 1, 20, 5)
        
        if st.button("Grow Forest", type="primary", use_container_width=True):
            rf = RandomForestScratch(n_trees=n_trees, max_depth=max_depth)
            with st.spinner(f"Training {n_trees} trees..."):
                rf.fit(X, y)
            st.session_state['rf_model'] = rf

    with col_viz:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if 'rf_model' in st.session_state:
            clf = st.session_state['rf_model']
            
            # Meshgrid for visualization
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdBu')
            ax.set_title(f"Random Forest ({n_trees} Trees)")
        else:
            ax.set_title("Ready to Train")

        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='k', s=50)
        st.pyplot(fig)

with tab2:
    st.markdown("## 🧠 Why build a Forest?")
    
    st.markdown("""
    A single Decision Tree is like a **Genius who overthinks**. 
    It memorizes the data so well that it stumbles when it sees something slightly new (Overfitting/High Variance).
    
    A Random Forest solves this by creating a **Committee of Idiots**. 
    Each tree is forced to be slightly different (randomized), so they make different mistakes.
    """)
    
    st.divider()
    
    st.markdown("### 1. The Secret Sauce: Bagging")
    st.write("**B**ootstrap **Agg**regat**ing**.")
    
    st.markdown("""
    * **Bootstrap:** We don't give the trees the same data! We create random "sub-datasets" by pulling names out of a hat (with replacement). Tree A might see point #5 three times, while Tree B never sees it.
    * **Feature Randomness:** Normally, a tree picks the *best* feature to split. In a forest, we force it to pick from a *random subset* of features. This makes the trees diverse.
    """)

    st.markdown("### 2. Aggregating (The Vote)")
    st.write("Once we have 100 trees, we ask them to vote:")
    
    st.code("""
    Tree 1 says: Class A
    Tree 2 says: Class A
    Tree 3 says: Class B
    ...
    Result: Class A wins!
    """)
    
    st.info("The math proves that averaging these errors cancels them out, leaving you with the true pattern.")