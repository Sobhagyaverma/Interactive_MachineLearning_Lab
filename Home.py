import streamlit as st

st.set_page_config(
    page_title="ML From Scratch Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

import utils
utils.navbar()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .hero-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        font-style: italic;
    }
    .card-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        height: 100%;
        transition: transform 0.2s;
    }
    .card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #111827;
    }
    .card-text {
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧪 Interactive Machine Learning Lab</div>
    <div class="hero-subtitle">"Glass Box" AI: Build the engine, see the magic.</div>
</div>
""", unsafe_allow_html=True)

st.write("###") # Spacer

# --- INTRODUCTION ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### 👋 Welcome to the Lab
    
    Most people use Machine Learning libraries like a **Black Box**: input goes in, magic comes out. 
    **Here, we build the engine ourselves.**
    
    This lab is designed to help you build an **intuition** for how these algorithms actually work by visualizing every step of the process.
    """)
with col2:
    st.info("""
    **💡 Philosophy:**
    1. **No Magic:** Pure NumPy implementations.
    2. **Visuals:** Real-time interactive plots.
    3. **Intuition:** Understand *why* it works.
    """)

st.divider()

# --- ALGORITHM GALLERY ---
st.markdown("## 📚 The Collection")
st.write("Choose an algorithm to explore:")

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 📏 Linear Regression")
        st.markdown("The 'Hello World' of ML. Fit a line to data to predict continuous values.")
        st.write("###")
        st.page_link("pages/1_Linear_Regression.py", label="Start Experiment", icon="🚀", use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("### 🧬 Logistic Regression")
        st.markdown("Classify data into categories (Yes/No). The foundation of Neural Networks.")
        st.write("###")
        st.page_link("pages/2_Logistic_Regression.py", label="Start Experiment", icon="🚀", use_container_width=True)

with col3:
    with st.container(border=True):
        st.markdown("### 📍 K-Nearest Neighbors")
        st.markdown("Simple but powerful. Classify based on who your neighbors are.")
        st.write("###")
        st.page_link("pages/3_KNN.py", label="Start Experiment", icon="🚀", use_container_width=True)

# Row 2
col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### ✨ K-Means Clustering")
        st.markdown("Find hidden groups in data without any labels (Unsupervised Learning).")
        st.write("###")
        st.page_link("pages/4_KMeans.py", label="Start Experiment", icon="🚀", use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("### 🧠 Neural Networks")
        st.markdown("*Coming Soon...* Build a multi-layer perceptron from scratch.")
        st.write("###")
        st.button("Coming Soon", disabled=True, key="nn_btn", use_container_width=True)

with col3:
    # Placeholder for future
    st.empty()

st.divider()

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 0;">
    Built with ❤️ using Streamlit & NumPy | 
    <a href="#" style="color: #6c757d; text-decoration: none;">View Source</a>
</div>
""", unsafe_allow_html=True)