import streamlit as st

st.set_page_config(
    page_title="ML From Scratch Lab",
    page_icon="🧪",
    layout="wide"
)

# --- HEADER ---
st.title("🧪 Interactive Machine Learning Lab")
st.markdown("""
### Welcome to the "Glass Box" AI.
Most people use Machine Learning libraries like a **Black Box**: input goes in, magic comes out.
**Here, we build the engine ourselves.**
""")

st.divider()

# --- NAVIGATION MENU ---
st.header("📚 Available Algorithms")

col1, col2 = st.columns(2)

with col1:
    st.info("### 1. Supervised Learning")
    st.write("Models that learn from labeled data.")
    
    # THIS IS THE CONNECTING LINK
    st.page_link("pages/1_Linear_Regression.py", label="Start Linear Regression", icon="📏")
    
    # Placeholder for future pages
    st.button("Logistic Regression (Coming Soon)", disabled=True)

with col2:
    st.warning("### 2. Unsupervised Learning")
    st.write("Models that find hidden patterns.")
    st.button("K-Means Clustering (Coming Soon)", disabled=True)

st.divider()

# --- FOOTER ---
st.markdown("### 🏗️ How it works")
st.markdown("""
1. **Math Engine:** Written from first principles (Pure NumPy).
2. **Visualization:** Plotly & Streamlit for real-time interactivity.
3. **The Goal:** Intuition > Memorization.
""")