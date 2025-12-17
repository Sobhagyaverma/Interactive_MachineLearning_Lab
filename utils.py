import streamlit as st
import numpy as np

def navbar():
    """
    Creates a navigation bar at the top of the page.
    """
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.page_link("Home.py", label="Home", icon="🏠")
        with col2:
            st.page_link("pages/1_Linear_Regression.py", label="Linear Regression", icon="📏")
        with col3:
            st.page_link("pages/2_Logistic_Regression.py", label="Logistic Regression", icon="🧬")
        with col4:
            st.page_link("pages/3_KNN.py", label="KNN", icon="📍")
        with col5:
            st.page_link("pages/4_KMeans.py", label="K-Means", icon="✨")
    
    st.divider()

    # CSS to hide the default sidebar navigation
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)


def mean_squared_error(y_true, y_pred):
    """Calculates the average squared difference between actual and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """Calculates how well the regression line fits the data (1.0 is perfect)."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)

# --- NEW FUNCTION FOR LOGISTIC REGRESSION ---
def accuracy_score(y_true, y_pred):
    """
    Calculates the percentage of correct predictions.
    Automatically fixes shape mismatches (e.g., (N,1) vs (N,)).
    """
    # Flatten both arrays to 1D lists so shapes always match
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    # Calculate accuracy
    return np.mean(y_true_flat == y_pred_flat)