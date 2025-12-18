import streamlit as st
import numpy as np

def navbar():
    """
    Creates a navigation bar at the top of the page.
    """
    # CSS for better navbar styling
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
            .nav-container {
                background: linear-gradient(to right, #f8f9fa, #e9ecef);
                padding: 0.75rem 1rem;
                border-radius: 10px;
                margin-bottom: 1.5rem;
            }
            .stButton button {
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        # Row 1: Core algorithms
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.page_link("Home.py", label="🏠 Home", use_container_width=True)
        with col2:
            st.page_link("pages/1_Linear_Regression.py", label="📏 Linear Reg", use_container_width=True)
        with col3:
            st.page_link("pages/2_Logistic_Regression.py", label="🧬 Logistic Reg", use_container_width=True)
        with col4:
            st.page_link("pages/3_KNN.py", label="📍 KNN", use_container_width=True)
        
        # Row 2: More algorithms
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.page_link("pages/4_KMeans.py", label="✨ K-Means", use_container_width=True)
        with col2:
            st.page_link("pages/5_Neural_Networks.py", label="🧠 Neural Net", use_container_width=True)
        with col3:
            st.page_link("pages/6_Polynomial_Regression.py", label="📈 Polynomial", use_container_width=True)
        with col4:
            st.page_link("pages/7_SVM.py", label="⚔️ SVM", use_container_width=True)
        
        # Row 3: New algorithms
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.page_link("pages/8_Naive_Bayes.py", label="🎲 Naive Bayes", use_container_width=True)
        with col2:
            st.page_link("pages/9_Decision_Tree.py", label="🌳 Decision Tree", use_container_width=True)
        with col3:
            st.page_link("pages/10_Random_Forest.py", label="🌲 Random Forest", use_container_width=True)
        with col4:
            st.page_link("pages/11_CNN.py", label="👁️ CNN", use_container_width=True)
        
        # Row 4: RL
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.page_link("pages/12_Reinforcement_Learning.py", label="🎮 Q-Learning", use_container_width=True)
    
    st.divider()



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