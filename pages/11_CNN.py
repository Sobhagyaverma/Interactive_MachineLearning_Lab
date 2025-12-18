import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.ndimage import center_of_mass
import utils

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Interactive CNN - ML Lab",
    page_icon="✍️",
    layout="wide"
)

utils.navbar()

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .hero-container { padding: 1rem 0; margin-bottom: 2rem; }
    .hero-title { font-size: 2.5rem; font-weight: 800; color: #1f2937; }
    .hero-subtitle { font-size: 1.1rem; color: #6c757d; font-style: italic; }
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

# --- CACHED MODEL TRAINING ---
@st.cache_resource
def train_model():
    """
    Trains a Logistic Regression model on MNIST.
    """
    # Load data
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False, parser='auto')
    X = mnist.data / 255.0  # Normalize
    y = mnist.target.astype(int)
    
    # Train on a subset for speed
    X_train, _, y_train, _ = train_test_split(X, y, train_size=10000, random_state=42, stratify=y)
    
    # Logistic Regression (One-vs-Rest)
    clf = LogisticRegression(solver='lbfgs', max_iter=100, multi_class='multinomial', tol=0.01)
    clf.fit(X_train, y_train)
    
    return clf

# --- ROBUST PREPROCESSING (THE FIX) ---
def process_canvas(canvas_result):
    """
    Robust preprocessing to match MNIST standard with IMPROVED centering.
    """
    if canvas_result.image_data is None:
        return np.zeros((28, 28))

    # 1. Get image & Convert to Grayscale
    img = canvas_result.image_data.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

    # 2. Invert if needed (MNIST has white digit on black background)
    # The canvas has white strokes on black, which is correct
    
    # 3. Check if image is empty
    if np.sum(img) == 0:
        return np.zeros((28, 28))

    # 4. Apply threshold to clean up noise
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 5. Find Bounding Box (Crop to the drawing)
    coords = cv2.findNonZero(img) 
    if coords is None:
        return np.zeros((28, 28))
        
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # 6. Add padding to prevent edge artifacts
    pad = 4
    digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    # 7. Resize to fit in 20x20 box (MNIST standard)
    # Keep aspect ratio
    if h > w:
        factor = 20.0 / (h + 2*pad)
        h_new = 20
        w_new = max(1, int((w + 2*pad) * factor))
    else:
        factor = 20.0 / (w + 2*pad)
        w_new = 20
        h_new = max(1, int((h + 2*pad) * factor))
        
    if h_new <= 0 or w_new <= 0: 
        return np.zeros((28, 28))
    
    # Use better interpolation
    digit_resized = cv2.resize(digit, (w_new, h_new), interpolation=cv2.INTER_AREA)

    # 8. Center in 28x28 canvas
    canvas_28 = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - h_new) // 2
    x_off = (28 - w_new) // 2
    canvas_28[y_off:y_off+h_new, x_off:x_off+w_new] = digit_resized

    # 9. Center of Mass alignment (more gentle)
    cy, cx = center_of_mass(canvas_28)
    if not np.isnan(cy) and not np.isnan(cx):
        dy = int(round(14 - cy))
        dx = int(round(14 - cx))
        # Limit shift to prevent bad centering
        dy = np.clip(dy, -3, 3)
        dx = np.clip(dx, -3, 3)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        canvas_28 = cv2.warpAffine(canvas_28, M, (28, 28))

    # 10. Normalize to [0, 1]
    return canvas_28.astype(np.float32) / 255.0

def convolve_step(patch, kernel, bias):
    weighted_sum = np.sum(patch * kernel)
    output = weighted_sum + bias
    activation = max(0, output)
    return weighted_sum, output, activation

# --- LOAD MODEL ---
with st.spinner("Training Brain (Logistic Regression on MNIST)..."):
    model = train_model()
    weights = model.coef_ 
    biases = model.intercept_

# --- HERO ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">✍️ Interactive CNN & Neuron Inspector</div>
    <div class="hero-subtitle">"Draw a digit. Watch the neurons fire. See the math."</div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col_draw, col_process = st.columns([1, 2])

with col_draw:
    st.subheader("1. Input: Draw Here")
    st.info("Draw a digit 0-9. **Draw BIG** and centered for best results!")
    
    # Improved Canvas
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=25,  # Thicker brush
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button("🗑️ Clear Canvas", use_container_width=True): 
        st.rerun()

input_img = process_canvas(canvas_result)

with col_process:
    st.subheader("2. Computer Vision Pipeline")
    
    tab1, tab2, tab3 = st.tabs(["👁️ Convolution (Filters)", "🧠 Neuron Math (Glass Box)", "🔮 Prediction"])
    
    # --- TAB 1: FILTERS ---
    with tab1:
        st.write("Apply filters to detect edges.")
        filter_type = st.selectbox("Choose Filter", ["Vertical Edge", "Horizontal Edge", "Sharpen", "Outline"])
        
        kernels = {
            "Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "Outline": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        }
        selected_kernel = kernels[filter_type]
        
        # Simple Convolution Loop
        h, w = input_img.shape
        kh, kw = selected_kernel.shape
        output_map = np.zeros((h-kh+1, w-kw+1))
        
        for y in range(output_map.shape[0]):
            for x in range(output_map.shape[1]):
                patch = input_img[y:y+kh, x:x+kw]
                output_map[y, x] = np.sum(patch * selected_kernel)
        
        c1, c2, c3 = st.columns(3)
        with c1: st.image(input_img, caption="Processed Input (Centered)", width=150, clamp=True)
        with c2: 
            st.write("**Filter**")
            st.write(selected_kernel)
        with c3:
            disp_map = np.maximum(0, output_map)
            disp_map = disp_map / (disp_map.max() + 1e-8)
            st.image(disp_map, caption="Feature Map", width=150, clamp=True)

    # --- TAB 2: NEURON INSPECTOR ---
    with tab2:
        st.info("Hover logic simulated: Inspecting the Center Pixel.")
        bias_val = st.slider("Neuron Bias", -5.0, 5.0, 0.0)
        
        cy, cx = 14, 14
        patch = input_img[cy:cy+3, cx:cx+3]
        weighted_sum, total, activation = convolve_step(patch, selected_kernel, bias_val)
        
        c_n1, c_n2, c_n3 = st.columns(3)
        with c_n1: 
            st.image(patch, width=80, clamp=True)
            st.caption("Input Patch")
        with c_n2: 
            st.image(selected_kernel, width=80, clamp=True)
            st.caption("Weights")
        with c_n3:
            st.metric("Output", f"{activation:.2f}")
            st.latex(r"ReLU(\sum (X \cdot W) + b)")

    # --- TAB 3: PREDICTION (IMPROVED) ---
    with tab3:
        # Prediction
        flat_input = input_img.flatten().reshape(1, -1)
        scores = model.predict_proba(flat_input)[0]
        prediction = np.argmax(scores)
        
        st.success(f"Model Prediction: **{prediction}** (Confidence: {scores[prediction]*100:.1f}%)")
        
        st.write("### 🧠 The Learned Weights")
        st.write("These heatmaps show what the model *looks for* in each digit.")
        st.write("* **Red:** Negative Weight (Should be black)")
        st.write("* **Blue:** Positive Weight (Should be white)")

        # Visualize Top 3
        top_3 = np.argsort(scores)[::-1][:3]
        cols = st.columns(3)
        
        for i, idx in enumerate(top_3):
            with cols[i]:
                w_img = weights[idx].reshape(28, 28)
                fig, ax = plt.subplots(figsize=(2,2))
                max_abs = np.max(np.abs(w_img))
                ax.imshow(w_img, cmap='coolwarm', vmin=-max_abs, vmax=max_abs)
                ax.axis('off')
                st.pyplot(fig)
                
                st.caption(f"Weights for '{idx}'")
                st.progress(float(scores[idx]))