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

# --- MAIN LAYOUT ---
st.info("✍️ Draw a digit (0-9). **Draw BIG and centered** for best results! The AI will predict in real-time.")

col_draw, col_predict = st.columns([1, 1])

with col_draw:
    st.subheader("📝 Draw Here")
    
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

with col_predict:
    st.subheader("🔮 Prediction")
    
    # Real-time Prediction
    flat_input = input_img.flatten().reshape(1, -1)
    scores = model.predict_proba(flat_input)[0]
    prediction = np.argmax(scores)
    confidence = scores[prediction]
    
    # Big prediction display
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1rem;'>
        <div style='font-size: 4rem; font-weight: 800; color: white;'>{prediction}</div>
        <div style='font-size: 1.2rem; color: #f0f0f0;'>Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Processed image preview
    st.write("**How the AI sees it (28x28 preprocessed):**")
    fig_preview, ax_preview = plt.subplots(figsize=(3, 3))
    ax_preview.imshow(input_img, cmap='gray')
    ax_preview.axis('off')
    st.pyplot(fig_preview)
    plt.close(fig_preview)

# --- TABS FOR ADVANCED FEATURES ---
st.divider()
tab1, tab2 = st.tabs(["🔬 Advanced: Filters & Neurons", "📚 Theory & Math"])

with tab1:
    st.subheader("Computer Vision Pipeline")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.write("**1. Choose Filter**")
        filter_type = st.selectbox("Filter Type", ["Vertical Edge", "Horizontal Edge", "Sharpen", "Outline"], label_visibility="collapsed")
        
        kernels = {
            "Vertical Edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            "Horizontal Edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "Outline": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        }
        selected_kernel = kernels[filter_type]
        
        st.write("**Kernel:**")
        st.write(selected_kernel)
    
    with col_f2:
        st.write("**2. Input Patch (Center)**")
        # Show a 3x3 patch from center
        cy, cx = 14, 14
        patch = input_img[cy:cy+3, cx:cx+3]
        st.image(patch, width=100, clamp=True)
        
    with col_f3:
        st.write("**3. Convolution Result**")
        bias_val = st.slider("Add Bias", -2.0, 2.0, 0.0, key="bias_slider")
        weighted_sum, total, activation = convolve_step(patch, selected_kernel, bias_val)
        
        st.metric("Output", f"{activation:.2f}")
        st.caption("ReLU(Σ(X·W) + b)")

with tab2:
    st.markdown("## 🧠 What is a Convolutional Neural Network?")
    
    st.info("""
    A CNN is like a **detective with a magnifying glass** scanning an image for patterns.
    Instead of looking at the whole image at once, it examines small patches (like a 3×3 window)
    and learns to recognize features like edges, curves, and textures.
    """)
    
    st.divider()
    
    st.markdown("### 1. Convolution: The Scanning Process")
    
    st.write("""
    Imagine you have a filter (kernel) that detects vertical edges. 
    The CNN slides this filter across the entire image, one patch at a time.
    """)
    
    col_theory1, col_theory2 = st.columns([1, 1])
    
    with col_theory1:
        st.write("**The Math:**")
        st.latex(r"Output_{(i,j)} = \sum_{m,n} Input_{(i+m, j+n)} \times Kernel_{(m,n)}")
        st.write("""
        - **Input**: A small patch (e.g., 3×3 pixels)
        - **Kernel/Filter**: Learned weights (also 3×3)
        - **Output**: Element-wise multiply and sum
        """)
    
    with col_theory2:
        st.write("**Why It Works:**")
        st.write("""
        - **Local Patterns**: Nearby pixels are related (a curve, an edge)
        - **Weight Sharing**: The same filter is used everywhere (efficiency!)
        - **Translation Invariance**: A "2" in the top-left looks the same as a "2" in the bottom-right
        """)
    
    st.divider()
    
    st.markdown("### 2. Filters: What Do They Detect?")
    
    st.write("""
    Different filters detect different features:
    - **Edge Detectors**: Highlight boundaries where brightness changes
    - **Texture Detectors**: Recognize patterns like dots, stripes
    - **High-Level Features**: In deep networks, later layers combine edges to detect noses, wheels, letters
    """)
    
    st.code("""
Vertical Edge Filter:
[[-1,  0,  1],
 [-1,  0,  1],
 [-1,  0,  1]]
 
Explanation:
- Left side (negative): Dark pixels
- Right side (positive): Bright pixels
- If there's a brightness jump → Output is HIGH
    """)
    
    st.divider()
    
    st.markdown("### 3. ReLU: The Activation Function")
    
    st.write("After convolution, we apply **ReLU (Rectified Linear Unit)**:")
    st.latex(r"ReLU(x) = \max(0, x)")
    
    st.write("""
    **Why?** 
    - It introduces non-linearity (allows the network to learn complex patterns)
    - It's simple and fast to compute
    - Negative values become 0 (only keep "activated" features)
    """)
    
    st.divider()
    
    st.markdown("### 4. How This Page Works")
    
    st.info("""
    **This demo uses Logistic Regression (not a true CNN)** for speed and simplicity.
    
    **What we're showing:**
    1. **Preprocessing**: Your drawing → Centered 28×28 image (matching MNIST format)
    2. **Convolution Demo**: Apply a filter to see how edge detection works
    3. **Classification**: A trained model predicts which digit (0-9) you drew
    
    **A real CNN would:**
    - Have multiple convolutional layers (10-100+ filters per layer)
    - Use pooling (downsampling) to reduce image size
    - Stack layers: Conv → ReLU → Pool → Conv → ReLU → Pool → Fully Connected → Output
    
    **Fun Fact:** The model was trained on 10,000 MNIST digits in just a few seconds!
    """)
    
    st.divider()
    
    st.markdown("### 5. Key Concepts Summary")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("""
        **Convolution:**
        - Slide filter over image
        - Multiply + Sum
        - Detect local patterns
        """)
        
        st.markdown("""
        **Padding:**
        - Add borders to image
        - Prevent shrinking
        - Preserve edge info
        """)
    
    with col_c2:
        st.markdown("""
        **Pooling:**
        - Downsample feature maps
        - Reduce computation
        - Add translation invariance
        """)
        
        st.markdown("""
        **Why CNNs Win:**
        - Spatial awareness
        - Parameter sharing
        - Hierarchical learning
        """)