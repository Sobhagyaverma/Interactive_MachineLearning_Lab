# 🧪 Interactive Machine Learning Lab

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**"See it. Play with it. Understand it."**

An **interactive, visual, and hands-on** platform to learn Machine Learning algorithms from scratch. No black boxes—just pure understanding through experimentation!

## 🎯 Philosophy

This isn't just another ML tutorial. It's an **interactive laboratory** where you can:
- 🎮 **Play** with real algorithms in real-time
- 👀 **Visualize** exactly what's happening under the hood
- 🧠 **Learn** the theory and math behind each algorithm
- 🔬 **Experiment** with different parameters and datasets

## ✨ Features

### � 12 Interactive Algorithm Playgrounds

1. **📏 Linear Regression** - Watch the line fit your data points in real-time
2. **🧬 Logistic Regression** - Interactive breast cancer diagnosis with probability visualization
3. **📍 K-Nearest Neighbors (KNN)** - Classify points and see decision boundaries live
4. **✨ K-Means Clustering** - Animate cluster formation step-by-step
5. **🧠 Neural Networks** - Interactive playground with live neuron activation visualization
6. **📈 Polynomial Regression** - See how different degrees fit curved data
7. **⚔️ Support Vector Machine (SVM)** - Visualize the "maximum margin street" concept
8. **🎲 Naive Bayes** - Interactive classification with probability explanations
9. **� Decision Tree** - Visualize tree splits and decision boundaries
10. **🌲 Random Forest** - See ensemble learning in action
11. **�️ Convolutional Neural Network (CNN)** - Draw digits and watch live predictions with MNIST
12. **🎮 Q-Learning (Reinforcement Learning)** - Watch an agent learn to navigate a maze

### 🎯 What Makes This Special?

- **🔍 Glass Box Approach**: Every algorithm shows you exactly what's happening internally
- **🎚️ Interactive Controls**: Adjust hyperparameters and see immediate effects
- **� Real-time Visualization**: Watch algorithms learn and make decisions
- **📚 Theory + Practice**: Each page includes comprehensive math explanations
- **🎨 Beautiful UI**: Modern, responsive design with smooth animations
- **🚀 Live Predictions**: Draw, click, or interact to get instant results

## � Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sobhagyaverma/Interactive_MachineLearning_Lab.git
cd Interactive_MachineLearning_Lab
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run Home.py
```

5. **Open in browser**
The app will automatically open at `http://localhost:8501`

## 🎓 Learning Path

### Beginner-Friendly Start
1. **Linear Regression** - Understand the foundation of ML
2. **Logistic Regression** - Learn classification basics
3. **K-Means** - Grasp unsupervised learning

### Intermediate Exploration
4. **KNN** - Distance-based classification
5. **Decision Tree** - Tree-based learning
6. **Naive Bayes** - Probabilistic classification

### Advanced Topics
7. **Neural Networks** - Deep learning fundamentals
8. **SVM** - Kernel methods and optimization
9. **Random Forest** - Ensemble methods
10. **CNN** - Computer vision basics
11. **Q-Learning** - Reinforcement learning

## �️ Tech Stack

- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Computer Vision**: OpenCV
- **Canvas Drawing**: streamlit-drawable-canvas

## 📁 Project Structure

```
Interactive_MachineLearning_Lab/
├── Home.py                              # Landing page
├── pages/                               # Algorithm pages
│   ├── 1_Linear_Regression.py
│   ├── 2_Logistic_Regression.py
│   ├── 3_KNN.py
│   ├── 4_KMeans.py
│   ├── 5_Neural_Networks.py
│   ├── 6_Polynomial_Regression.py
│   ├── 7_SVM.py
│   ├── 8_Naive_Bayes.py
│   ├── 9_Decision_Tree.py
│   ├── 10_Random_Forest.py
│   ├── 11_CNN.py
│   └── 12_Reinforcement_Learning.py
├── utils.py                             # Shared utilities & navbar
├── requirements.txt                     # Dependencies
└── .streamlit/                          # Streamlit configuration
    └── config.toml
```

## 🎯 Key Features by Algorithm

### CNN (Digit Recognition)
- ✍️ **Draw digits** directly on canvas
- 🔮 **Real-time prediction** as you draw
- 📊 **Preprocessing visualization** - see how the image is processed
- 🧠 **Convolution filters** - experiment with edge detection
- 📚 **Comprehensive theory** explaining CNNs from scratch

### Q-Learning (Reinforcement Learning)
- 🤖 **Interactive maze environment** with treasure and traps
- 🎬 **Watch live episodes** - see the agent learn step-by-step
- � **Q-Table visualization** - observe the agent's knowledge grow
- 📈 **Training progress** - track rewards and exploration rate
- 🎮 **Three modes**: Single step, Watch episode, Fast train

### Neural Networks
- 🎨 **Interactive playground** with adjustable architecture
- 👁️ **Neuron activation** - see weights and biases in action
- 📊 **Decision boundary** visualization
- 🎚️ **Real-time training** with live loss tracking

## 🎨 UI/UX Features

- **Modern Design**: Gradient backgrounds, smooth animations
- **Responsive Layout**: Works on desktop and tablets
- **Tab Navigation**: Clean separation of Playground and Theory
- **Hero Sections**: Beautiful introduction to each algorithm
- **Interactive Controls**: Intuitive sliders and buttons
- **Visual Feedback**: Toasts, success messages, and animations

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug fixes
- ✨ New algorithm implementations
- 📖 Documentation improvements
- 🎨 UI/UX enhancements

Please feel free to submit a Pull Request!

## � License

This project is open source and available under the MIT License.

## 🌟 Acknowledgments

Built with ❤️ to make Machine Learning accessible and fun for everyone!

## 📬 Contact

**Sobhagya Verma**
- GitHub: [@Sobhagyaverma](https://github.com/Sobhagyaverma)

---

**Star ⭐ this repository if you found it helpful!**
