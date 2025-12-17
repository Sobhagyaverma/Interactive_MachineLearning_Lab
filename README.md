# 🧪 Interactive Machine Learning Lab

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

**"Glass Box" AI: Build the engine, see the magic.**

An interactive web application that helps you understand machine learning algorithms from first principles. No black boxes—just pure math, visualization, and intuition.

## ✨ Features

- **🎮 Interactive Playgrounds**: Experiment with real-time parameters and visualize algorithm behavior
- **📊 Live Visualizations**: Watch algorithms learn step-by-step with dynamic plots
- **🧠 Theory & Math**: Understand the mathematics behind each algorithm
- **🔬 From Scratch**: Pure NumPy implementations—no ML frameworks
- **🌐 Web-Based**: Run entirely in your browser with Streamlit

## 🚀 Available Algorithms

### Supervised Learning
- **📏 Linear Regression**: Fit a line to data and predict continuous values
- **🧬 Logistic Regression**: Binary classification with the sigmoid function
- **📍 K-Nearest Neighbors (KNN)**: Classify based on proximity to neighbors

### Unsupervised Learning
- **✨ K-Means Clustering**: Discover hidden patterns in unlabeled data

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sobhagyaverma/Interactive_MachineLearning_Lab.git
   cd Interactive_MachineLearning_Lab
   ```

2. **Create a virtual environment** (recommended)
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

5. **Open your browser**
   
   The app should automatically open at `http://localhost:8501`

## 📚 Project Structure

```
Interactive_MachineLearning_Lab/
├── Home.py                          # Landing page
├── pages/                           # Algorithm pages
│   ├── 1_Linear_Regression.py
│   ├── 2_Logistic_Regression.py
│   ├── 3_KNN.py
│   └── 4_KMeans.py
├── models/                          # ML implementations
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   └── kmeans.py
├── utils.py                         # Shared utilities
└── .streamlit/                      # Streamlit config
    └── config.toml
```

## 🎯 How It Works

### Philosophy
1. **No Magic**: Pure NumPy implementations from scratch
2. **Visuals**: Real-time interactive plots using Plotly
3. **Intuition**: Understand *why* it works, not just *how*

### Learning Path
1. **Explore**: Play with the interactive controls
2. **Observe**: Watch the algorithm learn in real-time
3. **Understand**: Read the theory to grasp the mathematics
4. **Experiment**: Adjust parameters and see what happens

## 🖼️ Screenshots

<!-- Add screenshots here when available -->

## 🤝 Contributing

Contributions are welcome! If you'd like to add new algorithms or improve existing ones:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes (`git commit -m 'Add new algorithm'`)
4. Push to the branch (`git push origin feature/new-algorithm`)
5. Open a Pull Request

## 📝 To-Do

- [ ] Neural Networks (Multi-Layer Perceptron)
- [ ] Decision Trees
- [ ] Support Vector Machines
- [ ] Principal Component Analysis (PCA)
- [ ] Add more visualizations
- [ ] Export trained models

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Sobhagya Verma**

- GitHub: [@Sobhagyaverma](https://github.com/Sobhagyaverma)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/)
- Mathematical computations with [NumPy](https://numpy.org/)

---

⭐ If you found this project helpful, please give it a star!
