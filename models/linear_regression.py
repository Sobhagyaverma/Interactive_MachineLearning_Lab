import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = [] 

    def fit(self, X, y):
        """
        Trains the model using Gradient Descent.
        X: Input data (features)
        y: Target data (labels)
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize Weights (Parameters)
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.history = []

        # 2. Gradient Descent Loop
        for i in range(self.n_iterations):
            # A. Prediction: y = wx + b
            y_predicted = np.dot(X, self.weights) + self.bias

            # B. Calculate Gradients (The Slope)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # C. Update Parameters (Walk down the hill)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # D. Record Cost (For visualization later)
            cost = np.mean((y_predicted - y) ** 2)
            self.history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias