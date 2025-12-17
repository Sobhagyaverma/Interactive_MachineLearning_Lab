import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = []        # Stores Cost (Error)
        self.train_history = []  # Stores Weights/Bias for animation

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.history = []
        self.train_history = []

        for i in range(self.n_iterations):
            # Prediction
            y_predicted = np.dot(X, self.weights) + self.bias

            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Store history for animation
            cost = np.mean((y_predicted - y) ** 2)
            self.history.append(cost)
            self.train_history.append((self.weights.copy(), self.bias))

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias