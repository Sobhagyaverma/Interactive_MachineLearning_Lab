import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """
        KNN is a 'Lazy Learner'. It doesn't learn a formula.
        It simply memorizes the training data.
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        """Predict labels for an entire dataset X."""
        predicted_labels = [self._predict_single(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict_single(self, x):
        """
        Helper method to predict a single point.
        """
        
        distances = [np.sqrt(np.sum((x_train - x)**2)) for x_train in self.X_train] 
        
        k_indices = np.argsort(distances)[:self.k] 
        
        k_nearest_labels = [self.y_train[i][0] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]