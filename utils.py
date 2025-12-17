import numpy as np

def mean_squared_error(y_true, y_pred):
    """Calculates the average squared difference between actual and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """
    Calculates R-squared (1.0 is perfect, 0.0 is bad).
    Formula: 1 - (Sum of Squared Residuals / Total Sum of Squares)
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)