import numpy as np

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