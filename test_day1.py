import numpy as np
from models.linear_regression import LinearRegression
from utils import mean_squared_error, r2_score

print("🎲 Generating synthetic data...")
# We create data that follows the rule: y = 3x + 4
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) * 0.5 

print("🚀 Training Linear Regression Model...")
model = LinearRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# Evaluate
predictions = model.predict(X)
r2 = r2_score(y, predictions)

print("-" * 30)
print(f"✅ TARGET EQUATION: y = 3.00x + 4.00")
print(f"🤖 MODEL FOUND:    y = {model.weights[0][0]:.2f}x + {model.bias:.2f}")
print("-" * 30)
print(f"📊 R2 Score: {r2:.4f}")

if r2 > 0.90:
    print("\n🎉 SUCCESS: Your M4 Chip crushed the math!")
else:
    print("\n⚠️ FAIL: Something is wrong with the math.")