import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('regressionprob1_train0.csv')

# Extract features and target variable
X = df.iloc[:, 0:4].values  # First four columns as features
y = df['F'].values          # Target column

# Normalize features (zero mean, unit variance)
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X_normalized = (X - mean_X) / std_X

# Add intercept term (column of ones)
ones_ = np.ones(len(y), float)
A = np.column_stack((ones_, X_normalized))

# Initialize parameters
m, n = A.shape  # m = number of samples, n = number of features + 1 (intercept)
W = np.random.randn(n)  # Random initialization of weights

# Hyperparameters
alpha = 0.01
iterations = 1000  # Number of gradient descent steps

# Gradient Descent Loop
for i in range(iterations):
    # Predictions: A @ W gives predicted y values
    y_pred = A @ W  
    
    # Residuals: y - y_pred
    residuals = y - y_pred  
    
    # Compute gradients
    gradient_W = -(1 / m) * (A.T @ residuals)  # Gradient w.r.t. W
    
    # Clip gradients to prevent overflow
    gradient_W = np.clip(gradient_W, -1e6, 1e6)
    
    # Update weights
    W -= alpha * gradient_W  # Gradient descent update step

    # Optional: Debugging to check gradient and weight norms
    if i % 100 == 0:
        print(f"Iteration {i}, Gradient Norm: {np.linalg.norm(gradient_W)}, Weights Norm: {np.linalg.norm(W)}")

# Final weights (W[0] is intercept, rest are coefficients)
intercept = W[0]
weights = W[1:]

# Compute R-squared to evaluate model
def Rsquared(Y, Yp):
    V = Y - Yp
    Ymean = np.average(Y)
    totvar = np.sum((Y - Ymean) ** 2)
    unexpvar = np.sum(np.abs(V ** 2))
    R2 = 1 - unexpvar / totvar
    return R2

R2_gradient_descent = Rsquared(y, A @ W)

# Print final results
print(f"Intercept: {intercept}")
print(f"Weights: {weights}")
print(f"R-squared (R2) using Gradient Descent: {R2_gradient_descent}")
