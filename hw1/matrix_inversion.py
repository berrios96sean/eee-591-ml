import numpy as np
import pandas as pd

# Compute R-squared
def Rsquared(Y, Yp):
    V = Y - Yp
    Ymean = np.average(Y)
    totvar = np.sum((Y - Ymean) ** 2)
    unexpvar = np.sum(np.abs(V ** 2))
    R2 = 1 - unexpvar / totvar
    return R2

# Load the dataset
df = pd.read_csv('regressionprob1_train0.csv')

# Extract features and target variable
X = df.iloc[:, 0:4].values  # First four columns as features
y = df['F'].values          # Target column

# Add intercept term (column of ones) to X to form matrix A
ones_ = np.ones(len(y), float)
A = np.column_stack((ones_, X))

# Compute weights using matrix inversion (closed-form solution)
W = np.linalg.inv(A.T @ A) @ (A.T @ y)

np.save("trained_weights.npy", W)  # Save weights to file for part c

# Extract weights and intercept
intercept = W[0]
weights = W[1:]

# Make predictions
y_pred = A @ W

# Compute R2
R2 = Rsquared(y, y_pred)

# Print results
print(f"Intercept: {intercept}")
print(f"Weights: {weights}")
print(f"R-squared (R2): {R2}")
