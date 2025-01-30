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

# Compute weights using numpy.linalg.solve (Ax = b => x = A_inv @ b)
W_solve = np.linalg.solve(A.T @ A, A.T @ y)

# Extract weights and intercept
intercept_solve = W_solve[0]
weights_solve = W_solve[1:]

# Make predictions
y_pred_solve = A @ W_solve

# Compute R2
R2_solve = Rsquared(y, y_pred_solve)

# Print results
print(f"Intercept (using solve): {intercept_solve}")
print(f"Weights (using solve): {weights_solve}")
print(f"R-squared (R2) using solve: {R2_solve}")
