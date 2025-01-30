import numpy as np
import pandas as pd

# Function to compute R²
def Rsquared(Y, Yp):
    V = Y - Yp
    Ymean = np.average(Y)
    totvar = np.sum((Y - Ymean) ** 2)
    unexpvar = np.sum(np.abs(V ** 2))
    R2 = 1 - unexpvar / totvar
    return R2

# --- Load the precomputed weights from Part (a) ---
W = np.load("trained_weights.npy")

# Load the test dataset
df_test = pd.read_csv('regressionprob1_test0.csv')

# Extract test features and target variable
X_test = df_test.iloc[:, 0:4].values  # First four columns as features
y_test = df_test['F'].values          # Target column

# Add intercept term (column of ones) to X_test to form matrix A_test
ones_test = np.ones(len(y_test), float)
A_test = np.column_stack((ones_test, X_test))

# Use the trained model to predict the test set
y_pred_test = A_test @ W  

# Compute R² on test data
R2_test = Rsquared(y_test, y_pred_test)

# Print results
print(f"R-squared (R2) on test dataset: {R2_test}")
