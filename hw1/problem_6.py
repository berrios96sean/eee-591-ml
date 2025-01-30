import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the training dataset
df_train = pd.read_csv('regressionprob1_train0.csv')

# Extract features and target variable
X_train = df_train.iloc[:, 0:4].values  # First four columns as features
y_train = df_train['F'].values          # Target column

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Extract the weights and intercept
intercept = model.intercept_
weights = model.coef_

# Make predictions on the training data
y_pred_train = model.predict(X_train)

# Compute R-squared for the training data
R2_train = r2_score(y_train, y_pred_train)

# Print results
print(f"Intercept: {intercept}")
print(f"Weights: {weights}")
print(f"R-squared (R2) on training data: {R2_train}")
