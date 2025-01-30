import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('randPolyN2.csv')

# Extract features and target variable
X = df.iloc[:, :-1].values  # All columns except the last are features
y = df['Z'].values          # The column 'Z' is the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Degrees of polynomial to test
degrees = [1, 2, 3, 4, 5]

# Dictionary to store results
results = {}

# Train and evaluate Ridge Regression with polynomial features
for degree in degrees:
    # Create a pipeline with PolynomialFeatures and Ridge Regression
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute R-squared
    R2 = r2_score(y_test, y_pred)
    results[degree] = R2
    
    # Plot predictions vs true values
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    plt.title(f"Polynomial Degree {degree} (R2: {R2:.3f})")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.grid(True)
    
    # Save the plot as a PNG file
    filename = f"Polynomial_Degree_{degree}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# Print summary of R2 results
print("Model Performance (R2) for Different Degrees:")
for degree, R2 in results.items():
    print(f"Degree {degree}: {R2:.3f}")
