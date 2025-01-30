import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('randPolyN2.csv')

# Extract features and target variable
X = df.iloc[:, :-1].values  # All columns except the last are features
y = df['Z'].values          # The column 'Z' is the target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1),
    "Ridge (alpha=0.1)": Ridge(alpha=0.1),
    "ElasticNet (alpha=0.1)": ElasticNet(alpha=0.1)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Compute R-squared
    R2 = r2_score(y_test, y_pred)
    results[name] = R2
    
    # Plot predictions vs true values
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
    plt.title(f"{name} (R2: {R2:.3f})")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.grid(True)
    
    # Save the plot as a PNG file
    filename = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_").replace(":", "") + ".png"
    plt.savefig(filename, dpi=300)
    plt.close()

# Print summary of R2 results
print("Model Performance (R2):")
for name, R2 in results.items():
    print(f"{name}: {R2:.3f}")
