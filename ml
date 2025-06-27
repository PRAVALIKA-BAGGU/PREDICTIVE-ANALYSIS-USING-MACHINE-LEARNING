
# Install necessary libraries
!pip install scikit-learn pandas matplotlib seaborn

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model, can be changed
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset file
# If your dataset is in Google Drive, you might need to mount it first
# from google.colab import drive
# drive.mount('/content/drive')
# data = pd.read_csv('/content/drive/My Drive/your_dataset.csv')
# For this example, let's create a dummy dataset
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'target': [11, 11, 11, 11, 11, 11, 11, 11, 11, 11] # Dummy target for classification
    # For regression, use a continuous target:
    # 'target': [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
})

# --- Data Exploration and Preprocessing ---

print("Dataset Info:")
data.info()

print("\nFirst 5 rows:")
print(data.head())

print("\nDescriptive Statistics:")
print(data.describe())

# Visualize relationships (optional but recommended)
sns.pairplot(data)
plt.suptitle('Pairplot of Features and Target', y=1.02)
plt.show()

# Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# --- Feature Selection ---

# For simple datasets, visual inspection of correlation can guide selection.
# For more complex datasets, techniques like:
# - Variance Threshold
# - SelectKBest
# - Recursive Feature Elimination (RFE)
# can be used.

# Example: Selecting features based on correlation with the target (manual)
# In this dummy dataset, both features have some relationship with the target
features = ['feature1', 'feature2']
target = 'target'

X = data[features]
y = data[target]

# --- Split Data ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# --- Model Training ---

# For Classification: Use models like LogisticRegression, RandomForestClassifier, etc.
# For Regression: Use models like LinearRegression, RandomForestRegressor, etc.

# Example: Using Linear Regression for a potential regression task
# If your target is categorical, replace with a classification model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel training complete.")




# --- Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# For Regression:
mse = mean_squared_error(y_test, y_pred)
# Calculate RMSE by taking the square root of MSE
import numpy as np
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation (Regression):")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# For Classification:
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Assuming y_pred for classification would be class labels
# y_pred_classes = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred_classes)
# print(f"\nModel Evaluation (Classification):")
# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred_classes))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred_classes))

# Visualize predictions vs actual (for regression)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs. Predicted Target")
# Add a diagonal line for reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.show()

# --- Further Steps ---
# - Hyperparameter tuning: Improve model performance by optimizing parameters (e.g., using GridSearchCV or RandomizedSearchCV).
# - Cross-validation: Get a more robust estimate of model performance.
# - Interpretability: Understand which features are most important for the prediction.
# - Deployment: Save the trained model for future use.
