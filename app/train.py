import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
df = pd.read_csv('E:\\PlantPulse-main\\Data-raw\\Crop_recommendation.csv')

# Step 2: Inspect the data (optional)
print(df.head())  # Check the first few rows
print(df.info())  # Check for missing values or data types

# Step 3: Define the features (X) and the target (y)
X = df[['N', 'P', 'K', 'ph', 'humidity', 'temperature', 'rainfall']]  # Input features
y = df['label']  # Target variable (Crop recommendation)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Save the model to a file
joblib.dump(model, 'models\crop_recommendation_model.pkl')
