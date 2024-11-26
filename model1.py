# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
data = pd.read_csv("Crop recommendation changed.csv")

# Step 2: Inspect the dataset structure (Optional for debugging)
print(data.head())

# Step 3: Split the dataset into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column is the target (crop labels)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the trained model to a .pkl file
with open("crop_recommendation_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model has been trained and saved as 'crop_recommendation_model.pkl'")
