import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the crop rate data
rate_data = pd.read_csv(r"C:\Users\LENOVO-\OneDrive\Desktop\Project 1\Datasets project 1\3 years market values with dates .csv")

# Check the columns to ensure correct naming
print(rate_data.columns)

# Drop irrelevant columns and set features and target variable
X = rate_data.drop(columns=['Crop', 'Variety', 'Arrival_Date'])  # Features (we drop Crop, Variety, and Arrival_Date)
y = rate_data[['Crop', 'Variety']]  # Target variable (Crop and Variety)

# Convert categorical target to numeric (encoding)
y_encoded = pd.get_dummies(y['Crop'], drop_first=True)  # Assuming we need only one-hot encoding for crop types

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y_encoded)

# Save the trained model to a .pkl file
with open("crop_recommendation_model_with_rate.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
