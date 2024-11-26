import streamlit as st
import pickle
import numpy as np

# Load the trained model from the .pkl file
with open('crop_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit interface
st.title("Crop Recommendation System 🌱")

# Input fields for the user to enter data
st.header("Enter the following details:")
nitrogen = st.number_input("Nitrogen content (in soil):", min_value=0, max_value=200, value=0)
phosphorus = st.number_input("Phosphorus content (in soil):", min_value=0, max_value=200, value=0)
potassium = st.number_input("Potassium content (in soil):", min_value=0, max_value=200, value=0)
temperature = st.number_input("Temperature (°C):", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("pH level (soil):", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0)

# Button for prediction
if st.button("Get Recommendation"):
    # Prepare the input data
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    # Make a prediction
    prediction = model.predict(input_data)[0]
    
    # Display the recommendation
    st.success(f"Recommended Crop for Cultivation is: **{prediction}** 🌾")

# Footer or additional info
st.write("Developed with ❤️ using Streamlit.")
