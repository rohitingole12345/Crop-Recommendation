import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model for crop recommendation
with open('crop_recommendation_model.pkl', 'rb') as file:
    crop_model = pickle.load(file)

# Load the trained model for price prediction (price model)
with open('crop_recommendation_model_with_rate.pkl', 'rb') as file:
    price_model = pickle.load(file)

# Load the crop rate data for displaying prices
rate_data = pd.read_csv(r"./3 years market values with dates .csv")

# Streamlit interface
st.title("Crop Recommendation and Pricing System 🌱")

# Input fields for the user to enter data
st.header("Enter the following details:")

nitrogen = st.number_input("Nitrogen content (in soil):", min_value=0, max_value=200, value=0)
phosphorus = st.number_input("Phosphorus content (in soil):", min_value=0, max_value=200, value=0)
potassium = st.number_input("Potassium content (in soil):", min_value=0, max_value=200, value=0)
temperature = st.number_input("Temperature (°C):", min_value=-10.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("pH level (soil):", min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input("Rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0)
date = st.date_input("Date:")

# Button for prediction
if st.button("Get Recommendation"):
    # Prepare the input data for crop prediction
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    # Predict the crop recommendation
    crop_prediction = crop_model.predict(input_data)[0]
    
    # Use crop variety for price prediction
    # Filter the rate_data for the specific crop
    crop_rate = rate_data[rate_data['Crop'] == crop_prediction]
    
    # If there are no prices for this crop, give a warning
    if crop_rate.empty:
        st.warning(f"No price data available for {crop_prediction}.")
    else:
        # Extract min and max price from the rate data
        min_price = crop_rate['Min_Price'].values[0]
        max_price = crop_rate['Max_Price'].values[0]
        
        # Display crop recommendation and prices
        st.success(f"Recommended Crop for Cultivation is: **{crop_prediction}** 🌾")
        st.write(f"Minimum Price: ₹{min_price}")
        st.write(f"Maximum Price: ₹{max_price}")
    
    # Now use the crop variety as input to price_model (this assumes price_model was trained with crop variety as a feature)
    # Example: If price_model uses crop variety (encoded as numerical values), encode or pass the crop variety
    # You can adjust the input data based on your model's expectations
    crop_variety_input = np.array([[crop_prediction]])  # Adjust if necessary
    
    # Predict the price range using the price_model
    try:
        price_prediction = price_model.predict(crop_variety_input)
        st.write(f"Predicted Price Range for {crop_prediction}: ₹{price_prediction[0][0]} - ₹{price_prediction[0][1]}")
    except:
        pass
        
    
# Footer or additional info
st.write("Developed with ❤️ using Streamlit.")
