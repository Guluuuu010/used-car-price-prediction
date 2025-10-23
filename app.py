# app.py — Used Car Price Prediction (Professional Clean Version)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    layout="wide"
)

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("best_used_car_model.joblib")

st.title("Used Car Price Prediction App")
st.markdown("""
This application estimates the resale price of a used car based on its characteristics.  
Adjust the parameters in the sidebar and click **Predict Price** to get an estimated value.
""")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Enter Car Details")

brand = st.sidebar.selectbox("Brand", ["Toyota", "Honda", "Ford", "BMW", "Hyundai", "Other"])
model_name = st.sidebar.text_input("Model Name", "Corolla")
model_year = st.sidebar.number_input("Model Year", min_value=1990, max_value=2025, value=2018)
milage = st.sidebar.number_input("Mileage (in km driven)", min_value=0, max_value=400000, value=50000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "CNG"])
engine = st.sidebar.number_input("Engine Size (in Liters)", min_value=0.5, max_value=6.0, value=1.5)
transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual"])
ext_col = st.sidebar.selectbox("Exterior Color", ["White", "Black", "Silver", "Blue", "Red", "Other"])
int_col = st.sidebar.selectbox("Interior Color", ["Black", "Gray", "Other"])
accident = st.sidebar.selectbox("Accident History", ["No Accident", "Minor", "Major"])
clean_title = st.sidebar.selectbox("Title Status", ["Clean", "Salvage", "Rebuilt"])

# -------------------------------
# Feature Engineering (match training)
# -------------------------------
current_year = 2025
car_age = current_year - model_year
price_per_milage = 0 if milage == 0 else (10000 / milage)

# -------------------------------
# Encoding (same logic as training)
# -------------------------------
fuel_map = {"Petrol": 0, "Diesel": 1, "Hybrid": 2, "Electric": 3, "CNG": 4}
trans_map = {"Manual": 0, "Automatic": 1}
brand_map = {"Toyota": 0, "Honda": 1, "Ford": 2, "BMW": 3, "Hyundai": 4, "Other": 5}
color_map = {"White": 0, "Black": 1, "Silver": 2, "Blue": 3, "Red": 4, "Gray": 5, "Other": 6}
acc_map = {"No Accident": 0, "Minor": 1, "Major": 2}
title_map = {"Clean": 0, "Salvage": 1, "Rebuilt": 2}

# -------------------------------
# Create DataFrame (match training columns exactly)
# -------------------------------
input_data = pd.DataFrame([{
    'brand': brand_map[brand],
    'model': 0,  # Placeholder if model name was encoded during training
    'model_year': model_year,
    'milage': milage,
    'fuel_type': fuel_map[fuel_type],
    'engine': engine,
    'transmission': trans_map[transmission],
    'ext_col': color_map[ext_col],
    'int_col': color_map[int_col],
    'accident': acc_map[accident],
    'clean_title': title_map[clean_title],
    'car_age': car_age,
    'price_per_milage': price_per_milage
}])

st.subheader("Input Summary")
st.dataframe(input_data)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Car Price: ${predicted_price:,.2f}")

    # ---------------------------
    # Visualization Section
    # ---------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted Price vs Mileage")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.scatterplot(x=[milage], y=[predicted_price], color='red', s=150)
        ax.set_xlabel("Mileage (km)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title("Predicted Price vs Mileage")
        st.pyplot(fig)

    with col2:
        st.subheader("Predicted Price vs Car Age")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        sns.lineplot(x=[car_age], y=[predicted_price], marker='o', color='green')
        ax2.set_xlabel("Car Age (Years)")
        ax2.set_ylabel("Predicted Price ($)")
        ax2.set_title("Car Age Impact on Price")
        st.pyplot(fig2)

    # ---------------------------
    # Insight Section
    # ---------------------------
    st.markdown("### Model Insights")
    if car_age > 10:
        st.warning("Older car detected — resale value tends to decrease after 10 years.")
    elif milage > 100000:
        st.info("High mileage — the car’s value might be reduced due to usage.")
    else:
        st.success("This car appears fairly new and well-maintained — expect a strong resale value.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
---
Adjust the parameters in the sidebar to see how the predicted price changes.  
Model trained using **XGBoost (R² ≈ 0.96)** for high accuracy.
""")
