import streamlit as st
import pickle
import numpy as np

# Load the logistic regression model
with open("logistic_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("Heart Disease Risk Prediction")

# Define input fields for each feature
age = st.number_input("Age", min_value=0, max_value=120, value=50)
# Repeat for other features (cigsPerDay, sysBP, diaBP, etc.)

# Prediction button
if st.button("Predict"):
    # Collect inputs into an array
    input_data = np.array([[age, ...]])  # Add other features here
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.write("The model predicts a high risk of coronary heart disease.")
    else:
        st.write("The model predicts a low risk of coronary heart disease.")
