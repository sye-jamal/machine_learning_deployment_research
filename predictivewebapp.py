# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 02:28:46 2025

@author: Haier
"""

import numpy as np
import pickle
import streamlit as st
import xgboost
from sklearn.preprocessing import MinMaxScaler

# loading the saved model
loaded_model = pickle.load(open('XGBoost_grid_search.sav', 'rb'))

# Define MinMaxScaler (recreate with the same scaling range and feature min/max values as during training)
scaler = pickle.load(open('minmax_scaler.sav', 'rb'))

# Define valid input ranges
VALID_RANGES = {
    "D": (300, 400),  # Example: Pile diameter in mm
    "Z1": (3.4, 4.75),     # Depth of first soil layer in m
    "Z2": (5.15, 8),     # Depth of second soil layer in m
    "Z3": (0, 1.22),     # Depth of third soil layer in m
    "Zp": (1.95, 3.4),   # Elevation of pile top in m
    "Nsh": (8.55, 13.63),    # Average SPT count along pile shaft
    "Nt": (6.71, 7.75),    # Average SPT count at pile tip
}

# creating a function for Prediction
def pilebearingcapacity_prediction(input_data):
    try:
        # converting the input_data to a numpy array
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)

        # reshaping the array for normalization
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # applying normalization
        normalized_data = scaler.transform(input_data_reshaped)

        # making the prediction with normalized data
        prediction = loaded_model.predict(normalized_data)
        return f"Predicted Pile Bearing Capacity: {prediction[0]:.2f}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Validate input ranges
def validate_input(value, param_name):
    try:
        value = float(value)
        min_val, max_val = VALID_RANGES[param_name]
        if not (min_val <= value <= max_val):
            return False, f"{param_name} must be between {min_val} and {max_val}."
        return True, ""
    except ValueError:
        return False, f"{param_name} must be a valid numeric value."

# main function for Streamlit app
def main():
    # giving a title
    st.title('Pile Bearing Capacity Prediction Web App')

    # getting the input data from the user
    st.subheader("Enter the following parameters:")

    error_messages = {}
    inputs = {}

    for param_name, (min_val, max_val) in VALID_RANGES.items():
        user_input = st.text_input(f"{param_name} ({min_val} to {max_val}):")
        is_valid, error_message = validate_input(user_input, param_name)
        if not is_valid:
            error_messages[param_name] = error_message
        inputs[param_name] = user_input

    # code for Prediction
    PileBearingCapacity = ''

    # creating a button for Prediction
    if st.button('Predict Pile Bearing Capacity'):
        if error_messages:
            for param, error in error_messages.items():
                st.error(f"Error in {param}: {error}")
        else:
            try:
                # converting valid inputs to a list of floats
                input_data = [float(inputs[param]) for param in VALID_RANGES]
                PileBearingCapacity = pilebearingcapacity_prediction(input_data)
            except Exception as e:
                PileBearingCapacity = f"An error occurred: {str(e)}"

    # displaying the result
    st.success(PileBearingCapacity)

    # Add important note for input ranges
    st.markdown("""
    **Important Note:**
    Please ensure that all input values are within the specified ranges:
    - **D**: Pile Diameter (300 or 400 mm)
    - **Z1**: Depth of first soil layer (3.4 to 4.75 m)
    - **Z2**: Depth of second soil layer (5.15 to 8 m)
    - **Z3**: Depth of third soil layer (0 to 1.22 m)
    - **Zp**: Elevation of pile top (1.95 to 3.4 m)
    - **Nsh**: Average SPT count along pile shaft (8.55 to 13.63)
    - **Nt**: Average SPT count at pile tip (6.71 to 7.75)
    """)

if __name__ == '__main__':
    main()
