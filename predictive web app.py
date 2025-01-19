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

# main function for Streamlit app
def main():
    # giving a title
    st.title('Pile Bearing Capacity Prediction Web App')

    # getting the input data from the user
    st.subheader("Enter the following parameters:")
    D = st.text_input('Pile Diameter (numeric value in meters)')
    Z1 = st.text_input('Z1 (numeric value)')
    Z2 = st.text_input('Z2 (numeric value)')
    Z3 = st.text_input('Z3 (numeric value)')
    Zp = st.text_input('Zp (numeric value)')
    Nsh = st.text_input('Nsh (numeric value)')
    Nt = st.text_input('Nt (numeric value)')

    # code for Prediction
    PileBearingCapacity = ''

    # creating a button for Prediction
    if st.button('Predict Pile Bearing Capacity'):
        try:
            # converting inputs to a list of floats
            input_data = [float(D), float(Z1), float(Z2), float(Z3), float(Zp), float(Nsh), float(Nt)]
            PileBearingCapacity = pilebearingcapacity_prediction(input_data)
        except ValueError:
            PileBearingCapacity = "Please enter valid numeric values for all inputs."
        except Exception as e:
            PileBearingCapacity = f"An error occurred: {str(e)}"

    # displaying the result
    st.success(PileBearingCapacity)

if __name__ == '__main__':
    main()
