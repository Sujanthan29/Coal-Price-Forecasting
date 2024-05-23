# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:12:26 2024

@author: sujan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:55:17 2024

@author: sujan
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from custom_models import CustomKerasRegressor 


# Load the model and preprocessing objects

# Load the saved model
with open('lstm.pkl', 'rb') as file:
    model = pickle.load(file)  # Load the saved model, change the filename if needed
impute_data = joblib.load('impute.pkl')
winzor = joblib.load('winzor.pkl')

# Function to preprocess and make predictions
def predict(data):
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    # Apply preprocessing pipelines
    newprocessed1 = pd.DataFrame(impute_data.transform(data), columns=data.columns)
    newprocessed2 = pd.DataFrame(winzor.transform(newprocessed1), columns=newprocessed1.columns)
    # Make predictions
    predictions = pd.DataFrame(model.predict(newprocessed2), columns=['Forecasted_Value'])

    return predictions


# Streamlit interface
def main():
    st.title('COAL PRICE FORECASTING')

    # Get data from the user
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Print column names for debugging
        print("Column Names in DataFrame:")
        print(data.columns)

        # Display the column names using Streamlit
        st.subheader('Column Names:')
        st.write(data.columns)

        # Display the uploaded data
        st.subheader('Uploaded data:')
        st.write(data)

        # Make predictions using a predefined column
        predefined_column = 'Forecasted_Value'
        predictions = predict(data[[predefined_column]])

        # Display results
        st.subheader('Forecast Results:')
        # Display forecasted values
        st.write("Forecasted Values:")
        st.write(predictions)

        # Display forecasted graph using st.line_chart()
        num_days = 30  # Adjust this according to your needs
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(start=pd.Timestamp.today(), periods=num_days),
            'Forecasted Values': predictions.squeeze()
        })
        st.line_chart(forecast_df.set_index('Date'))

if __name__ == "__main__":
    main()
