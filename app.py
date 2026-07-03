import streamlit as st

import pandas as pd
import joblib

# Load the saved pipeline
pipeline = joblib.load('model.pkl')

st.title("Online Shopper Intention Predictor")
st.write("Enter session details to predict if the visitor will generate revenue.")

# Inputs for all features expected by the pipeline
admin = st.number_input("Administrative", min_value=0, value=0)
admin_dur = st.number_input("Administrative Duration", min_value=0.0, value=0.0)
info = st.number_input("Informational", min_value=0, value=0)
info_dur = st.number_input("Informational Duration", min_value=0.0, value=0.0)
prod_rel = st.number_input("Product Related", min_value=0, value=0)
prod_rel_dur = st.number_input("Product Related Duration", min_value=0.0, value=0.0)
bounce = st.number_input("Bounce Rates", min_value=0.0, max_value=1.0, value=0.0)
exit = st.number_input("Exit Rates", min_value=0.0, max_value=1.0, value=0.0)
page_val = st.number_input("Page Values", min_value=0.0, value=0.0)
special = st.number_input("Special Day", min_value=0.0, max_value=1.0, value=0.0)
month = st.selectbox("Month", ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
os = st.number_input("Operating Systems", min_value=1, value=1)
browser = st.number_input("Browser", min_value=1, value=1)
region = st.number_input("Region", min_value=1, value=1)
traffic = st.number_input("Traffic Type", min_value=1, value=1)
visitor = st.selectbox("Visitor Type", ['Returning_Visitor', 'New_Visitor', 'Other'])
weekend = st.checkbox("Weekend")

if st.button("Predict"):
    # Create DataFrame for prediction
    input_df = pd.DataFrame([[admin, admin_dur, info, info_dur, prod_rel, prod_rel_dur, 
                              bounce, exit, page_val, special, month, os, browser, 
                              region, traffic, visitor, weekend]],
                            columns=['Administrative', 'Administrative_Duration', 'Informational', 
                                     'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                                     'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 
                                     'Month', 'OperatingSystems', 'Browser', 'Region', 
                                     'TrafficType', 'VisitorType', 'Weekend'])
    
    prediction = pipeline.predict(input_df)
    
    if prediction[0]:
        st.success("The visitor is likely to generate Revenue!")
    else:
        st.info("The visitor is unlikely to generate Revenue.")
