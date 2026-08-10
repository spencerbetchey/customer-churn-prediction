"""
Customer Churn Predictor - Streamlit App

Loads a pre trained Logistic Regression model to predict churn probability
for a given customer, based on inputs from the Telco Customer Churn dataset.
"""
import streamlit as st

st.title("Customer Churn Predictor")
st.write("This app predicts whether a customer is likely to churn.")