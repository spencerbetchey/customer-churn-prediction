"""
Customer Churn Predictor - Streamlit App (Deployed on Streamlit Community Cloud)

Loads a pre trained Logistic Regression model and scaler to predict churn probability
for a hypothetical customer, based on interactive form inputs. Displays results as a
Low/Medium/High risk level along with the raw probability.

Note: since the model was trained on a NumPy array (not a named DataFrame),
feature_names_in_ is unavailable. Column order is instead hardcoded to match the exact
training column order, and enforced before scaling/prediction.
"""
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("Customer Churn Predictor")
st.write("This app predicts whether a customer is likely to churn.")

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method",
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

if st.button("Predict"):
    input_dict = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'MultipleLines': 1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'OnlineBackup': 1 if online_backup == "Yes" else 0,
        'DeviceProtection': 1 if device_protection == "Yes" else 0,
        'TechSupport': 1 if tech_support == "Yes" else 0,
        'StreamingTV': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies': 1 if streaming_movies == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': monthly_charges * tenure,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    expected_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_Fiber optic', 'InternetService_No',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check']

    input_df = input_df[expected_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.3:
        risk_level = "Low"
        risk_color = "green"
    elif probability < 0.6:
        risk_level = "Medium"
        risk_color = "orange"
    else:
        risk_level = "High"
        risk_color = "red"

    st.markdown(f"### Risk Level: :{risk_color}[{risk_level}]")
    st.write(f"Churn Probability: {probability:.1%}")

    with st.expander("See model input details"):
        st.write(input_df)