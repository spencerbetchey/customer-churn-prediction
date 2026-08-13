import pandas as pd


def load_raw_data(path='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """Load the raw Telco churn dataset."""
    return pd.read_csv(path)


def clean_data(df):
    """
    Clean and encode the raw Telco churn dataset.

    - Fixes TotalCharges (blank values for 0 tenure customers get set to 0, cast to numbers)
    - Drops customerID (non predictive)
    - Collapses "No internet/phone service" into "No" for redundant columns
    - Encodes binary columns as 0/1
    - One hot encodes multi category columns (InternetService, Contract, PaymentMethod)
    """
    df = df.copy()

    # Fix TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Drop non predictive identifier
    df = df.drop('customerID', axis=1)

    # Collapse redundant "No service" categories
    no_internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in no_internet_cols:
        df[col] = df[col].replace('No internet service', 'No')
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

    # Binary encode Yes/No (and gender) columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # One hot encode multi category columns
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'],
                         drop_first=True)

    return df