import os

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "xgboost.pkl"),
}

# Replace with your actual feature names
FEATURE_NAMES = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

# Map categorical features to their options
FEATURE_OPTIONS = {
    'gender': ['Male', 'Female'],
    'SeniorCitizen': [0, 1],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    #'tenure': None,  # numeric
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No', 'No phone service'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No', 'No internet service'],
    'OnlineBackup': ['Yes', 'No', 'No internet service'],
    'DeviceProtection': ['Yes', 'No', 'No internet service'],
    'TechSupport': ['Yes', 'No', 'No internet service'],
    'StreamingTV': ['Yes', 'No', 'No internet service'],
    'StreamingMovies': ['Yes', 'No', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    #'MonthlyCharges': None,  # numeric
    #'TotalCharges': None     # numeric
}

TOTAL_FEATURE_LIST = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                      'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
                      'MultipleLines_No phone service', 'MultipleLines_Yes',
                      'InternetService_Fiber optic', 'InternetService_No',
                      'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                      'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                      'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                      'TechSupport_No internet service', 'TechSupport_Yes',
                      'StreamingTV_No internet service', 'StreamingTV_Yes',
                      'StreamingMovies_No internet service', 'StreamingMovies_Yes',
                      'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
                      'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                      'PaymentMethod_Mailed check']