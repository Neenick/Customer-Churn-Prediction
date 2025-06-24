import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel

# Paths to your saved models
MODEL_PATHS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

# Replace with your actual feature names
feature_names = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
       'MonthlyCharges', 'TotalCharges']

# Map categorical features to their options
feature_options = {
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

total_feature_list = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
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

st.title("Customer Churn Prediction with Explainability")

# Option selection
mode = st.radio("Select what you want to do:", ["Model Evaluation", "Single Prediction"])

# === MODEL EVALUATION ===
if mode == "Model Evaluation":
    st.header("Model Evaluation")

    # 1. Model Selector
    model_choice = st.selectbox("Choose model:", ["Logistic Regression", "Random Forest", "XGBoost"])

    # 2. Train (If no model saved)
    # Load and evaluate the model here (e.g., show accuracy, SHAP, feature importance, etc.)

# === SINGLE PREDICTION ===
elif mode == "Single Prediction":
    # 1. Model selector
    model_choice = st.selectbox("Choose your model", list(MODEL_PATHS.keys()))

    # 2. Load model based on choice
    if model_choice == "Logistic Regression":
        model = LogisticRegressionModel()
    elif model_choice == "Random Forest":
        model = RandomForestModel()
    elif model_choice == "XGBoost":
        model = XGBoostModel()
    model.load('saved_models/' + MODEL_PATHS[model_choice])

    # 3. User inputs
    def user_input_features():
        data = {}
        for feature in feature_names:
            if feature in feature_options:
                val = st.selectbox(f'Select {feature}', options=feature_options[feature])
                data[feature] = val
            else:
                val = st.number_input(f'Input value for {feature}', value=0)
                data[feature] = val

        # Convert to pandas and make numeric
        df = pd.DataFrame(data, index=[0])
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=total_feature_list, fill_value=0)
        return df

    input_df = user_input_features()

    if st.button('Predict and Explain'):
        prediction = model.model.predict(input_df)[0]
        prediction_proba = model.model.predict_proba(input_df)[0, 1]
        
        st.write(f"Model: {model_choice}")
        st.write(f"Prediction: {'Churn' if prediction == 1 else 'No churn'}")
        st.write(f"Churn Probability: {prediction_proba:.2f}")
        
        st.write(f"Feature importance for {model_choice}:")

        if model_choice == "XGBoost":
            shap_values = model.explain(total_feature_list, input_df)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, input_df, max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            importance = model.explain(total_feature_list, input_df)  # make sure explain returns something, e.g., dict or list of tuples
            st.write(importance)

