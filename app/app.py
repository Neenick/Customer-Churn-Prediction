import os
import sys

# Voeg project root toe aan sys.path zodat 'src' importeert werkt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Tuple, Union

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from main import load_data
from config import DATA_PATH, MODEL_PATHS, FEATURE_NAMES, FEATURE_OPTIONS, TOTAL_FEATURE_LIST

from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel

X_train, X_test, y_train, y_test = load_data(DATA_PATH, split=0.2)

def select_model() -> Tuple[str, Union[LogisticRegressionModel, RandomForestModel, XGBoostModel], str]:
    """Let user select a model and returns model info.

    Returns:
        Tuple[str, Union[LogisticRegressionModel, RandomForestModel, XGBoostModel], str]:
            model_choice: The name of the chosen model.
            model: The instantiated model object.
            model_file: The filename for the saved model.
    """
    # 1. Model selector
    model_choice = st.selectbox("Choose your model", list(MODEL_PATHS.keys()))

    # 2. Load model based on choice
    if model_choice == "Logistic Regression":
        model = LogisticRegressionModel()
    elif model_choice == "Random Forest":
        model = RandomForestModel()
    elif model_choice == "XGBoost":
        model = XGBoostModel()
    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    model_file = MODEL_PATHS[model_choice]
    return model_choice, model, model_file



st.title("Customer Churn Prediction with Explainability")

# Option selection
mode = st.radio("Select what you want to do:", ["Model Evaluation", "Single Prediction"])

# === MODEL EVALUATION ===
if mode == "Model Evaluation":
    st.header("Model Evaluation")

    # Optional: show dataset
    if st.checkbox("Show raw dataset"):
        st.subheader("Raw Dataset")
        st.dataframe(pd.concat([X_train, y_train], axis=1))

    # 1. Select Model
    model_choice, model, model_file = select_model()
    model_trained = os.path.exists(model_file)

    # 2. Train (If no model saved)
    if not model_trained:
        st.warning("This model has not been trained yet.")
        if st.button(f"Train {model_choice} Model"):
            with st.spinner("Training model with Grid Search..."):
                model.grid_search(X_train, y_train)
                model.save(model_file)
            st.success("Training completed and model saved!")
            st.rerun()
    else:
        st.success("Model is trained.")
        if st.button("Evaluate Model"):
            model.load(model_file)
            model.evaluate(X_test, y_test)

    # Load and evaluate the model here (e.g., show accuracy, SHAP, feature importance, etc.)

# === SINGLE PREDICTION ===
elif mode == "Single Prediction":
    # 1. Select Model
    model_choice, model, model_file = select_model()
    model.load(model_file)

    # 2. User inputs
    def user_input_features():
        data = {}
        for feature in FEATURE_NAMES:
            if feature in FEATURE_OPTIONS:
                val = st.selectbox(f'Select {feature}', options=FEATURE_OPTIONS[feature])
                data[feature] = val
            else:
                val = st.number_input(f'Input value for {feature}', value=0)
                data[feature] = val

        # Convert to pandas and make numeric
        df = pd.DataFrame(data, index=[0])
        df = pd.get_dummies(df, drop_first=True)
        df = df.reindex(columns=TOTAL_FEATURE_LIST, fill_value=0)
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
            shap_values = model.explain(TOTAL_FEATURE_LIST, input_df)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, input_df, max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            importance = model.explain(TOTAL_FEATURE_LIST, input_df)  # make sure explain returns something, e.g., dict or list of tuples
            st.write(importance)

