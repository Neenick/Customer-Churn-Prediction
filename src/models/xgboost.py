from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import shap
import streamlit as st
import pandas as pd
from typing import List

class XGBoostModel:
    """An XGBoost classifier model with training, evaluation, explanation, and persistence support."""

    def __init__(self) -> None:
        """Initializes the XGBoost model with default parameters."""
        self.model: XGBClassifier = XGBClassifier(eval_metric='logloss', random_state=0)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the XGBoost model.

        Args:
            X_train (pd.DataFrame): Training input features.
            y_train (pd.Series): Target feature for the training set.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluates the model and displays results in Streamlit.

        Applies a custom probability threshold of 0.35 for classification.

        Args:
            X_test (pd.DataFrame): Test input features.
            y_test (pd.Series): Target feature for the test set.
        """
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.35).astype(int)

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(cr).transpose())

        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["F1 Score", "ROC AUC"],
            "Score": [f"{f1:.4f}", f"{auc:.4f}"]
        })
        st.table(metrics_df)

    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """Performs grid search to find the best hyperparameters.

        Args:
            X_train (pd.DataFrame): Training input features.
            y_train (pd.Series): Target feature for the training set.

        Returns:
            XGBClassifier: The best model found by grid search.
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1]
        }

        grid = GridSearchCV(XGBClassifier(eval_metric='auc', random_state=0), param_grid, cv=5)
        grid.fit(X_train, y_train)

        print("Best parameters:", grid.best_params_)
        print("Best score:", grid.best_score_)

        self.model = grid.best_estimator_
        return self.model

    def explain(self, feature_names: List[str], X_train: pd.DataFrame) -> shap.Explanation:
        """Generates and plots SHAP values for the model.

        Args:
            feature_names (List[str]): List of feature names (not used directly in SHAP but may help elsewhere).
            X_train (pd.DataFrame): Input features used to compute SHAP values.

        Returns:
            shap.Explanation: SHAP values for the input data.
        """
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, max_display=10)
        return shap_values

    def save(self, filename: str) -> None:
        """Saves the model to a file using joblib.

        Args:
            filename (str): Path to save the model.
        """
        joblib.dump(self.model, filename)

    def load(self, filename: str) -> None:
        """Loads the model from a file using joblib.

        Args:
            filename (str): Path to the saved model file.
        """
        self.model = joblib.load(filename)