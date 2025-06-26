from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import List

class LogisticRegressionModel:
    """A logistic regression model with training, evaluation, and explanation utilities."""

    def __init__(self) -> None:
        """Initializes a Logistic Regression model with balanced class weights."""
        self.model: LogisticRegression = LogisticRegression(
            class_weight='balanced',
            random_state=0,
            max_iter=10000
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the logistic regression model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Target feature for the training set.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluates the model and displays results in Streamlit.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): Target feature for the test set.
        """
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

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

    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        """Performs grid search to find the best hyperparameters.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Target feature for the training set.

        Returns:
            LogisticRegression: The best estimator found by grid search.
        """
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l2']
        }
        grid = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=10000), param_grid, cv=5)
        grid.fit(X_train, y_train)
        print("Best parameters:", grid.best_params_)
        print("Best score:", grid.best_score_)
        self.model = grid.best_estimator_
        return self.model

    def explain(self, feature_names: List[str], X_train: pd.DataFrame) -> pd.DataFrame:
        """Returns the top 10 most important features based on model coefficients.

        Args:
            feature_names (List[str]): Names of the input features.
            X_train (pd.DataFrame): Training data (not used directly).

        Returns:
            pd.DataFrame: DataFrame with top features and their coefficients.
        """
        coefs = self.model.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        })
        coef_df['Abs(Coefficient)'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values(by='Abs(Coefficient)', ascending=False).head(10).reset_index(drop=True)
        return coef_df

    def save(self, filename: str) -> None:
        """Saves the model to a file.

        Args:
            filename (str): Path to save the model.
        """
        joblib.dump(self.model, filename)

    def load(self, filename: str) -> None:
        """Loads a model from a file.

        Args:
            filename (str): Path to the saved model file.
        """
        self.model = joblib.load(filename)