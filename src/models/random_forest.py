from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd
import streamlit as st
from typing import List

class RandomForestModel:
    """A Random Forest classifier model with training, evaluation, and explanation functionality."""

    def __init__(self) -> None:
        """Initializes the Random Forest model."""
        self.model: RandomForestClassifier = RandomForestClassifier(random_state=0)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the Random Forest model.

        Args:
            X_train (pd.DataFrame): Training input features.
            y_train (pd.Series): Target feature for the training set.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluates the model and displays results in Streamlit.

        Applies a custom threshold (0.35) to predicted probabilities.

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

    def grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        """Performs grid search to find the best hyperparameters.

        Args:
            X_train (pd.DataFrame): Training input features.
            y_train (pd.Series): Target feature for the training set.

        Returns:
            RandomForestClassifier: The best model found by grid search.
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=5)
        grid.fit(X_train, y_train)
        print("Best parameters:", grid.best_params_)
        print("Best score:", grid.best_score_)
        self.model = grid.best_estimator_
        return self.model

    def explain(self, feature_names: List[str], X_train: pd.DataFrame) -> pd.DataFrame:
        """Returns the top 10 most important features based on feature importances.

        Args:
            feature_names (List[str]): List of feature names.
            X_train (pd.DataFrame): Training input features (not used directly, included for consistency).

        Returns:
            pd.DataFrame: DataFrame with top features and their importance scores.
        """
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        df = df.sort_values(by='Importance', ascending=False).head(10).reset_index(drop=True)
        return df

    def save(self, filename: str) -> None:
        """Saves the model to a file.

        Args:
            filename (str): File path to save the model.
        """
        joblib.dump(self.model, filename)

    def load(self, filename: str) -> None:
        """Loads a model from a file.

        Args:
            filename (str): File path to load the model from.
        """
        self.model = joblib.load(filename)