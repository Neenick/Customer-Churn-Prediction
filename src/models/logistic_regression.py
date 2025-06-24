from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import pandas as pd


class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight='balanced', random_state=0, max_iter=10000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # For ROC AUC

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print(f"Test Accuracy: {self.model.score(X_test, y_test):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    def grid_search(self, X_train, y_train):
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
    
    def explain(self, X_train):
        coefs = self.model.coef_[0]  # Get the coefficients
        feature_names = X_train.columns
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefs
        })
        coef_df['Abs(Coefficient)'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values(by='Abs(Coefficient)', ascending=False)
        print(coef_df.head(10))

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)