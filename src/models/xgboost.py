from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib

class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=0)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print(f"Test Accuracy: {self.model.score(X_test, y_test)}")

        print("F1 Score:", f1_score(y_test, y_pred))

        print("ROC AUC Score:")
        print(roc_auc_score(y_test, y_proba))

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)