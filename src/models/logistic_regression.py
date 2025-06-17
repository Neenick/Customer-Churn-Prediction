from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class BaseModel:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, max_iter=10000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]  # For ROC AUC
        
        # Accuracy
        print(f"Test Accuracy {self.model.score(X_test, y_test)}")

        # Confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Classification report (precision, recall, F1, support)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ROC AUC
        print("ROC AUC Score:")
        print(roc_auc_score(y_test, y_proba))