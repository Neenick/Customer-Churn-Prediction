from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=0)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.35).astype(int) 

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print(f"Test Accuracy: {self.model.score(X_test, y_test)}")

        print("F1 Score:", f1_score(y_test, y_pred))

        print("ROC AUC Score:")
        print(roc_auc_score(y_test, y_proba))

    def grid_search(self, X_train, y_train):
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
    
    def explain(self, feature_names):
        importances = self.model.feature_importances_
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"{name}: {importance:.4f}")

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)