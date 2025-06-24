from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from data.dataset_loader import load_data
import os
import pandas

def run_model(model_class, model_name, X_train, y_train, X_test, y_test):
    model = model_class()
    model_file = f'saved_models/{model_name}.pkl'

    if model_name in ["logistic_regression", "random_forest", "xgboost"]:
        if os.path.exists(model_file):
            print(f"Loading saved {model_name} model...")
            model.load(model_file)
        else:
            print(f"Running grid search for {model_name}...")
            model.grid_search(X_train, y_train)
            model.save(model_file)
            print(f"{model_name} model saved.")
    else:
        model.train(X_train, y_train)
    
    model.evaluate(X_test, y_test)
    print(model.explain(X_train))

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X_train, X_test, y_train, y_test = load_data(data_path, split=0.2)
    #run_model(LogisticRegressionModel, "logistic_regression", X_train, y_train, X_test, y_test)
    #run_model(RandomForestModel, "random_forest", X_train, y_train, X_test, y_test)
    #run_model(XGBoostModel, "xgboost", X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()