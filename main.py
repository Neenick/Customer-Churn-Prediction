from src.models.logistic_regression import BaseModel
from data.dataset_loader import load_data

import os

# Get path to the CSV file relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

X_train, X_test, y_train, y_test = load_data(data_path, split=0.2)

model = BaseModel()
model.train(X_train, y_train)
model.evaluate(X_test, y_test)