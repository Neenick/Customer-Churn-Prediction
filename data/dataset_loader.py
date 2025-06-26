from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str, split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads, preprocesses, and splits the Telco customer churn dataset.

    Args:
        path (str): Path to the CSV data file.
        split (float, optional): Fraction of data to use as test set. Defaults to 0.2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            X_train, X_test, y_train, y_test datasets.
    """
    data = pd.read_csv(path)

    # Drop customer ID column as it is irrelevant for prediction
    data = data.drop('customerID', axis=1)

    # Convert 'TotalCharges' to numeric and drop rows with NaN values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna()

    # Map churn labels from 'Yes'/'No' to 1/0
    data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

    # One-hot encode categorical variables (drop first to avoid multicollinearity)
    data = pd.get_dummies(data, drop_first=True)

    # Separate features and target variable
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0)

    return X_train, X_test, y_train, y_test
