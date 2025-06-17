import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path, split=0.2):
    data = pd.read_csv(path)

    # Drop Customer ID (irrelevant)
    data = data.drop('customerID', axis=1)

    # Remove missing values. Only Total Charges are sometimes NaN when a customer is new (tenure = 0)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna()

    # Convert Churn to numerical
    data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

    # Convert to numeric
    data = pd.get_dummies(data, drop_first=True)

    # Split input features & output feature
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=0)


    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)


    return X_train, X_test, y_train, y_test