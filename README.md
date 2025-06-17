# Customer Churn Prediction

This project predicts customer churn for a telecom company using machine learning. The goal is to identify which customers are likely to leave, so the company can take action to retain them.

---

## Dataset

The dataset used is [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn), which contains customer information and their churn status.

---

## 📁 Project Structure

```bash
├───app
│   ├───app.py
├───data
│   ├───dataset_loader.py
│   └───WA_Fn-UseC_-Telco-Customer-Churn.csv
├───src
│   ├───models
│       ├───logistic_regression.py
│       └───random_forest.py
├───main.py
├───README.md
├───requirements.txt
```

- `data/` — Dataset and data loading scripts  
- `src/models/` — Machine learning models (logistic regression, random forest)  
- `main.py` — Script to train and evaluate models  
- `README.md` — Project overview and documentation

---

## Preprocessing

- Removed irrelevant features (e.g., customer ID)  
- Converted categorical variables to numerical using one-hot encoding  
- Handled missing values in `TotalCharges` by converting to numeric and dropping NaNs  
- Scaled features (for logistic regression)

---

## Models

Two models were implemented and compared:

- **Logistic Regression**: A linear model suitable for binary classification.  
- **Random Forest Classifier**: An ensemble of decision trees that captures non-linear relationships.

---

## Results

| Model               | Accuracy | ROC AUC Score | Notes                                    |
|---------------------|----------|---------------|------------------------------------------|
| Logistic Regression  | 80.7%    | 0.86          | Best overall, interpretable, simple      |
| Random Forest       | 79.5%    | 0.84          | Slightly lower, may overfit on small data |

---

## Insights

Despite Random Forest being more complex, Logistic Regression performed better. This indicates that the relationship between the features and churn is mostly linear. Logistic Regression also offers better interpretability, which is valuable for business decisions and explaining results to stakeholders.

---

## How to Run

1. Clone the repository  
2. Install dependencies from `requirements.txt`  
3. Run `main.py` to train and evaluate models

---

## Future Work

- Hyperparameter tuning using GridSearchCV  
- Adding more advanced models (e.g., XGBoost)  
- Creating a Streamlit dashboard for interactive predictions  
- Applying cross-validation for more robust evaluation  

---

## Contact

Yannick van Maanen — yannick@van-maanen.net 
GitHub: [Neenick](https://github.com/Neenick)