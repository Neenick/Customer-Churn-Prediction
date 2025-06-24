# Customer Churn Prediction

This project predicts customer churn for a telecom company using machine learning. The goal is to identify which customers are likely to leave, so the company can take action to retain them.

---

## Dataset

The dataset used is [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn), which contains customer information and their churn status.

---

## 📁 Project Structure

```bash
├───app
│   └───app.py
├───data
│   ├───dataset_loader.py
│   └───WA_Fn-UseC_-Telco-Customer-Churn.csv
├───src
│   └───models
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

| Model               | F1 Score | ROC AUC Score | Notes                                     |
|---------------------|----------|---------------|-------------------------------------------|
| Logistic Regression | 0.63     | 0.837         | Best overall, interpretable, simple       |
| Random Forest       | 0.62     | 0.840         | Slightly lower, may overfit on small data |
| XGBoost             | 0.63     | 0.843         |                                           |

---

### Feature Importance

The table below shows the top 10 features ranked by their combined importance across three different models: Logistic Regression, Random Forest, and XGBoost. Out of the 20 features, each model ranked its top 10 features from 10 (most important) down to 1 point. We summed these points from all three models to get an overall score for each feature.

| Rank | Feature                | Points |
|-------|-----------------------|--------|
| 1     | Contract Two Year      | 25     |
| 2     | Contract One Year      | 19     |
| 3     | Tenure                | 19     |
| 4     | Payment Elect Check    | 18     |
| 5     | InternetService Optic  | 18     |
| 6     | Total Charges          | 12     |
| 7     | Monthly Charges        | 12     |
| 8     | Online Security       | 9      |
| 9     | Tech Support          | 9      |
| 10    | MultipleLines No Phone | 8      |

**Notes:**
- This point system helps combine feature rankings from different models into a single overview.
- Features with higher total points were consistently important across models.

---

## Insights



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