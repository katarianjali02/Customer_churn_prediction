# 📉 Customer Churn Prediction

A complete, end-to-end machine learning project that predicts whether a telecom
customer is likely to churn (cancel their service), built with **Python** and
**scikit-learn**.

The project covers the full ML workflow: data cleaning, preprocessing,
training and comparing multiple classification models, evaluation, and
making predictions on new customer data.

---

## 🎯 Problem Statement

Customer churn — when a customer stops doing business with a company — is
one of the most important metrics for subscription-based businesses like
telecoms. This project uses the well-known **Telco Customer Churn** dataset
to build a model that flags customers at high risk of churning, so a
business could proactively intervene (offers, support outreach, etc.).

## 📊 Dataset

- **Source:** [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset, widely used on Kaggle)
- **Rows:** 7,043 customers
- **Target:** `Churn` (Yes / No)
- **Features:** demographics (gender, senior citizen, partner, dependents),
  account info (tenure, contract type, payment method, billing), and
  subscribed services (phone, internet, streaming, security add-ons, etc.)

The raw CSV is included at `data/Telco-Customer-Churn.csv`.

## 🗂️ Project Structure

```
churn-prediction/
├── data/
│   └── Telco-Customer-Churn.csv     # Raw dataset
├── notebooks/
│   └── EDA_and_Modeling.ipynb       # Exploratory analysis + walkthrough
├── src/
│   ├── data_preprocessing.py        # Cleaning, encoding, scaling, splitting
│   ├── train_model.py               # Trains & compares 6 models, saves the best
│   └── predict.py                   # Loads saved model and predicts on new data
├── models/                          # Saved model + encoders + scaler (generated)
├── outputs/                         # Metrics, plots, reports (generated)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ How It Works

### 1. Preprocessing (`src/data_preprocessing.py`)
- Drops the non-predictive `customerID` column
- Fixes `TotalCharges` (stored as text, blank for a handful of new customers) and imputes missing values with the median
- Label-encodes binary categorical columns (e.g. `gender`, `Partner`)
- One-hot encodes multi-class categorical columns (e.g. `Contract`, `PaymentMethod`)
- Scales numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`) with `StandardScaler`
- Splits data into train/test sets (80/20, stratified by churn)
- Saves the fitted encoders/scaler/feature list to `models/` so new data can be transformed identically at inference time

### 2. Model Training (`src/train_model.py`)
Trains and compares six classifiers:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.655 | 0.559 | 0.603 | 0.842 |
| Decision Tree | 0.797 | 0.675 | 0.455 | 0.543 | 0.828 |
| Random Forest | 0.790 | 0.634 | 0.492 | 0.554 | 0.826 |
| **Gradient Boosting** ⭐ | **0.799** | **0.654** | **0.516** | **0.577** | **0.843** |
| K-Nearest Neighbors | 0.779 | 0.587 | 0.559 | 0.573 | 0.823 |
| SVM | 0.795 | 0.655 | 0.481 | 0.555 | 0.795 |

*(Your exact numbers may vary slightly by random seed / environment.)*

The model with the best ROC-AUC is automatically selected and saved to
`models/churn_model.pkl`, along with:
- `outputs/model_comparison.csv` / `.json` — metrics for every model
- `outputs/classification_report.txt` — precision/recall/F1 for the best model
- `outputs/confusion_matrix.png`
- `outputs/roc_curves.png` — ROC curve comparison across all models
- `outputs/feature_importance.png` — top predictive features
- `outputs/model_comparison.png` — bar chart comparing all models

### 3. Prediction (`src/predict.py`)
Loads the saved model, encoders, and scaler, and exposes a `predict_churn()`
function that takes a raw customer record (a dict matching the original CSV
columns) and returns a churn prediction + probability. Run it directly to
see a sample prediction printed to the console.

## 🚀 Getting Started

### 1. Clone and set up the environment
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model
```bash
python src/train_model.py
```
This preprocesses the data, trains all candidate models, picks the best one
by ROC-AUC, and saves everything needed for inference to `models/` and all
evaluation artifacts to `outputs/`.

### 3. Run a prediction from the command line
```bash
python src/predict.py
```

### 4. Explore the notebook (optional)
```bash
jupyter notebook notebooks/EDA_and_Modeling.ipynb
```

## 🧠 Key Learnings / Techniques Demonstrated

- Handling messy real-world data (blank strings in a numeric column)
- Encoding a mix of binary and multi-class categorical features correctly
- Feature scaling for algorithms sensitive to feature magnitude
- Comparing multiple classification algorithms on the same train/test split
- Choosing an appropriate metric (ROC-AUC) for a moderately imbalanced
  target (~26% churn rate)
- Persisting a full inference pipeline (model + encoders + scaler + feature
  order) so predictions on new, unseen data are consistent with training
- Building a lightweight UI on top of a trained model with Streamlit

## 📌 Possible Next Steps

- Hyperparameter tuning (GridSearchCV / Optuna) for the top models
- Handle class imbalance explicitly with SMOTE or class weighting
- Add SHAP-based explainability for individual predictions
- Add automated tests (pytest) for the preprocessing and prediction functions
- Wrap `predict.py` in a simple REST API (e.g. Flask/FastAPI) if a frontend/backend is needed later

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
