Tashkent Housing Price Prediction: Overcoming Overfitting in Regression

An end-to-end Machine Learning regression pipeline built to predict apartment prices in Tashkent, Uzbekistan. This project focuses on solid software engineering principles applied to data science, specifically addressing data cleaning, modular preprocessing pipelines, detecting model overfitting using 10-Fold Cross-Validation, and deploying a robust Random Forest ensemble model.

📌 Project Overview

The real estate market in Tashkent is highly dynamic but heavily plagued by extreme data anomalies, typos, and outliers. The objective of this project was to construct a robust machine learning system capable of handling messy, non-numeric values and predicting housing prices with high generalization stability.

Key Highlights:

Data Health Assessment: Solved data type anomalies where numeric fields (size and price) were stored as messy object types using pd.to_numeric with error coercing.

Rigorous Outlier Filtering: Sanitized the data by filtering realistic market limits (apartment sizes between $15$ and $500$ sq.m, prices between $\$5\text{k}$ and $\$1\text{M}$, and validating floors).

Modular Architecture: Built a unified preprocessing pipeline using Scikit-Learn's Pipeline and ColumnTransformer to avoid data leakage.

Overfitting Diagnostics: Exposed the illusion of a greedy Decision Tree's low training error ($\$7,013.26$) using 10-Fold Cross-Validation ($\$29,768.51$).

Ensemble Resolution: Deployed a robust Random Forest Regressor yielding a highly stable error of $\$21,678.79$ on completely untouched test data.

📂 Repository Structure

Your project directory is structured as follows:

TASHKENT_HOUSING_PREDICTION/
├── data/
│   └── housing_data_08-02-2021.csv   # Tashkent Real Estate Raw Dataset
├── 05_ml_05_amaliyot.ipynb          # End-to-End ML Pipeline (Jupyter Notebook)
└── README.md                         # Professional Project Documentation


⚙️ How the Pipeline Works

The system cleanly separates the data flow to ensure there is absolutely no data leakage from the testing set into the training steps.

1. Preprocessing Steps:

Train/Test Split: Locked away $20\%$ of the data using a strict train_test_split(..., test_size=0.2, random_state=21).

Feature Selection: Selected district over location because they represent almost the exact same geographical information, reducing dimensionality and redundant features.

Numerical Features ['rooms', 'size', 'level', 'max_levels'] $\rightarrow$ Standardized using StandardScaler.

Categorical Features ['district'] $\rightarrow$ Encoded using OneHotEncoder(handle_unknown='ignore') for production safety (gracefully handles unseen districts).

📊 Performance Comparison & Evaluation

The models were evaluated using Root Mean Squared Error ($RMSE$). A lower $RMSE$ indicates better pricing predictions (errors are represented in USD $).

Model

Evaluation Method

Training RMSE

Validation Mean RMSE

Status / Verdict

Linear Regression

Training Set Evaluation

$26,582.97

N/A

Underfitting (Too simple)

Decision Tree Regressor

Training Set Evaluation

$7,013.26

N/A

Overfitting (Memorization)

Decision Tree Regressor

$10$-Fold Cross-Validation

N/A

$29,768.51 ($\pm$ $7,841.74)

Severely Overfitted

Random Forest Regressor

$10$-Fold Cross-Validation

N/A

$24,206.62 ($\pm$ $5,433.91)

Highly Stable Champion

Random Forest (Final)

Unseen Test Set Exam

N/A

🔥 $21,678.79

Robust Generalization

Key Takeaway:

The Decision Tree appeared highly accurate during training ($\$7,013.26$ error) but completely collapsed under $10$-Fold Cross-Validation ($\$29,768.51$ Mean $RMSE$), proving it had memorized noise. Deploying an Ensemble Random Forest Regressor solved this, yielding a highly robust error of $\$21,678.79$ on completely untouched test data.

🛠️ Complete Implementation Code

Here is the exact implementation of our Scikit-Learn pipeline and regression models used inside the 05_ml_05_amaliyot.ipynb notebook:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# ==========================================
# 1. Data Splitting & Feature Grouping
# ==========================================
# df_clean is our cleaned dataset after outlier filtering
train_set, test_set = train_test_split(df_clean, test_size=0.2, random_state=21)

X_train = train_set.drop('price', axis=1)
y_label = train_set['price'].copy()

# Numeric columns
num_attribs = ['rooms', 'size', 'level', 'max_levels']

# String (categorical) columns
cat_attribs = ['district'] # we took district instead of location because they both represent almost same thing

print("Columns successfully grouped")

# ==========================================
# 2. Pipeline Preprocessing Architecture
# ==========================================
num_pipeline = Pipeline([
    ('std', StandardScaler())
])
print("Numeric Pipeline is ready")

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
])
print("Full Pipeline is ready")

# Fit & Transform training features
X_train_prepared = full_pipeline.fit_transform(X_train)

print(f"Not prepared data: {X_train.shape}")
print(f"Prepared data for ML: {X_train_prepared.shape}")

# ==========================================
# 3. Model Training & Diagnostics
# ==========================================

# Model A: Linear Regression Baseline
LG_model = LinearRegression()
LG_model.fit(X_train_prepared, y_label)

LG_predictions = LG_model.predict(X_train_prepared)
LG_mse = mean_squared_error(y_label, LG_predictions)
LG_rmse = np.sqrt(LG_mse)
print(f"Linear Regression model RMSE: ${LG_rmse:.2f}")

# Model B: Decision Tree (Overfitting Illusion)
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train_prepared, y_label)

tree_predictions = tree_model.predict(X_train_prepared)
tree_mse = mean_squared_error(y_label, tree_predictions)
tree_rmse = np.sqrt(tree_mse)
print(f"Decision Tree model RMSE: ${tree_rmse:.2f}")

# Rigorous 10-Fold Cross-Validation for Decision Tree
scores = cross_val_score(tree_model, X_train_prepared, y_label, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print("10 ta qatlamning alohida xatoliklari:")
print(tree_rmse_scores)
print(f"\nRMSE: ${tree_rmse_scores.mean():.2f}")
print(f"Standart Deviation: ${tree_rmse_scores.std():.2f}")

# Model C: Random Forest Ensemble (The Champion)
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train_prepared, y_label)

# Cross-Validation on Random Forest
forest_scores = cross_val_score(forest_model, X_train_prepared, y_label, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
print(f"Random Forest Mean RMSE: ${forest_rmse_scores.mean():.2f}")
print(f"Random Forest Standard Deviation: ${forest_rmse_scores.std():.2f}")

# ==========================================
# 4. Final Evaluation on Untouched Test Set
# ==========================================
X_test = test_set.drop("price", axis=1)
y_test = test_set['price'].copy()

# Crucial: Only transform the test set, NEVER fit!
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = forest_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(f"Final test set RMSE: ${final_rmse:.2f}")


🚀 Getting Started & Replication

Follow these steps to replicate the pipeline and run the diagnostics on your local machine:

1. Clone the repository

git clone https://github.com/dilmuratkenjabayev-commits/Tashkent.git
cd Tashkent


2. Create and activate a Virtual Environment

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


3. Install required packages

Create a requirements.txt file containing the following:

pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter


Then run:

pip install -r requirements.txt


4. Run the Jupyter Notebook

Start Jupyter Notebook to open and execute the training pipeline steps:

jupyter notebook


Open the 05_ml_05_amaliyot.ipynb notebook and run all cells to reproduce our final champion model.

🎓 Author & Credits

Developed by a passionate software engineer bridging the gap between clean code and reliable machine learning.

Developer: Dilmurat Kenjabayev

Education:

Software Engineering Student at IT Park University (Tashkent, Uzbekistan)

Data Science Student at School 21 (Tashkent, Uzbekistan)

LinkedIn: [Your LinkedIn URL]

Medium Article: https://medium.com/@dilmuratkenjabayev/dont-trust-your-first-model-overcoming-overfitting-in-tashkent-real-estate-regression-792528a53ce8
