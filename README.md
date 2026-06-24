# 🏠 Tashkent Housing Price Prediction

An end-to-end Machine Learning regression project for predicting apartment prices in Tashkent, Uzbekistan. This project demonstrates how proper data cleaning, preprocessing pipelines, cross-validation, and ensemble learning can be used to build a reliable housing price prediction system.

---

## 📌 Project Overview

The Tashkent real estate market contains numerous data quality issues, including outliers, missing values, inconsistent formats, and non-numeric entries stored as text. The goal of this project was to build a robust machine learning pipeline capable of handling noisy real-world data while maintaining strong generalization performance.

### Key Features

* ✔️ Data cleaning and type conversion
* ✔️ Outlier detection and filtering
* ✔️ Modular preprocessing using Scikit-Learn Pipelines
* ✔️ Prevention of data leakage through proper train/test separation
* ✔️ Model comparison and overfitting analysis
* ✔️ 10-Fold Cross-Validation evaluation
* ✔️ Random Forest ensemble model deployment

---

## 📂 Project Structure

```text
TASHKENT_HOUSING_PREDICTION/
│
├── data/
│   └── housing_data_08-02-2021.csv
│
├── 05_ml_05_amaliyot.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 🔍 Data Preparation

The original dataset contained several inconsistencies:

* Numeric values stored as strings
* Missing values
* Unrealistic apartment sizes
* Invalid floor information
* Extreme pricing outliers

### Data Cleaning

Numeric columns were converted using:

```python
pd.to_numeric(..., errors="coerce")
```

Rows containing invalid values were removed after conversion.

### Outlier Filtering

To improve model stability, records were filtered using realistic market constraints:

| Feature | Range                          |
| ------- | ------------------------------ |
| Size    | 15 – 500 m²                    |
| Price   | $5,000 – $1,000,000            |
| Floors  | Valid floor relationships only |

---

## ⚙️ Machine Learning Pipeline

### 1. Train/Test Split

The dataset was split before any preprocessing:

```python
train_test_split(
    df_clean,
    test_size=0.2,
    random_state=21
)
```

This ensured that the test set remained completely unseen during training.

### 2. Feature Selection

Selected Features:

#### Numerical Features

```python
['rooms', 'size', 'level', 'max_levels']
```

#### Categorical Features

```python
['district']
```

The `district` feature was selected instead of `location` because both represented nearly identical geographical information.

---

### 3. Preprocessing

#### Numerical Pipeline

```python
StandardScaler()
```

Used to normalize numerical features.

#### Categorical Pipeline

```python
OneHotEncoder(handle_unknown='ignore')
```

Used to safely encode districts while handling unseen categories during deployment.

#### Unified Pipeline

```python
ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_attribs)
])
```

This architecture prevents data leakage and guarantees consistent preprocessing.

---

## 📊 Model Evaluation

Performance was measured using **Root Mean Squared Error (RMSE)**.

Lower RMSE indicates better predictive performance.

### Model Comparison

| Model             | Evaluation Method        | RMSE (USD)             | Verdict            |
| ----------------- | ------------------------ | ---------------------- | ------------------ |
| Linear Regression | Training Set             | $26,582.97             | Underfitting       |
| Decision Tree     | Training Set             | $7,013.26              | Misleadingly Good  |
| Decision Tree     | 10-Fold Cross Validation | $29,768.51 ± $7,841.74 | Severe Overfitting |
| Random Forest     | 10-Fold Cross Validation | $24,206.62 ± $5,433.91 | Stable             |
| Random Forest     | Unseen Test Set          | **$21,678.79**         | Best Model         |

---

## 🚨 Overfitting Analysis

At first glance, the Decision Tree model appeared excellent:

```text
Training RMSE = $7,013.26
```

However, after applying 10-Fold Cross-Validation:

```text
Validation RMSE = $29,768.51
```

The dramatic increase revealed that the model had memorized training data rather than learning meaningful patterns.

### Solution

A Random Forest ensemble model was introduced.

Benefits:

* Reduces variance
* Improves generalization
* More resistant to noise and outliers
* Produces more stable predictions

Final performance on completely unseen data:

```text
RMSE = $21,678.79
```

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/dilmuratkenjabayev-commits/Tashkent.git

cd Tashkent
```

### Create Virtual Environment

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
05_ml_05_amaliyot.ipynb
```

Run all cells to reproduce the complete machine learning workflow and model evaluation process.

---

## 🎓 Author

### Dilmurat Kenjabayev

Software Engineering Student at IT Park University (Tashkent)

Data Science Student at School 21 (Tashkent)

---

## 📄 License

This project is licensed under the MIT License.

Feel free to use, modify, and distribute this project for educational and research purposes.
