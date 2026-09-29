# 50 Startups - Profit Prediction

Author: Ahmed Shahzad

A complete end-to-end machine learning project for predicting startup profit using the classic 50 Startups dataset. This project demonstrates the full data science workflow: data inspection, exploratory data analysis (EDA), feature engineering, model comparison, model selection, and deployment-ready artifact generation.

## Project Overview

This project focuses on predicting the profit of startups using business investment features such as:

- R&D Spend
- Administration
- Marketing Spend
- State

The final model is trained and saved as a reusable scikit-learn pipeline, enabling efficient inference on new raw records without manual preprocessing.

## Business Problem

Startups often need to understand how their spending patterns influence profitability. This project builds a predictive model that estimates profit based on investment allocation and location, helping evaluate which features most strongly influence performance.

## Dataset

The dataset contains 50 startup records with 5 columns:

- R&D Spend
- Administration
- Marketing Spend
- State
- Profit

### Dataset Characteristics

- 50 rows
- 5 columns
- No missing values
- No duplicate rows
- No required data cleaning

## Key Insights from EDA

- R&D Spend has the strongest correlation with profit (~0.97)
- Marketing Spend also shows a meaningful relationship with profit (~0.75)
- Administration has a weak relationship with profit (~0.20)
- State does not show a strong standalone impact on profit

These findings suggest that investment in research and development plays the most important role in driving startup profitability.

## Modeling Approach

The project evaluates multiple regression models using 5-fold cross-validation:

- Random Forest
- Lasso
- Linear Regression
- Ridge
- Gradient Boosting
- SVR

### Selected Model

The final deployed model is:

- Random Forest Regressor

### Model Performance

Final selected model results:

- CV R² (mean): 0.946
- Test set R²: 0.899
- Test set MAE: 6,265
- Test set RMSE: 9,047

The Random Forest model outperformed the other candidates and was selected as the best-performing pipeline for deployment.

## Project Structure

```text
startup_profit_project/
├── data/
│   └── 50_Startups_dataset.csv
├── model/
│   ├── model_metadata.json
│   └── startup_profit_pipeline.joblib
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_eda.ipynb
│   └── 03_modeling_pipeline.ipynb
├── report/
│   ├── analysis_report.md
│   └── analysis_summary.txt
└── README.md
```

## Production Artifacts

The trained pipeline and metadata are stored in the `model/` directory:

- `model/startup_profit_pipeline.joblib`
- `model/model_metadata.json`

These artifacts contain the preprocessing pipeline and trained model, enabling direct prediction on new raw records.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook
- Joblib

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyter
```

4. Open the notebooks in the `notebooks/` folder to inspect the workflow.
5. Use the trained pipeline for prediction with new input records.

## Usage

The project includes a pipeline that accepts raw dataset-like input without requiring manual preprocessing. Predictions can be generated using the saved model artifact in `model/startup_profit_pipeline.joblib`.

## Author

Ahmed Shahzad

## Notes

This is a practical example of an end-to-end machine learning workflow, designed for learning, experimentation, and deployment-oriented modeling in a real-world problem setting.
