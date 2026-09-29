# 50 Startups — Profit Prediction: Analysis Report

## 1. Dataset

- **Source file:** `50_Startups_dataset.csv`
- **Rows / columns:** 50 rows, 5 columns
- **Columns:** `R&D Spend`, `Administration`, `Marketing Spend` (numeric), `State` (categorical: New York, California, Florida), `Profit` (target)
- **Quality:** No missing values, no duplicate rows, no negative values. Dataset required no cleaning.

## 2. Exploratory Data Analysis — Key Findings

| Feature | Correlation with Profit | Notes |
|---|---|---|
| R&D Spend | ≈ 0.97 | Dominant, near-linear predictor |
| Marketing Spend | ≈ 0.75 | Moderate predictor; correlated with R&D Spend (≈0.72) |
| Administration | ≈ 0.20 | Weak predictor |
| State | ~none | Mean profit similar across all 3 states |

**Takeaways:**
- Profit is driven almost entirely by R&D Spend.
- Marketing Spend adds secondary signal but overlaps with R&D Spend (mild multicollinearity risk for linear models).
- Administration and State contribute little on their own but were kept as features since they cost nothing to include and tree-based models handle irrelevant features gracefully.
- No meaningful outliers — low-spend startups have proportionally low profit, consistent with the overall trend.

## 3. Modeling

### Setup
- 80/20 train/test split (`random_state=42`)
- Preprocessing: `StandardScaler` on numeric features, `OneHotEncoder` on `State`, combined via `ColumnTransformer`
- Model comparison: 5-fold cross-validation on the training set, scored on R², MAE, RMSE

### Model comparison (cross-validated R², training data)

| Model | CV R² (mean) | CV MAE | CV RMSE |
|---|---|---|---|
| **Random Forest** | **0.946** | 7,158 | 9,002 |
| Lasso | 0.934 | 7,424 | 9,548 |
| Linear Regression | 0.934 | 7,426 | 9,551 |
| Ridge | 0.930 | 7,716 | 9,726 |
| Gradient Boosting | 0.901 | 9,186 | 11,718 |
| SVR | -0.275 | 35,205 | 42,603 |

### Selected model: **Random Forest Regressor**

- Best mean cross-validated R² and lowest error among all candidates.
- **Held-out test set performance:** R² = 0.899, MAE ≈ 6,265, RMSE ≈ 9,047 — consistent with cross-validation, no signs of overfitting.
- Linear models performed almost as well (R² ≈ 0.93), confirming the underlying relationship is largely linear — but Random Forest edged them out and handles the R&D/Marketing correlation and weak features (Administration, State) without needing explicit multicollinearity treatment.
- SVR performed very poorly without hyperparameter tuning and was rejected outright.

## 4. Production Pipeline

- Final Random Forest pipeline (preprocessing + model, single `sklearn.Pipeline` object) refit on **all 50 rows** for deployment.
- Saved artifacts:
  - `model/startup_profit_pipeline.joblib` — the deployable pipeline
  - `model/model_metadata.json` — chosen model, feature list, test metrics, full CV comparison table
- Inference interface: `predict_profit(pipeline, records)` accepts new records in the same raw format as the original CSV (no manual preprocessing needed — it's baked into the pipeline) and returns predicted profit.

## 5. Recommendations / Next Steps

- With only 50 rows, results should be treated as directional; more data would improve robustness of the model comparison.
- R&D Spend is the single most actionable lever for profit in this dataset — if this generalizes to real decision-making, it warrants further investigation (causal vs. correlational).
- For production, consider periodic retraining as new startup records become available, and log prediction vs. actual profit to monitor drift.
