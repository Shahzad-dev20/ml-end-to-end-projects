# ML End-to-End Projects

A collection of complete, production-style machine learning projects — covering the full pipeline from raw data ingestion to trained, deployment-ready models (`.pkl` / `.joblib`).

Each project follows a consistent workflow:
**Raw Data → EDA & Cleaning → Feature Engineering → Model Training & Evaluation → Production-Ready Model**

Datasets are not included in this repo (to keep it lightweight) — each project's README links to the original data source.

---

## 📂 Projects

| Project | Problem Type | Domain | Key Output |
|---|---|---|---|
| [EXIM Bank Export Credit Data](./EXIM%20Bank%20Export%20Credit%20Data) | Regression | Finance / Trade Credit | Production regression pipeline with cancellation model |
| [Hate Crime Classification](./Hate%20crime) | Multiclass Classification | Social / Public Safety | Trained classification model |
| [House Price Prediction](./House%20price%20prediiction) | Regression | Real Estate | Price prediction model |
| [Lung Cancer Prediction](./Lungs%20cancer) | Classification | Healthcare | Best-performing classification model |
| [U.S. Chronic Disease Indicators](./U.S._Chronic_Disease_Indicators) | Analysis / Prediction | Public Health | Chronic disease prevalence model |
| [US Employee Payroll](./US_Employee_payroll) | Regression | HR / Finance | Payroll prediction model |

Each folder contains its own `README.md` with problem statement, dataset link, approach, and results.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Core Libraries:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib, seaborn
- **Notebook Environment:** Jupyter
- **Model Serialization:** pickle / joblib
- **Version Control:** Git & GitHub

---

## 📁 Repo Structure

Each project generally follows this internal structure:

```
Project Name/
├── Data Analysis/           # EDA notebooks, analysis reports
├── Data Cleaning/            # Cleaning scripts/notebooks
├── Production ML Pipeline/   # Final training pipeline, model.pkl, model card
├── Model/                    # Model comparison, evaluation artifacts
└── README.md                 # Project-specific details + dataset link
```

## 🎯 About This Repo

This repository is part of my ongoing practice as an **aspiring ML Engineer**, focused on building complete, real-world ML pipelines rather than isolated notebooks — including data cleaning, feature engineering, model evaluation, and production-readiness (model cards, config files, serialized models).

Currently a BSAI student and participant in the **Algoverse AI Research Program (2026 cohort)**.

---

## 📬 Contact

Feel free to reach out or connect if you'd like to discuss any of these projects.
Portfolio: (https://shazeportfolio.netlify.app/)
