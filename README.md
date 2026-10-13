# ML End-to-End Projects

This repository is a portfolio-style collection of end-to-end machine learning projects covering forecasting, classification, and analysis workflows across different domains.

The goal is to demonstrate practical ML work from raw-data handling through preprocessing, modeling, evaluation, and documentation. Most projects are designed to be self-contained and use public or externally hosted datasets that are not committed to GitHub to keep the repo lightweight.

## Repository status

- The tracked projects are the current portfolio items in this repository.
- The fish weight/body fat project exists in the workspace but is still untracked in Git and should be reviewed before the first public push.
- Many projects rely on externally hosted datasets and saved model artifacts that are intentionally kept in their project folders.

## Project index

| Project | Domain | Task | Notes |
|---|---|---|---|
| [EXIM Bank Export Credit Data](./EXIM%20Bank%20Export%20Credit%20Data) | Finance | Binary classification / risk ranking | Includes a deployment-style pipeline with model config and metrics |
| [Hate crime](./Hate%20crime) | Social analysis | Classification / pattern analysis | Includes notebooks, report text, and saved model |
| [House price prediiction](./House%20price%20prediiction) | Real estate | Regression | Includes a Streamlit app, feature encoding, and model artifact |
| [Lungs cancer](./Lungs%20cancer) | Healthcare | Classification | Includes a saved model and analysis notebook |
| [U.S._Chronic_Disease_Indicators](./U.S._Chronic_Disease_Indicators) | Public health | EDA / analysis | Notebook-based project with public health indicators |
| [US_Employee_payroll](./US_Employee_payroll) | HR / workforce | Regression / predictive analytics | Includes a trained payroll prediction model |
| [fish-weight-body-fat-predictor](./fish-weight-body-fat-predictor) | Nutrition / sports analytics | Regression | Flask + Gradio web apps with two saved models |

## Technologies used

- Python 3.10+
- pandas, numpy
- scikit-learn, XGBoost
- matplotlib, seaborn, plotly
- Jupyter notebooks
- Flask, Gradio, Streamlit
- joblib and pickle for model serialization

## Typical workflow

Each project follows a practical ML lifecycle:

1. Data loading and validation
2. Exploratory analysis and preprocessing
3. Feature engineering and model training
4. Evaluation with appropriate metrics
5. Model serialization and artifact output
6. App or notebook-based inference workflow

## Installation

Use a virtual environment and install only the dependencies needed for a specific project:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Then install dependencies from a project folder, for example:

```bash
cd "EXIM Bank Export Credit Data"
pip install -r requirements.txt
```

## Repository structure

```text
ml-end-to-end-projects/
├── README.md
├── .gitignore
├── AUDIT_CHECKLIST.md
├── EXIM Bank Export Credit Data/
├── Hate crime/
├── House price prediiction/
├── Lungs cancer/
├── U.S._Chronic_Disease_Indicators/
├── US_Employee_payroll/
├── fish-weight-body-fat-predictor/
└── .git/
```

## Dataset and artifact notes

- Most projects do not include raw datasets in Git because they are large, externally hosted, or not intended for redistribution.
- Trained model artifacts are stored in project folders where they were created and are treated as project outputs.
- Notebook and script paths expect to be run from their project directory unless the code explicitly handles relative paths.

## Verification

The repository was reviewed for:

- Python syntax correctness
- Git cleanliness and remote setup
- README accuracy and drift
- Hardcoded local paths and machine-specific references
- Dependency and execution metadata consistency

## Contribution and usage

This repository is mainly a personal ML portfolio and learning project collection. If you want to reuse the code, please review the project's own README, install its dependencies, and adjust dataset paths as needed.

## Author

Ahmed Shahzad

## Contact

Portfolio: https://shazeportfolio.netlify.app/
