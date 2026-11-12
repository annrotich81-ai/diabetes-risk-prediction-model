# Diabetes Risk Prediction Using Supervised Machine Learning

**Project developer:** Ann Jepchumba Rotich  
**Student ID:** SJEPRo2412

## Project overview

This project explores binary classification of diabetes status using the Pima Indians Diabetes dataset. It compares three supervised-learning algorithms: Logistic Regression, Decision Tree and Random Forest.

The Python workflow checks and cleans the data, treats zero values in selected clinical measurement columns as missing, applies median imputation inside model pipelines, creates a stratified 80:20 train/test split, and performs five-fold stratified cross-validation on the training data. It reports accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices.

## Dataset

The script reads the public Pima Indians Diabetes CSV from:

`https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`

The commonly distributed dataset contains 768 records, eight predictor variables and one binary outcome. It represents a specific population and must not be assumed to generalise to the Kenyan population or other groups without external validation.

## Setup and execution

Python 3.10 or later is recommended. Open a terminal in this folder and run:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python diabetes_model.py
```

An internet connection is required to download the dataset on the first run.

## Output files

The script creates an `outputs/` folder containing:

- `metrics.csv` - model metrics on the held-out test set and cross-validation F1 summary.
- `confusion_matrices.txt` - confusion matrices and classification reports.
- `diabetes_risk_model.joblib` - the selected fitted pipeline.

Only use values produced by `outputs/metrics.csv` as the performance scores for this implementation. Interpret false negatives and false positives from the confusion matrices.

## Results note

The PDF report includes a separately identified 2025 published benchmark comparing the same three algorithms on the Pima dataset. Those published values are background comparison figures; they are **not** presented as the output of this project's Python script. Run the script to generate the project's own evaluation scores before treating the model assessment as complete.

## Limitations and future improvement

1. Validate the model on an independent and representative dataset, preferably collected with appropriate consent in the intended setting.
2. Tune hyperparameters through training-only cross-validation while preserving the final test set.
3. Calibrate predicted probabilities and choose thresholds with qualified clinical input if the model is ever considered for screening research.
4. Assess subgroup performance and possible bias.
5. Obtain ethical and clinical review before any use involving patient care.

## Repository publication

See [GITHUB_UPLOAD_STEPS.md](GITHUB_UPLOAD_STEPS.md) for instructions to create a repository and publish these files. Once the repository exists, add its actual public URL to the PDF under **GitHub Repository Link**.

## Safety note

This is an academic machine-learning prototype, not a medical diagnostic tool. It must not be used to diagnose, treat or reassure patients.
