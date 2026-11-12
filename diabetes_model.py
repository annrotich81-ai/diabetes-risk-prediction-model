"""Diabetes risk classification project.

Author: Ann Jepchumba Rotich
Student ID: SJEPRo2412
Educational project only; not a medical diagnostic tool.

Run:
    pip install -r requirements.txt
    python diabetes_model.py

The dataset is downloaded from the public CSV URL below. The script saves
metrics.csv, confusion_matrices.txt, and diabetes_risk_model.joblib.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
FEATURES = COLUMNS[:-1]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "diabetes_risk_model.joblib"


def load_data() -> pd.DataFrame:
    """Download and clean the public Pima Indians Diabetes CSV."""
    try:
        df = pd.read_csv(DATA_URL, header=None, names=COLUMNS)
    except Exception as exc:  # provide a useful explanation on network errors
        raise RuntimeError(
            "Could not download the dataset. Check your internet connection "
            "or download the CSV from the DATA_URL and place it in this folder."
        ) from exc

    if df.empty or list(df.columns) != COLUMNS:
        raise ValueError("The downloaded dataset is empty or has an unexpected format.")

    df = df.drop_duplicates().copy()
    for column in ZERO_AS_MISSING:
        df.loc[df[column] == 0, column] = np.nan
    return df


def build_models() -> dict[str, Pipeline]:
    """Create reproducible model pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=8,
                class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=400, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }


def main() -> None:
    df = load_data()
    X = df[FEATURES]
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    confusion_text = []
    fitted_models: dict[str, Pipeline] = {}

    print(f"Dataset shape after duplicate removal: {df.shape}")
    print("Target distribution:")
    print(y.value_counts().sort_index().rename(index={0: "Negative", 1: "Positive"}))
    print(f"Train records: {len(X_train)} | Test records: {len(X_test)}")

    for name, pipeline in build_models().items():
        cv_f1 = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=None
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

        row = {
            "Model": name,
            "CrossValidation_F1_Mean": float(cv_f1.mean()),
            "CrossValidation_F1_Std": float(cv_f1.std()),
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Precision": float(precision_score(y_test, predictions, zero_division=0)),
            "Recall": float(recall_score(y_test, predictions, zero_division=0)),
            "F1_Score": float(f1_score(y_test, predictions, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_test, probabilities)),
        }
        metric_rows.append(row)
        fitted_models[name] = pipeline
        confusion_text.append(
            f"{name}\nConfusion matrix [[TN, FP], [FN, TP]]:\n{matrix}\n"
            f"Classification report:\n{classification_report(y_test, predictions, zero_division=0)}"
        )

    results = pd.DataFrame(metric_rows).sort_values(
        by=["F1_Score", "Recall"], ascending=False
    )
    results.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "confusion_matrices.txt").write_text(
        "\n\n".join(confusion_text), encoding="utf-8"
    )

    best_name = str(results.iloc[0]["Model"])
    joblib.dump(fitted_models[best_name], MODEL_PATH)

    print("\nMODEL COMPARISON")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model by held-out F1-score: {best_name}")
    print(f"Saved metrics: {OUTPUT_DIR / 'metrics.csv'}")
    print(f"Saved model: {MODEL_PATH}")
    print("Educational prototype only - not a clinical diagnosis.")


if __name__ == "__main__":
    main()
