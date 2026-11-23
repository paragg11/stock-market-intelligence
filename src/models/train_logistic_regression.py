

import numpy as np

from sklearn.impute import SimpleImputer

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("data/processed/modeling")
MODEL_DIR = Path("data/processed/models")
REPORT_DIR = Path("data/reports/modeling")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_dataset.csv"
TEST_FILE = DATA_DIR / "test_dataset.csv"

TARGET = "Target_5D"

EXCLUDED_FEATURES = [
    "Date",
    "Ticker",
    "Target_5D",
    "Forward_Return_5D",
    "Future_Close_5D"
]


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    return train_df, test_df


def get_features(train_df, test_df):
    features = [
        col for col in train_df.select_dtypes(include="number").columns
        if col not in EXCLUDED_FEATURES
    ]

    # Keep only features available in both datasets
    features = [
        col for col in features
        if col in test_df.columns
    ]

    # Remove columns that contain missing values
    features = [
        col for col in features
        if train_df[col].notna().all()
        and test_df[col].notna().all()
    ]

    return features


def train_model(X_train, y_train):

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    if y_test.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(
            y_test, probabilities
        )
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return predictions, probabilities, metrics, cm, report


def main():

    train_df, test_df = load_data()

    print("Train dataset:", train_df.shape)
    print("Test dataset:", test_df.shape)

    X_train = train_df.copy()
    X_test = test_df.copy()

    y_train = X_train[TARGET]
    y_test = X_test[TARGET]

    # Replace infinity values with missing values
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Remove features that are completely missing in training data
    valid_features = X_train.columns[
    X_train.notna().any()
    ].tolist()

    X_train = X_train[valid_features]
    X_test = X_test[valid_features]

    feature_columns = valid_features

    feature_columns = get_features(X_train, X_test)

    X_train = X_train[feature_columns]
    X_test = X_test[feature_columns]

    print("\nNumber of features:", len(feature_columns))
    print("Training target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True))

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    predictions, probabilities, metrics, cm, report = (
        evaluate_model(model, X_test, y_test)
    )

    print("\n========== LOGISTIC REGRESSION RESULTS ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # Save model
    model_path = MODEL_DIR / "logistic_regression_model.joblib"

    joblib.dump(model, model_path)

    # Save predictions
    prediction_df = test_df[
        ["Date", "Ticker", TARGET]
    ].copy()

    prediction_df["Predicted_Target"] = predictions
    prediction_df["Predicted_Probability_Up"] = probabilities

    prediction_file = REPORT_DIR / "logistic_regression_predictions.csv"

    prediction_df.to_csv(prediction_file, index=False)

    # Save metrics
    metrics["model"] = "Logistic Regression"
    metrics["features_count"] = len(feature_columns)
    metrics["train_rows"] = len(X_train)
    metrics["test_rows"] = len(X_test)

    metrics_file = REPORT_DIR / "logistic_regression_metrics.json"

    with open(metrics_file, "w") as file:
        json.dump(metrics, file, indent=4)

    # Save feature names
    feature_file = REPORT_DIR / "logistic_regression_features.json"

    with open(feature_file, "w") as file:
        json.dump(feature_columns, file, indent=4)

    print("\nSaved model:", model_path)
    print("Saved predictions:", prediction_file)
    print("Saved metrics:", metrics_file)
    print("Saved features:", feature_file)

    print("\nBASELINE MODEL TRAINING COMPLETED")


if __name__ == "__main__":
    main()
