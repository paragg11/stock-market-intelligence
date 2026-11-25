
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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

EXCLUDED_COLUMNS = [
    "Date",
    "Ticker",
    "Target_5D",
    "Forward_Return_5D",
    "Future_Close_5D"
]


def load_data():

    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Training or testing data not found."
        )

    return (
        pd.read_csv(TRAIN_FILE),
        pd.read_csv(TEST_FILE)
    )


def prepare_features(train_df, test_df):

    features = [
        col
        for col in train_df.select_dtypes(include="number").columns
        if col not in EXCLUDED_COLUMNS
        and col in test_df.columns
    ]

    X_train = train_df[features].replace(
        [np.inf, -np.inf], np.nan
    )

    X_test = test_df[features].replace(
        [np.inf, -np.inf], np.nan
    )

    # Remove columns entirely missing in training data
    valid_features = [
        col for col in features
        if X_train[col].notna().any()
    ]

    X_train = X_train[valid_features]
    X_test = X_test[valid_features]

    # Remove constant columns using training data only
    valid_features = [
        col for col in valid_features
        if X_train[col].nunique(dropna=True) > 1
    ]

    X_train = X_train[valid_features]
    X_test = X_test[valid_features]

    return X_train, X_test, valid_features


def build_model(y_train):

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())

    if positive_count == 0 or negative_count == 0:
        raise ValueError("Both target classes must exist in training data.")

    scale_pos_weight = negative_count / positive_count

    pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=2,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() > 1 else None
        )
    }

    matrix = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    return predictions, probabilities, metrics, matrix, report


def main():

    train_df, test_df = load_data()

    print("Training dataset:", train_df.shape)
    print("Testing dataset:", test_df.shape)

    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    X_train, X_test, features = prepare_features(
        train_df, test_df
    )

    if y_train.nunique() < 2:
        raise ValueError("Training data must contain both target classes.")

    if X_train.shape[1] == 0:
        raise ValueError("No usable features found.")

    print("\nFeatures selected:", len(features))

    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True))

    # Train XGBoost
    model = build_model(y_train)

    print("\nTraining XGBoost...")

    model.fit(X_train, y_train)

    # Evaluate
    predictions, probabilities, metrics, matrix, report = (
        evaluate_model(model, X_test, y_test)
    )

    print("\n========== XGBOOST RESULTS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    # Save model
    model_file = MODEL_DIR / "xgboost_model.joblib"
    joblib.dump(model, model_file)

    # Save predictions
    prediction_df = test_df[
        ["Date", "Ticker", TARGET]
    ].copy()

    prediction_df["Predicted_Target"] = predictions
    prediction_df["Predicted_Probability_Up"] = probabilities

    prediction_file = REPORT_DIR / "xgboost_predictions.csv"
    prediction_df.to_csv(prediction_file, index=False)

    # Save metrics
    metrics["model"] = "XGBoost"
    metrics["features_count"] = len(features)
    metrics["train_rows"] = len(X_train)
    metrics["test_rows"] = len(X_test)

    metrics_file = REPORT_DIR / "xgboost_metrics.json"

    with open(metrics_file, "w") as file:
        json.dump(metrics, file, indent=4)

    # Save feature names
    feature_file = REPORT_DIR / "xgboost_features.json"

    with open(feature_file, "w") as file:
        json.dump(features, file, indent=4)

    # Save feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.named_steps[
            "model"
        ].feature_importances_
    }).sort_values(
        "Importance", ascending=False
    )

    importance_file = REPORT_DIR / "xgboost_feature_importance.csv"

    importance.to_csv(importance_file, index=False)

    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))

    print("\nSaved model:", model_file)
    print("Saved predictions:", prediction_file)
    print("Saved metrics:", metrics_file)
    print("Saved feature importance:", importance_file)

    print("\nXGBOOST TRAINING COMPLETED")


if __name__ == "__main__":
    main()
