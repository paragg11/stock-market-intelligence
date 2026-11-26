
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

REPORT_DIR = Path("data/reports/modeling")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "XGBoost": "xgboost_metrics.json"
}

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]


def load_model_metrics():

    results = []

    for model_name, filename in MODEL_FILES.items():

        file_path = REPORT_DIR / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing metrics file: {file_path}\n"
                "Train all three models before comparison."
            )

        with open(file_path, "r") as file:
            metrics = json.load(file)

        if metrics.get("model") != model_name:
            raise ValueError(
                f"Unexpected model name in {filename}: "
                f"{metrics.get('model')}"
            )

        results.append({
            "Model": model_name,
            **{
                metric: metrics.get(metric)
                for metric in METRICS
            },
            "Features": metrics.get("features_count"),
            "Train Rows": metrics.get("train_rows"),
            "Test Rows": metrics.get("test_rows")
        })

    return pd.DataFrame(results)


def validate_comparison(df):

    # Ensure all models were evaluated on the same number of rows
    if df["Test Rows"].nunique() != 1:
        raise ValueError(
            "Models were not evaluated on the same number of test rows."
        )

    for metric in METRICS:
        if df[metric].isna().any():
            raise ValueError(
                f"Missing metric values detected: {metric}"
            )

    print("Validation passed: all models have matching test-row counts.")


def rank_models(df):

    # ROC-AUC is the primary ranking metric for this comparison.
    # Other metrics remain important for final selection.
    df = df.sort_values(
        by="roc_auc",
        ascending=False
    ).reset_index(drop=True)

    df.insert(
        0,
        "ROC_AUC_Rank",
        range(1, len(df) + 1)
    )

    return df


def create_comparison_chart(df):

    chart_file = REPORT_DIR / "model_comparison.png"

    plot_df = df.set_index("Model")[METRICS]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 7),
        width=0.75
    )

    ax.set_title("Stock Prediction Model Performance Comparison")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=15)
    plt.legend(title="Evaluation Metrics")
    plt.tight_layout()

    plt.savefig(chart_file, dpi=150)
    plt.close()

    print("Comparison chart saved:", chart_file)


def main():

    print("\n========== MODEL COMPARISON ==========")

    comparison_df = load_model_metrics()

    validate_comparison(comparison_df)

    comparison_df = rank_models(comparison_df)

    # Save detailed comparison
    output_file = REPORT_DIR / "model_comparison.csv"

    comparison_df.to_csv(
        output_file,
        index=False
    )

    print("\nMODEL PERFORMANCE TABLE\n")

    print(
        comparison_df[
            ["ROC_AUC_Rank", "Model"] + METRICS
        ].to_string(index=False)
    )

    print("\nMODEL RANKING")

    for _, row in comparison_df.iterrows():
        print(
            f"{int(row['ROC_AUC_Rank'])}. "
            f"{row['Model']} | ROC-AUC: {row['roc_auc']:.4f}"
        )

    print("\nTOP MODEL BY ROC-AUC:")
    print(comparison_df.iloc[0]["Model"])

    create_comparison_chart(comparison_df)

    print("\nComparison CSV saved:", output_file)

    print("\nMODEL COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
