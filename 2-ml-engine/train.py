import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    fbeta_score
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

DEFAULT_FILE_ID = "1da4T1dtMsg8xwVwq76U0WbMV_n5U1NMJ"
DEFAULT_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DEFAULT_FILE_ID}"

RAIN_FLOOR = 8.0
SOIL_FLOOR = 40.0

FEATURE_COLS = [
    "soil_moisture_pct",
    "rainfall_mm_hr",
    "vibration_index",
    "tilt_deg",
    "temperature_C",
    "latitude",
    "longitude",
    "rain_soil_interaction",
    "tilt_vib_interaction",
    "rainfall_sq",
    "soil_moisture_sq",
    "combined_risk_index",
    "pore_pressure_proxy",
    "shear_force_proxy",
]


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """Direct remote download without Google Colab mounts."""
    if data_path and os.path.exists(data_path):
        print(f"[INFO] Loading local dataset from: {data_path}")
        return pd.read_csv(data_path)

    if os.path.exists("landslide_dataset_full.csv"):
        print("[INFO] Loading local 'landslide_dataset_full.csv'")
        return pd.read_csv("landslide_dataset_full.csv")

    print(f"[INFO] Downloading remote dataset from: {DEFAULT_DOWNLOAD_URL}")

    try:
        df = pd.read_csv(DEFAULT_DOWNLOAD_URL)
        print(f"[SUCCESS] Dataset loaded: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates physical and geotechnical interaction features."""
    df = df.copy()

    df["rain_soil_interaction"] = (
        df["rainfall_mm_hr"] *
        df["soil_moisture_pct"] / 100.0
    )

    df["tilt_vib_interaction"] = (
        df["tilt_deg"] *
        df["vibration_index"]
    )

    df["rainfall_sq"] = df["rainfall_mm_hr"] ** 2
    df["soil_moisture_sq"] = df["soil_moisture_pct"] ** 2

    df["combined_risk_index"] = (
        df["rain_soil_interaction"] +
        df["tilt_vib_interaction"]
    )

    df["pore_pressure_proxy"] = (
        df["soil_moisture_pct"] / 100.0
    ) * np.sin(np.radians(df["tilt_deg"]))

    df["shear_force_proxy"] = (
        np.sin(np.radians(df["tilt_deg"])) *
        (1.0 + df["vibration_index"])
    )

    return df


def optimize_threshold_balanced(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_specificity: float = 0.80
):
    """
    Finds the optimal threshold balancing Sensitivity (Landslide Recall)
    and Specificity (No-Landslide Recall) using Youden's J statistic
    subject to keeping No-Landslide Recall >= min_specificity.
    """

    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_j = -1.0
    best_metrics = {}

    for t in thresholds:
        pred = (y_proba >= t).astype(int)

        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        j = sens + spec - 1.0

        if spec >= min_specificity and j > best_j:
            best_j = j
            best_threshold = t

            best_metrics = {
                "threshold": round(float(t), 4),
                "landslide_recall": round(float(sens), 4),
                "no_landslide_recall": round(float(spec), 4),
                "youden_j": round(float(j), 4)
            }

    if best_j == -1.0:
        for t in thresholds:
            pred = (y_proba >= t).astype(int)

            tp = ((pred == 1) & (y_true == 1)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            tn = ((pred == 0) & (y_true == 0)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)

            j = sens + spec - 1.0

            if j > best_j:
                best_j = j
                best_threshold = t

                best_metrics = {
                    "threshold": round(float(t), 4),
                    "landslide_recall": round(float(sens), 4),
                    "no_landslide_recall": round(float(spec), 4),
                    "youden_j": round(float(j), 4)
                }

    return best_threshold, best_metrics


def apply_guard_rule(
    X_df: pd.DataFrame,
    proba: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Filters out spurious sensor blips that lack physical rainfall/moisture backing.
    """

    model_flag = proba >= threshold

    physical_support = (
        (X_df["rainfall_mm_hr"] >= RAIN_FLOOR) |
        (X_df["soil_moisture_pct"] >= SOIL_FLOOR)
    )

    return (model_flag & physical_support).astype(int)


def train_pipeline(
    data_path: str = None,
    output_dir: str = "models",
    plots_dir: str = "outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    df = load_dataset(data_path)
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df["landslide"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    if HAS_SMOTE:
        smote = SMOTE(
            random_state=42,
            sampling_strategy=0.45
        )

        X_train_res, y_train_res = smote.fit_resample(
            X_train,
            y_train
        )

    else:
        minority = X_train[y_train == 1]
        majority_count = (y_train == 0).sum()

        n_needed = (
            int(majority_count * 0.45) -
            len(minority)
        )

        if n_needed > 0:
            reps = int(
                np.ceil(n_needed / len(minority))
            )

            synthetic = pd.concat(
                [minority] * reps,
                ignore_index=True
            ).iloc[:n_needed].copy()

            noise = np.random.default_rng(42).normal(
                0,
                0.02,
                synthetic.shape
            )

            synthetic = synthetic * (1 + noise)

            X_train_res = pd.concat(
                [X_train, synthetic],
                ignore_index=True
            )

            y_train_res = pd.concat(
                [
                    y_train,
                    pd.Series([1] * n_needed)
                ],
                ignore_index=True
            )

        else:
            X_train_res, y_train_res = X_train, y_train

    print(
        f"[INFO] Resampled Training Set -> "
        f"Landslides: {(y_train_res == 1).sum()} / "
        f"Total: {len(y_train_res)}"
    )

    print("[INFO] Training Random Forest...")

    rf_base = RandomForestClassifier(
        n_estimators=350,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 3.0},
        random_state=42,
        n_jobs=-1
    )

    rf_base.fit(
        X_train_res,
        y_train_res
    )

    print("[INFO] Calibrating probabilities (Isotonic Regression)...")

    model = CalibratedClassifierCV(
        rf_base,
        method="isotonic",
        cv=3
    )

    model.fit(
        X_train_res,
        y_train_res
    )

    val_proba = model.predict_proba(X_val)[:, 1]

    best_threshold, val_metrics = optimize_threshold_balanced(
        y_val.values,
        val_proba,
        min_specificity=0.80
    )

    print(
        f"[INFO] Optimal Threshold: {best_threshold:.3f}"
    )

    print(
        f"[INFO] Validation Metrics -> "
        f"Landslide Recall: "
        f"{val_metrics['landslide_recall']:.3f} | "
        f"No-Landslide Recall: "
        f"{val_metrics['no_landslide_recall']:.3f}"
    )

    test_proba = model.predict_proba(X_test)[:, 1]

    y_pred = apply_guard_rule(
        X_test,
        test_proba,
        best_threshold
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    acc = (tp + tn) / len(y_test)

    auc = roc_auc_score(
        y_test,
        test_proba
    )

    pr_auc = average_precision_score(
        y_test,
        test_proba
    )

    f2 = fbeta_score(
        y_test,
        y_pred,
        beta=2
    )

    print("\n" + "=" * 55)
    print("FINAL TEST EVALUATION (SAFETY OPTIMIZED)")
    print("=" * 55)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Landslide",
                "Landslide Risk"
            ]
        )
    )

    print(
        f"No-Landslide Recall (Specificity): "
        f"{spec * 100:.2f}%"
    )

    print(
        f"Landslide Recall (Sensitivity):    "
        f"{sens * 100:.2f}%"
    )

    print(
        f"False Alarms (Normal days halted): "
        f"{fp} out of {tn + fp}"
    )

    print(
        f"ROC-AUC: {auc:.4f} | "
        f"PR-AUC: {pr_auc:.4f} | "
        f"F2-Score: {f2:.4f} | "
        f"Accuracy: {acc * 100:.2f}%"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Landslide",
            "Landslide"
        ]
    )

    disp.plot(cmap="Blues")

    plt.title(
        f"Optimized Confusion Matrix "
        f"(Threshold={best_threshold:.2f} + Guard)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plots_dir,
            "confusion_matrix.png"
        )
    )

    plt.close()

    artifact_path = os.path.join(
        output_dir,
        "landslide_model.joblib"
    )

    artifact = {
        "model": model,
        "model_raw": rf_base,
        "feature_cols": FEATURE_COLS,
        "best_threshold": float(best_threshold),
        "advisory_threshold": float(
            max(0.10, best_threshold * 0.65)
        ),
        "rain_floor": float(RAIN_FLOOR),
        "soil_floor": float(SOIL_FLOOR),
        "metrics": {
            "no_landslide_recall": float(spec),
            "landslide_recall": float(sens),
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "f2_score": float(f2)
        }
    }

    joblib.dump(
        artifact,
        artifact_path
    )

    print(
        f"[SUCCESS] Model artifact saved to: "
        f"{artifact_path}"
    )

    return artifact_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Landslide Prediction Model"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Local CSV path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Artifact folder"
    )

    args = parser.parse_args()

    train_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir
    )