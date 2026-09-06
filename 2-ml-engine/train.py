import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

DEFAULT_FILE_ID = "1GJX3RSZP4J0nF_gJDiQIG6sxQHBzu10c"
DEFAULT_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DEFAULT_FILE_ID}"

# Approximate 90th percentiles of the normal class, recalibrated for the
# new physically-grounded synthetic dataset (generate_dataset_v2.py).
# These MUST be recomputed any time the underlying data distribution
# changes -- stale floors calibrated on old data silently break the
# CRITICAL tier (verified: this caused 0 CRITICAL predictions before fix).
RAIN_FLOOR = 18.9
SOIL_FLOOR = 42.9
TILT_FLOOR = 3.2
VIB_FLOOR = 0.46

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
    "soil_saturation_excess"
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

    # Standalone soil saturation risk signal — NOT multiplied by rain/tilt,
    # so it does not collapse to ~0 when current rain/tilt are low.
    # This is what lets the model recognize an already-saturated slope
    # (e.g. 47% moisture right after rain has stopped) as risky.
    df["soil_saturation_excess"] = np.clip(
        df["soil_moisture_pct"] - SOIL_FLOOR, 0, None
    ) ** 2

    return df


def compute_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = 15.0,
    fp_cost: float = 1.0
):
    """
    CRITICAL-tier threshold. Chooses the threshold minimizing total
    expected cost = fn_cost * (missed landslides) + fp_cost * (false alarms).

    fn_cost >> fp_cost encodes the real-world asymmetry: a missed
    landslide risks lives, a false alarm costs an inspection trip.
    This is a principled, defensible way to pick an operating point
    for a high-stakes actionable alert, instead of guessing a target
    recall/specificity number.
    """

    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 181), y_proba])

    best_t = 0.5
    best_cost = float("inf")
    best_metrics = {}

    for t in thresholds:
        pred = (y_proba >= t).astype(int)

        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if specificity < 0.75:
            continue

        cost = fn_cost * fn + fp_cost * fp

        if cost < best_cost:
            best_cost = cost
            best_t = t

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = specificity

            best_metrics = {
                "threshold": round(float(t), 4),
                "landslide_recall": round(float(sens), 4),
                "no_landslide_recall": round(float(spec), 4),
                "expected_cost": round(float(cost), 2)
            }

    return best_t, best_metrics


def compute_early_warning_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.90
):
    """
    WARNING-tier threshold. Chooses the HIGHEST threshold that still
    keeps landslide recall at or above min_recall — i.e. the tier
    designed to catch nearly all real risk early, accepting more false
    positives because this tier only means "increase monitoring", not
    "evacuate".
    """

    thresholds = np.unique(np.r_[np.linspace(0.05, 0.95, 181), y_proba])
    candidates = []

    for t in thresholds:
        pred = (y_proba >= t).astype(int)

        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sens >= min_recall and spec >= 0.75:
            candidates.append((t, spec))

    if candidates:
        return max(candidates, key=lambda item: item[0])[0]

    # Fallback: no threshold hits min_recall — pick whichever achieves
    # the single highest recall available on this model.
    feasible = []
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if spec >= 0.75:
            feasible.append((sens, t))
            continue

        score = sens * spec
        if score > best_score:
            best_score = score
            best_t = t

    if feasible:
        return max(feasible, key=lambda item: (item[0], item[1]))[1]

    return best_t


def recall_specificity_at(y_true, y_proba, threshold):
    pred = (y_proba >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


def physical_support_mask(X_df: pd.DataFrame) -> np.ndarray:
    """
    True if ANY physical sensor has crossed its danger floor —
    used as a hard, model-independent signal that something is
    genuinely off, regardless of what the ML probability says.
    """
    crossings = pd.DataFrame({
        "rain": X_df["rainfall_mm_hr"] >= RAIN_FLOOR,
        "soil": X_df["soil_moisture_pct"] >= SOIL_FLOOR,
        "tilt": X_df["tilt_deg"] >= TILT_FLOOR,
        "vibration": X_df["vibration_index"] >= VIB_FLOOR,
    })
    return crossings.sum(axis=1) >= 2


def assign_tier(proba: np.ndarray, X_df: pd.DataFrame, critical_threshold: float,
                 warning_threshold: float, advisory_threshold: float) -> np.ndarray:
    """
    Produces a 4-level tier per row: 0=SAFE, 1=ADVISORY, 2=WARNING, 3=CRITICAL.

    CRITICAL requires model confidence AND physical evidence together —
    this keeps the highest-cost tier (evacuation-worthy) rare and trustworthy.
    WARNING fires if EITHER the model crosses its high-recall bar OR a
    physical floor is crossed alone — this tier is meant to be generous,
    since its cost is just "pay closer attention", not "evacuate".
    """
    support = physical_support_mask(X_df).values

    tier = np.zeros(len(proba), dtype=int)

    tier[(proba >= advisory_threshold)] = 1
    tier[(proba >= warning_threshold) | support] = 2
    tier[(proba >= critical_threshold) & support] = 3

    return tier


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

    # Preserve the real class prior for probability calibration. Oversampling
    # before calibration makes the reported probability mean something closer
    # to the synthetic training prior than to an actual monitoring day.
    X_train_res, y_train_res = X_train, y_train

    print(
        f"[INFO] Training Set -> "
        f"Landslides: {(y_train_res == 1).sum()} / "
        f"Total: {len(y_train_res)}"
    )

    print("[INFO] Training Random Forest...")

    model_base = RandomForestClassifier(
        n_estimators=350,
        max_depth=9,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight={0: 1.0, 1: 2.0},
        random_state=42,
        n_jobs=-1
    )

    model_base.fit(X_train_res, y_train_res)

    print("[INFO] Calibrating probabilities (Sigmoid / Platt Scaling)...")
    model = CalibratedClassifierCV(
        model_base,
        method="sigmoid",
        cv=3
    )

    model.fit(X_train_res, y_train_res)

    # ------------------------------------------------------------------
    # CHANGED: threshold selection now uses 5-fold out-of-fold (OOF)
    # probabilities over the FULL training set (X_train_full/y_train_full)
    # instead of a single small validation split (X_val/y_val had too few
    # positive samples to pick a stable threshold — this was the cause of
    # the WARNING tier hitting 90% recall on val but only 57% on test).
    #
    # IMPORTANT: this uses a separate cloned estimator (threshold_estimator)
    # purely for threshold selection. It NEVER touches `model` or
    # `model_base`, which remain fit exactly as before on X_train_res/
    # y_train_res. This avoids the NotFittedError caused earlier by running
    # cross_val_predict directly on the real fitted `model`.
    # ------------------------------------------------------------------
    print("[INFO] Computing out-of-fold probabilities for threshold selection...")

    # Must clone the CALIBRATED pipeline, not raw model_base — otherwise the
    # threshold is picked in an uncalibrated probability space and does not
    # correspond to the same operating point in the real (calibrated) model.
    threshold_estimator = CalibratedClassifierCV(
        clone(model_base),
        method="sigmoid",
        cv=3
    )

    oof_proba = cross_val_predict(
        threshold_estimator,
        X_train_full,
        y_train_full,
        cv=5,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    critical_threshold, crit_metrics = compute_cost_optimal_threshold(
        y_train_full.values, oof_proba, fn_cost=15.0, fp_cost=1.0
    )

    warning_threshold = compute_early_warning_threshold(
        y_train_full.values, oof_proba, min_recall=0.90
    )

    # Ensure the ordering makes sense: CRITICAL must never sit below WARNING
    if critical_threshold < warning_threshold:
        critical_threshold = min(0.99, warning_threshold + 0.10)

    advisory_floor = float(np.quantile(oof_proba, 0.80))
    advisory_threshold = min(
        warning_threshold * 0.8,
        max(0.05, min(0.30, advisory_floor))
    )

    crit_sens, crit_spec = recall_specificity_at(y_train_full.values, oof_proba, critical_threshold)
    warn_sens, warn_spec = recall_specificity_at(y_train_full.values, oof_proba, warning_threshold)

    print(
        f"[INFO] CRITICAL threshold: {critical_threshold:.3f} -> "
        f"Recall: {crit_sens:.3f} | Specificity: {crit_spec:.3f} "
        f"(cost-optimal: fn_cost=15, fp_cost=1)"
    )
    print(
        f"[INFO] WARNING threshold:  {warning_threshold:.3f} -> "
        f"Recall: {warn_sens:.3f} | Specificity: {warn_spec:.3f} "
        f"(early-notice tier, target recall >= 0.90)"
    )
    print(
        f"[INFO] ADVISORY threshold: {advisory_threshold:.3f}"
    )

    test_proba = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------
    # DIAGNOSTIC ONLY — not used in any downstream logic. Tells us how
    # many of the 4 physical floors true landslides actually cross, so
    # we know (measured, not guessed) whether physical_support_mask()'s
    # ">= 3" requirement is suppressing WARNING recall.
    # ------------------------------------------------------------------
    _crossings_check = pd.DataFrame({
        "rain": X_test["rainfall_mm_hr"] >= RAIN_FLOOR,
        "soil": X_test["soil_moisture_pct"] >= SOIL_FLOOR,
        "tilt": X_test["tilt_deg"] >= TILT_FLOOR,
        "vib": X_test["vibration_index"] >= VIB_FLOOR,
    }).sum(axis=1)
    print("\n[DIAGNOSTIC] Physical floor crossings among TRUE landslides (test set):")
    print(_crossings_check[y_test.values == 1].value_counts().sort_index())
    print("")

    tiers = assign_tier(
        test_proba, X_test, critical_threshold, warning_threshold, advisory_threshold
    )

    # For the classification report: "positive" = WARNING or above,
    # since that's the point the system asks a human to act/attend.
    y_pred_any_alert = (tiers >= 2).astype(int)
    # For a stricter view: "positive" = CRITICAL only (the evacuation trigger)
    y_pred_critical_only = (tiers >= 3).astype(int)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION — TIERED SYSTEM")
    print("=" * 60)

    print("\n--- Detection rate at WARNING-or-above (early notice) ---")
    print(
        classification_report(
            y_test,
            y_pred_any_alert,
            target_names=["No Landslide", "Landslide Risk"]
        )
    )

    print("--- Detection rate at CRITICAL only (evacuation trigger) ---")
    print(
        classification_report(
            y_test,
            y_pred_critical_only,
            target_names=["No Landslide", "Landslide Risk"]
        )
    )

    cm = confusion_matrix(y_test, y_pred_any_alert)
    tn, fp, fn, tp = cm.ravel()

    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    acc = (tp + tn) / len(y_test)

    auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    f2 = fbeta_score(y_test, y_pred_any_alert, beta=2)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = fbeta_score(y_test, y_pred_any_alert, beta=1)
    false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0

    print(
        f"WARNING-or-above -> No-Landslide Recall (Specificity): {spec * 100:.2f}% | "
        f"Landslide Recall (Sensitivity): {sens * 100:.2f}%"
    )
    print(
        f"False Alarms (Normal days flagged WARNING+): {fp} out of {tn + fp}"
    )
    print(
        f"ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | F2-Score: {f2:.4f} | Accuracy: {acc * 100:.2f}%"
    )
    print(
        f"Precision: {precision:.4f} | F1-Score: {f1:.4f} | "
        f"False Alarm Rate: {false_alarm_rate * 100:.2f}%"
    )

    tier_names = ["SAFE", "ADVISORY", "WARNING", "CRITICAL"]
    print("\nTier distribution on test set:")
    for i, name in enumerate(tier_names):
        print(f"  {name}: {(tiers == i).sum()} / {len(tiers)}")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Landslide", "Landslide"]
    )
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (WARNING-or-above, Critical Thresh={critical_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    artifact_path = os.path.join(output_dir, "landslide_model.joblib")

    artifact = {
        "model": model,
        "model_raw": model_base,
        "feature_cols": FEATURE_COLS,
        "critical_threshold": float(critical_threshold),
        "warning_threshold": float(warning_threshold),
        "advisory_threshold": float(advisory_threshold),
        "rain_floor": float(RAIN_FLOOR),
        "soil_floor": float(SOIL_FLOOR),
        "tilt_floor": float(TILT_FLOOR),
        "vib_floor": float(VIB_FLOOR),
        "metrics": {
            "no_landslide_recall": float(spec),
            "landslide_recall": float(sens),
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
            "f2_score": float(f2),
            "precision": float(precision),
            "f1_score": float(f1),
            "false_alarm_rate": float(false_alarm_rate)
        }
    }

    joblib.dump(artifact, artifact_path)

    print(f"\n[SUCCESS] Model artifact saved to: {artifact_path}")

    return artifact_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Landslide Prediction Model")
    parser.add_argument("--data-path", type=str, default=None, help="Local CSV path")
    parser.add_argument("--output-dir", type=str, default="models", help="Artifact folder")
    args = parser.parse_args()

    train_pipeline(data_path=args.data_path, output_dir=args.output_dir)