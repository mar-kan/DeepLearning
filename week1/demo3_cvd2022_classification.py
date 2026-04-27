#!/usr/bin/env python3
"""
Demo 3 — Big tabular classification on CVD2022
- Trains Logistic Regression, Random Forest, Deep Neural Net (MLP)
- Handles large, mixed-type data with robust preprocessing
- Tracks train & predict (inference) times
- Metrics: Accuracy, F1, ROC-AUC, PR-AUC + ROC/PR curves + confusion matrices

Usage:
  python demo3_cvd2022_classification.py --csv heart_2022_no_nans.csv --target HadHeartAttack --sample_rows 50000
"""

import argparse
import time
import warnings
import inspect
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ---------- Helpers ----------

def make_ohe():
    """Create a OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def coerce_target(y_raw):
    """Convert target to binary 0/1 (if string labels exist)."""
    y = pd.Series(y_raw).copy()
    if y.dtype == object or y.dtype == "string":
        y = y.astype(str).str.strip().str.lower()
        true_set = {"1", "true", "yes", "y", "t"}
        false_set = {"0", "false", "no", "n", "f"}
        y = y.map(lambda v: 1 if v in true_set else (0 if v in false_set else np.nan))
    return y.astype(float)

def cap_high_cardinality(df, cat_cols, max_categories=100):
    """Cap categorical levels to top-K most frequent (rest = '__other__')."""
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            vc = out[c].value_counts(dropna=False)
            keep = set(vc.index[:max_categories])
            out[c] = out[c].map(lambda v: v if v in keep else "__other__")
    return out

def stratified_subsample(X, y, n_samples):
    """Stratified subsample to n_samples while preserving class balance."""
    if n_samples is None or n_samples >= len(y):
        return X, y
    test_size = 1.0 - n_samples / len(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    idx = np.arange(len(y))
    for _, keep_idx in sss.split(idx.reshape(-1, 1), y):
        return X.iloc[keep_idx], y.iloc[keep_idx]
    return X, y

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CVD2022 CSV")
    ap.add_argument("--target", default="CovidPos", help="Target column (binary)")
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--sample_rows", type=int, default=None, help="Optional stratified subsample size")
    ap.add_argument("--max_categories", type=int, default=100, help="Cap for category cardinality before one-hot")
    args = ap.parse_args()

    # 1) Load
    print(f"Loading: {args.csv}")
    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Available columns: {list(df.columns)[:10]} ...")

    # 2) Target prep
    y = coerce_target(df[args.target])
    if y.isna().any():
        try:
            y = pd.to_numeric(df[args.target], errors="coerce").astype(float)
        except Exception:
            pass
    mask = ~pd.isna(y)
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)

    # 3) Features
    X = df.drop(columns=[args.target])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce", downcast="float")
    X[cat_cols] = cap_high_cardinality(X[cat_cols], cat_cols, max_categories=args.max_categories)

    # Subsample if requested
    if args.sample_rows is not None:
        if args.sample_rows < len(y):
            X, y = stratified_subsample(X, y, args.sample_rows)
            print(f"Subsampled to {len(y):,} rows (stratified).")

    # 4) Split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    val_rel = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, random_state=42, stratify=y_trainval
    )

    print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")

    # 5) Preprocessing
    numeric_tf = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num_cols),
            ("cat", categorical_tf, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    class_weights_lr_rf = "balanced"
    sw_train = compute_sample_weight(class_weight="balanced", y=y_train)

    # 6) Models
    models = {
        "Logistic Regression": Pipeline(steps=[
            ("prep", preprocess),
            ("model", LogisticRegression(
                max_iter=1000, class_weight=class_weights_lr_rf,
                n_jobs=-1 if "n_jobs" in LogisticRegression().get_params() else None,
                solver="lbfgs"
            )),
        ]),
        "Random Forest": Pipeline(steps=[
            ("prep", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=None, random_state=42,
                n_jobs=-1, class_weight=class_weights_lr_rf
            )),
        ]),
        "Deep NN (MLP)": Pipeline(steps=[
            ("prep", preprocess),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", solver="adam", alpha=1e-4,
                batch_size=256, learning_rate_init=1e-3,
                max_iter=25, early_stopping=True,
                random_state=42, verbose=False
            )),
        ]),
    }

    # 7) Train, evaluate
    rows, curves, cms, reports = [], {}, {}, {}

    for name, pipe in models.items():
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()

        if name.startswith("Deep NN"):
            # Auto-detect support for sample_weight
            fit_sig = inspect.signature(MLPClassifier.fit)
            if "sample_weight" in fit_sig.parameters:
                pipe.fit(X_train, y_train, model__sample_weight=sw_train)
            else:
                print("⚠️ sample_weight not supported in this sklearn version — training MLP without it.")
                pipe.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train)

        fit_time = time.perf_counter() - t0

        # Predict
        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        pred_time = time.perf_counter() - t1

        try:
            y_score = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = pipe.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        prec, rec, _ = precision_recall_curve(y_test, y_score)

        cm = confusion_matrix(y_test, y_pred)
        cms[name] = cm
        curves[name] = dict(fpr=fpr, tpr=tpr, roc_auc=roc_auc, prec=prec, rec=rec, ap=ap)
        reports[name] = classification_report(y_test, y_pred, output_dict=True)

        rows.append({
            "model": name, "Accuracy": acc, "F1": f1, "ROC_AUC": roc_auc,
            "PR_AUC(AP)": ap, "fit_time_sec": fit_time, "predict_time_sec": pred_time,
        })

        val_pred = pipe.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        print(f"Train time: {fit_time:.2f}s | Predict time: {pred_time:.2f}s | "
              f"Test Acc: {acc:.4f} | Test F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {ap:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    results = pd.DataFrame(rows).sort_values(by="ROC_AUC", ascending=False).reset_index(drop=True)
    print("\n=== Test Performance Summary ===")
    print(results.to_string(index=False))

    # --- Plots (Accuracy, F1, ROC-AUC, PR-AUC, Times, ROC/PR curves, Confusion Matrices) ---
    plt.figure(figsize=(8,5)); plt.bar(results["model"], results["Accuracy"]); plt.title("Accuracy (Test)")
    plt.ylabel("Accuracy"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5)); plt.bar(results["model"], results["F1"]); plt.title("F1-score (Test)")
    plt.ylabel("F1"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5)); plt.bar(results["model"], results["ROC_AUC"]); plt.title("ROC-AUC (Test)")
    plt.ylabel("AUC"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5)); plt.bar(results["model"], results["PR_AUC(AP)"]); plt.title("PR-AUC (Test)")
    plt.ylabel("AP"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5)); plt.bar(results["model"], results["fit_time_sec"]); plt.title("Training Time (s)")
    plt.ylabel("Seconds"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,5)); plt.bar(results["model"], results["predict_time_sec"]); plt.title("Prediction Time (s)")
    plt.ylabel("Seconds"); plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7,6))
    for name, d in curves.items():
        plt.plot(d["fpr"], d["tpr"], label=f"{name} (AUC={d['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend(); plt.show()

    plt.figure(figsize=(7,6))
    for name, d in curves.items():
        plt.plot(d["rec"], d["prec"], label=f"{name} (AP={d['ap']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curves"); plt.legend(); plt.show()

    for name, cm in cms.items():
        plt.figure(figsize=(5,4))
        plt.imshow(cm, interpolation="nearest"); plt.title(f"Confusion Matrix — {name}"); plt.colorbar()
        ticks = np.arange(2); classes = ["Negative (0)", "Positive (1)"]
        plt.xticks(ticks, classes, rotation=20, ha="right"); plt.yticks(ticks, classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]), ha="center", va="center")
        plt.ylabel("True label"); plt.xlabel("Predicted label"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
