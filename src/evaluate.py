"""Evaluation: metrics, confusion matrices, ROC, McNemar tests (Sec. 3.8).

All models are evaluated per test ROW in a canonical order (rows sorted by
their original position in the dataset), so pairwise McNemar tests between
classical models and the LSTM are computed on identical units.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.contingency_tables import mcnemar

from . import config
from .models import make_windows

log = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "knn": "KNN",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}


def _canonical(split) -> tuple[np.ndarray, np.ndarray]:
    """Canonical test order: source-row ascending. Returns (order, y_sorted)."""
    order = np.argsort(split.test_index)
    return order, split.y_test[order]


def evaluate_classical(name: str, estimator, split) -> dict:
    """Per-row test metrics for a tuned classical model (canonical order)."""
    order, y = _canonical(split)
    proba = estimator.predict_proba(split.X_test[order])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return _metrics(DISPLAY_NAMES.get(name, name), y, pred, proba)


def evaluate_lstm(model, split) -> dict:
    """Per-row test metrics for the LSTM.

    One window ends at each test row (stride 1, windows drawn strictly from
    test rows), giving a kick probability for every test row in canonical
    order. Window steps overlap, which is noted where relevant.
    """
    order, y = _canonical(split)
    Xw, yw = make_windows(split.X_test[order], y, np.sort(split.test_index), stride=1)
    proba = model.predict(Xw, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    res = _metrics("LSTM", yw, pred, proba)
    res["_windowed"] = True
    return res


def _metrics(display: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "model": display,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "_y": y,
        "_pred": pred,
        "_proba": proba,
    }


def metrics_table(results: list[dict], cv_recalls: dict[str, float]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1": round(r["f1"], 4),
            "ROC-AUC": round(r["roc_auc"], 4),
            "CV Recall (train)": round(cv_recalls.get(r["model"], np.nan), 4),
        })
    return pd.DataFrame(rows).sort_values(["Recall", "F1"], ascending=False).reset_index(drop=True)


def mcnemar_matrix(results: list[dict]) -> pd.DataFrame:
    """Pairwise McNemar tests on per-row correctness (canonical order)."""
    names = [r["model"] for r in results]
    recs = []
    for i, a in enumerate(names):
        ra = results[i]
        for j in range(i + 1, len(names)):
            rb = results[j]
            ca = (ra["_pred"] == ra["_y"]).astype(int)
            cb = (rb["_pred"] == rb["_y"]).astype(int)
            table = [
                [int(((ca == 1) & (cb == 1)).sum()), int(((ca == 1) & (cb == 0)).sum())],
                [int(((ca == 0) & (cb == 1)).sum()), int(((ca == 0) & (cb == 0)).sum())],
            ]
            if table[0][1] + table[1][0] == 0:
                # identical predictions on every test row -> no evidence of difference
                res = type("R", (), {"statistic": 0.0, "pvalue": 1.0})()
            else:
                res = mcnemar(table, exact=False, correction=True)
            note = "LSTM unit = overlapping windows" if (ra.get("_windowed") or rb.get("_windowed")) else ""
            recs.append({
                "model_a": a, "model_b": rb["model"],
                "statistic": round(float(res.statistic), 4),
                "p_value": round(float(res.pvalue), 5),
                "significant (p<0.05)": bool(res.pvalue < 0.05),
                "note": note,
            })
    return pd.DataFrame(recs)


def plot_confusion(results: list[dict], out: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.2))
    for ax, r in zip(np.atleast_1d(axes), results):
        cm = confusion_matrix(r["_y"], r["_pred"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["no kick", "kick"], yticklabels=["no kick", "kick"])
        ax.set_title(r["model"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(results: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(r["_y"], r["_proba"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['model']} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves - test partition")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kick_series(df: pd.DataFrame, out: Path) -> None:
    """Kick-onset overview for the report's data-description section."""
    t = df["WellDepth"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    onset = t[df[config.TARGET_COLUMN] == 1].min()
    for ax, col, label in zip(
        axes,
        ["FPress", "ActiveGL", "ATVolume"],
        ["Formation pressure, psi",
         "Active gain/loss (kick indicator)",
         "Active tank volume (PVT), bbl"],
    ):
        ax.plot(t, df[col], lw=0.8)
        ax.axvline(onset, color="r", ls="--", lw=1, label=f"kick onset t={onset:.2f}")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Simulation time, s")
    fig.suptitle("DataDRILL Kick_Detection - kick signature")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_importance(name: str, estimator, feature_names: list[str], out: Path) -> None:
    """Feature importances for tree ensembles."""
    if not hasattr(estimator, "feature_importances_"):
        return
    imp = pd.Series(estimator.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    imp.plot(kind="barh", ax=ax)
    ax.set_title(f"Feature importance - {DISPLAY_NAMES.get(name, name)}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_probability_timeline(
    results: list[dict], times: np.ndarray, onset_time: float, out: Path
) -> None:
    """Kick probability vs simulation time on the test partition.

    Shows when each model first becomes confident relative to the true kick
    onset -- the operationally relevant detection behaviour that a single
    aggregate score hides.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results:
        ax.plot(times, r["_proba"], lw=1.4, label=r["model"])
    ax.axvline(onset_time, color="r", ls="--", lw=1.2, label=f"kick onset t={onset_time:.2f}")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Simulation time, s")
    ax.set_ylabel("P(kick)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Kick probability over time - test partition")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

