"""Single-command orchestrator: data -> preprocessing -> tuning -> evaluation.

Usage:
    python -m src.run            # full pipeline (train + evaluate + report)
    python -m src.run --skip-train   # rebuild report from saved artifacts
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_io import derive_label, load_dataset
from .evaluate import (
    evaluate_classical,
    evaluate_lstm,
    mcnemar_matrix,
    metrics_table,
    plot_confusion,
    plot_importance,
    plot_kick_series,
    plot_probability_timeline,
    plot_roc,
)
from .features import derive_features
from .preprocessing import build_split, handle_gaps
from .train import train_lstm, tune_classical

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("src.run")


def ensure_dirs() -> None:
    for d in (config.MODELS_DIR, config.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_pipeline(skip_train: bool = False) -> pd.DataFrame:
    """Execute the full pipeline and return the metrics table."""
    ensure_dirs()
    t0 = time.time()

    # 1. Data -----------------------------------------------------------------
    df = derive_label(load_dataset())
    df = derive_features(df)
    df = handle_gaps(df)
    plot_kick_series(df, config.RESULTS_DIR / "kick_timeseries.png")

    # 2. Preprocessing (Sec. 3.6) --------------------------------------------
    split = build_split(df)

    # 3. Training (Sec. 3.7) --------------------------------------------------
    cv_recalls: dict[str, float] = {}
    results: list[dict] = []
    if skip_train:
        from .evaluate import DISPLAY_NAMES  # noqa: F401  (rebuild path)

        cv_store = {}
        cv_path = config.RESULTS_DIR / "cv_recall.json"
        if cv_path.exists():
            cv_store = json.loads(cv_path.read_text())
        for name in ("decision_tree", "random_forest", "svm", "knn", "xgboost"):
            art = config.MODELS_DIR / f"best_{name}.pkl"
            with open(art, "rb") as fh:
                blob = pickle.load(fh)
            results.append(evaluate_classical(name, blob["estimator"], split))
            cv_recalls[results[-1]["model"]] = cv_store.get(name, float("nan"))
            log.info("Loaded %s from %s", name, art)
        import tensorflow as tf

        lstm = tf.keras.models.load_model(config.MODELS_DIR / "lstm.keras")
        results.append(evaluate_lstm(lstm, split))
        cv_recalls["LSTM"] = cv_store.get("lstm", float("nan"))
    else:
        classical = tune_classical(split, config.MODELS_DIR)
        for tr in classical:
            results.append(evaluate_classical(tr.name, tr.estimator, split))
            cv_recalls[results[-1]["model"]] = tr.cv_recall
            if tr.name in ("random_forest", "xgboost"):
                plot_importance(tr.name, tr.estimator, split.feature_names,
                                config.RESULTS_DIR / f"importance_{tr.name}.png")
        lstm_tr = train_lstm(split, config.MODELS_DIR)
        results.append(evaluate_lstm(lstm_tr.estimator, split))
        cv_recalls["LSTM"] = lstm_tr.cv_recall
        # persist best parameters for the report
        params = {tr.name: tr.best_params for tr in classical}
        params["lstm"] = lstm_tr.best_params
        (config.RESULTS_DIR / "best_params.json").write_text(json.dumps(params, indent=2))
        cv_file = {tr.name: tr.cv_recall for tr in classical}
        cv_file["lstm"] = lstm_tr.cv_recall
        (config.RESULTS_DIR / "cv_recall.json").write_text(json.dumps(cv_file, indent=2))

    # 4. Evaluation (Sec. 3.8) ------------------------------------------------
    table = metrics_table(results, cv_recalls)
    table.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    mc = mcnemar_matrix(results)
    mc.to_csv(config.RESULTS_DIR / "mcnemar.csv", index=False)
    plot_confusion(results, config.RESULTS_DIR / "confusion_matrices.png")
    plot_roc(results, config.RESULTS_DIR / "roc_curves.png")
    times = df["WellDepth"].to_numpy()[np.sort(split.test_index)]
    onset = df.loc[df[config.TARGET_COLUMN] == 1, "WellDepth"].min()
    plot_probability_timeline(
        results, times, float(onset), config.RESULTS_DIR / "probability_timeline.png"
    )

    # test predictions for downstream use / inspection
    preds = pd.DataFrame(
        {r["model"]: r["_pred"] for r in results}
    )
    preds.insert(0, "source_row", np.sort(split.test_index))
    preds.insert(1, "actual", results[0]["_y"])
    preds.to_csv(config.RESULTS_DIR / "test_predictions.csv", index=False)

    write_summary(table, mc, df, split)
    log.info("Pipeline finished in %.1f s.", time.time() - t0)
    return table


def md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids the `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if not np.isnan(v) else ""
        return str(v)
    rows = [[fmt(v) for v in r] for r in df.itertuples(index=False)]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_summary(table: pd.DataFrame, mc: pd.DataFrame, df: pd.DataFrame, split) -> None:
    """Human-readable summary (copyable into the report results chapter)."""
    pref = ["Random Forest", "XGBoost", "Decision Tree", "SVM (RBF)", "KNN", "LSTM"]
    best = table.sort_values(["Recall", "F1", "ROC-AUC"], ascending=False)
    top_score = best.iloc[0][["Recall", "F1", "ROC-AUC"]].tolist()
    tied = best[best.apply(
        lambda r: [r["Recall"], r["F1"], r["ROC-AUC"]] == top_score, axis=1)]
    winner = sorted(tied.to_dict("records"), key=lambda r: pref.index(r["Model"]))[0]
    sig = mc[mc["p_value"] < 0.05]
    lines = [
        "# Gas-kick detection - results summary",
        "",
        "## Data",
        f"- DataDRILL `Kick_Detection.csv` (Zenodo record 12759014, DOI {config.DATASET_DOI}, CC-BY-4.0),",
        f"  integrity verified: md5 `{config.DATASET_MD5}` ({config.DATASET_SIZE_BYTES:,} bytes).",
        f"- {len(df)} rows after preprocessing (initialisation row dropped), 28 channels, "
        f"{len(config.FEATURE_COLUMNS)} modelling features (18 raw + 3 derived).",
        f"- Label: ActiveGL > 0 -> {int(df[config.TARGET_COLUMN].sum())} kick rows "
        f"({100 * df[config.TARGET_COLUMN].mean():.1f}% positive).",
        f"- Stratified 80/20 split (seed {config.SEED}): train {len(split.y_train)}, "
        f"test {len(split.y_test)}; SMOTE applied to the training partition only.",
        "",
        "## Test-partition metrics (sorted by recall)",
        md_table(table),
        "",
        "## Pairwise McNemar tests (test partition)",
        md_table(mc),
        "",
        f"- Statistically significant differences (p<0.05): {len(sig)} of {len(mc)} pairs.",
        "",
        "## Recommendation",
        f"**{winner['Model']}** is the recommended deployment model: highest recall "
        f"({winner['Recall']:.4f}) on the held-out test partition with "
        f"F1={winner['F1']:.4f} and ROC-AUC={winner['ROC-AUC']:.4f}; "
        "recall is the primary criterion because a missed kick (false negative) "
        "is the catastrophic error in drilling operations.",
        "",
    ]
    (config.RESULTS_DIR / "summary.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-train", action="store_true",
                    help="reload saved models, rerun evaluation only")
    args = ap.parse_args(argv)
    table = run_pipeline(skip_train=args.skip_train)
    print("\n=== TEST-PARTITION METRICS ===")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
