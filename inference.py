#!/usr/bin/env python3
"""Inference CLI: score new drilling rows / sequences for gas-kick risk.

Accepts a CSV with the 28 original DataDRILL channels (header row required)
and prints, for every input row, the kick probability and class.

Examples
--------
Single batch of rows with the recommended model (Random Forest):

    python inference.py --input new_rows.csv

Sequence scoring with the LSTM (uses the trailing 50-row window, padding
shorter inputs with a validity mask):

    python inference.py --input new_rows.csv --model lstm

Quick self-test on the shipped dataset:

    python inference.py --smoke
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.data_io import derive_label, load_dataset  # noqa: E402
from src.features import derive_features, feature_matrix  # noqa: E402
from src.models import make_windows  # noqa: E402
from src.preprocessing import handle_gaps  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("inference")

CLASSICAL = {
    "decision_tree": "best_decision_tree.pkl",
    "random_forest": "best_random_forest.pkl",
    "svm": "best_svm.pkl",
    "knn": "best_knn.pkl",
    "xgboost": "best_xgboost.pkl",
}
DEFAULT_MODEL = "random_forest"  # recommended; see results/summary.md


def prepare_features(df_raw: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Derived features + the modelling feature matrix for raw 28-col rows."""
    df = derive_features(df_raw)
    X = feature_matrix(df)
    return X.to_numpy(dtype=float), df


def predict_classical(name: str, X: np.ndarray) -> np.ndarray:
    artifact = config.MODELS_DIR / CLASSICAL[name]
    with open(artifact, "rb") as fh:
        blob = pickle.load(fh)
    Xs = blob["scaler"].transform(X)
    return blob["estimator"].predict_proba(Xs)[:, 1]


def predict_lstm(X: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.models.load_model(config.MODELS_DIR / "lstm.keras")
    # reuse the scaler shipped with the recommended classical artifact
    scaler_artifact = config.MODELS_DIR / CLASSICAL[DEFAULT_MODEL]
    with open(scaler_artifact, "rb") as fh:
        scaler = pickle.load(fh)["scaler"]
    Xs = scaler.transform(X)
    # a fresh sequence: rows arrive in time order; one window ends per row
    Xw, _ = make_windows(Xs, np.zeros(len(Xs), dtype=int), np.arange(len(Xs)), stride=1)
    return model.predict(Xw, verbose=0).ravel()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=str, help="CSV with the 28 DataDRILL channels")
    ap.add_argument(
        "--model", choices=[*CLASSICAL, "lstm"], default=DEFAULT_MODEL,
        help=f"model to score with (default: {DEFAULT_MODEL})",
    )
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold on P(kick) (default: 0.5)")
    ap.add_argument("--smoke", action="store_true",
                    help="self-test: score 5 rows around the kick onset")
    args = ap.parse_args(argv)

    if args.smoke:
        df_raw = derive_label(load_dataset())
        onset = int(df_raw[df_raw[config.TARGET_COLUMN] == 1].index[0])
        df_raw = df_raw.iloc[onset - 2 : onset + 3, :].drop(columns=[config.TARGET_COLUMN])
        log.info("Smoke test: 2 rows before onset, kick onset row, 2 rows after.")
    else:
        if not args.input:
            ap.error("--input is required (or use --smoke)")
        path = Path(args.input)
        if not path.exists():
            ap.error(f"input file not found: {path}")
        df_raw = pd.read_csv(path, encoding="utf-8-sig")

    missing = [c for c in config.RAW_FEATURE_COLUMNS if c not in df_raw.columns]
    if missing:
        ap.error(f"input is missing required channels: {missing}")

    X, _ = prepare_features(df_raw)
    proba = (predict_lstm(X) if args.model == "lstm" else predict_classical(args.model, X))
    pred = (proba >= args.threshold).astype(int)

    out = pd.DataFrame({
        "row": np.arange(len(df_raw)),
        "P(kick)": np.round(proba, 4),
        "class": np.where(pred == 1, "KICK", "no kick"),
    })
    if "WellDepth" in df_raw.columns:
        out.insert(1, "time_s", df_raw["WellDepth"].to_numpy())
    print(out.to_string(index=False))
    n_kick = int(pred.sum())
    log.info("%d/%d rows flagged as KICK (model=%s, threshold=%.2f).",
             n_kick, len(pred), args.model, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
