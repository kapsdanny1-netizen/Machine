"""Training orchestration: tuning on the training partition only (Sec. 3.7)."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from . import config
from .models import build_lstm, classical_models, make_windows
from .preprocessing import SplitData

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Fitted artefacts plus search metadata for one model."""

    name: str
    estimator: object
    best_params: dict
    cv_recall: float
    fit_seconds: float
    artifact: Path | None = None
    extra: dict = field(default_factory=dict)


def tune_classical(split: SplitData, out_dir: Path) -> list[TrainResult]:
    """Grid/Randomized search with k-fold CV on the SMOTE-resampled train set."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    results: list[TrainResult] = []
    for name, (est, grid) in classical_models().items():
        searcher = (
            RandomizedSearchCV(
                est, grid, n_iter=24, scoring="recall", cv=cv,
                random_state=config.SEED, n_jobs=config.N_JOBS, refit=True,
            )
            if name == "xgboost"
            else GridSearchCV(
                est, grid, scoring="recall", cv=cv, n_jobs=config.N_JOBS, refit=True
            )
        )
        log.info("Tuning %s ...", name)
        searcher.fit(split.X_train_res, split.y_train_res)
        log.info(
            "%s: CV recall=%.4f best=%s",
            name, searcher.best_score_, searcher.best_params_,
        )
        art = out_dir / f"best_{name}.pkl"
        with open(art, "wb") as fh:
            pickle.dump({"estimator": searcher.best_estimator_,
                         "scaler": split.scaler,
                         "feature_names": split.feature_names}, fh)
        results.append(
            TrainResult(name, searcher.best_estimator_, searcher.best_params_,
                        float(searcher.best_score_), 0.0, art)
        )
    return results


def train_lstm(split: SplitData, out_dir: Path) -> TrainResult:
    """Train the LSTM on temporal windows of the (non-resampled) train rows.

    SMOTE interpolation cannot synthesise physically coherent time series,
    so class imbalance is handled with class weights instead; this is the
    standard practice for temporal models and keeps the training partition
    free of test information.
    """
    Xw_tr, yw_tr = make_windows(
        split.X_train, split.y_train, split.train_index, config.LSTM_TRAIN_STRIDE
    )
    pos = float(yw_tr.sum())
    class_weight = {0: len(yw_tr) / (2 * (len(yw_tr) - pos)),
                    1: len(yw_tr) / (2 * pos)}
    log.info("LSTM windows: %s (positive %.1f%%).", Xw_tr.shape, 100 * yw_tr.mean())

    model = build_lstm(Xw_tr.shape[2])
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_recall", mode="max", patience=10,
        restore_best_weights=True,
    )
    hist = model.fit(
        Xw_tr, yw_tr,
        validation_split=0.15,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[stop],
        verbose=0,
    )
    best_epoch = int(np.argmax(hist.history["val_recall"]))
    cv_recall = float(max(hist.history["val_recall"]))
    log.info("LSTM trained; best val_recall=%.4f at epoch %d.", cv_recall, best_epoch + 1)

    art = out_dir / "lstm.keras"
    model.save(art)
    return TrainResult("lstm", model, {"epochs_run": best_epoch + 1,
                                       "window": config.LSTM_WINDOW},
                       cv_recall, 0.0, art,
                       extra={"class_weight": class_weight})
