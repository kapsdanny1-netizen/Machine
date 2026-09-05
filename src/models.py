"""Model definitions and hyper-parameter search spaces (report Sec. 3.7).

Classical models: Decision Tree, Random Forest, SVM (RBF), KNN, XGBoost.
Deep model: bidirectional LSTM over temporal windows of scaled features.

All estimators share the fixed global seed from ``config.SEED``.
"""
from __future__ import annotations

import logging

import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from . import config

log = logging.getLogger(__name__)

# Deterministic TensorFlow (CPU) for reproducible LSTM training.
tf.keras.utils.set_random_seed(config.SEED)
tf.config.experimental.enable_op_determinism()


def classical_models() -> dict[str, tuple[object, dict]]:
    """Return {name: (estimator, search grid)} for the classical models.

    Grids are modest but meaningful; scoring is recall (missed kick is the
    catastrophic error), applied via k-fold CV on the training partition.
    """
    return {
        "decision_tree": (
            DecisionTreeClassifier(random_state=config.SEED, class_weight="balanced"),
            {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [1, 3, 5, 10],
                "criterion": ["gini", "entropy"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                random_state=config.SEED, class_weight="balanced", n_jobs=config.N_JOBS
            ),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [5, 10, 20, None],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.5],
            },
        ),
        "svm": (
            SVC(random_state=config.SEED, probability=True, class_weight="balanced"),
            {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", 0.01, 0.1],
                "kernel": ["rbf"],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 9, 11, 15],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        ),
        "xgboost": (
            XGBClassifier(
                random_state=config.SEED,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=config.N_JOBS,
            ),
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "scale_pos_weight": [1, (1500 - 836) / 836],
            },
        ),
    }


def build_lstm(n_features: int) -> tf.keras.Model:
    """Bidirectional LSTM classifier over ``config.LSTM_WINDOW``-row windows."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(config.LSTM_WINDOW, n_features)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(48, return_sequences=True)
            ),
            tf.keras.layers.Dropout(0.25, seed=config.SEED),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(24)),
            tf.keras.layers.Dropout(0.25, seed=config.SEED),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="kick_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    stride: int,
    window: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build temporal windows from contiguous rows of one partition.

    Windows are built strictly inside each partition (no train/test row
    sharing). Contiguity is established on the source dataframe row index so
    that a shuffled split still yields correct temporal windows: rows are
    re-ordered by original position, windowed, and the window keeps its
    position in partition order. A window is labelled with the class of its
    LAST row (the label the operator would need at that moment).

    Padded (shorter than the window) leading sequences are included with a
    validity mask so early rows are still usable; the Masking layer ignores
    the padded steps.
    """
    window = window or config.LSTM_WINDOW
    order = np.argsort(indices)
    Xs, ys = X[order], y[order]
    feats: list[np.ndarray] = []
    labels: list[int] = []
    n = len(Xs)
    for end in range(1, n + 1, stride):
        start = max(0, end - window)
        w = Xs[start:end]
        pad = window - len(w)
        if pad:
            w = np.vstack([np.zeros((pad, X.shape[1])), w])
        feats.append(w)
        labels.append(int(ys[end - 1]))
    return np.stack(feats), np.asarray(labels, dtype=int)
