"""Preprocessing pipeline (report Sec. 3.6).

Order of operations (leakage-safe):
1. row-local derived features (features.derive_features) -- no fitted stats;
2. gap handling on the time axis: short gaps linearly interpolated,
   extended dropouts dropped (rows removed);
3. stratified 80/20 train/test split (seeds fixed);
4. IQR outlier fences and scalers fitted on the TRAIN partition only;
5. SMOTE applied to the TRAIN partition only (never to test).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitData:
    """Container for the processed train/test arrays."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_res: np.ndarray          # SMOTE-resampled training features
    y_train_res: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    train_index: np.ndarray          # row indices of the source dataframe
    test_index: np.ndarray


def handle_gaps(df: pd.DataFrame, time_col: str = "WellDepth") -> pd.DataFrame:
    """Interpolate short gaps and drop extended dropouts.

    The verified DataDRILL export is a continuous simulation and contains no
    missing values, so in practice this is a no-op that logs its findings;
    the logic is kept general for field data with real dropouts.

    Rows with non-physical depth (BDepth <= 0: the simulator's zero-state
    initialisation row) are dropped because depth-normalised features
    (d-exponent denominators, ECD) are undefined there.
    """
    out = df.copy()
    n_init = int((out["BDepth"] <= 0).sum())
    if n_init:
        out = out[out["BDepth"] > 0]
        log.info("Dropped %d non-physical initialisation row(s) (BDepth<=0).", n_init)
    n_missing = int(out.isna().sum().sum())
    if n_missing:
        log.info("Missing cells detected: %d -> interpolating short gaps.", n_missing)
        out = out.interpolate(method="linear", limit=config.MAX_GAP_SAMPLES, limit_area="inside")
        # extended dropouts: any row still carrying NaN in a feature is dropped
        before = len(out)
        out = out.dropna(subset=[c for c in config.FEATURE_COLUMNS if c in out.columns])
        log.info("Dropped %d rows with extended dropouts.", before - len(out))
    else:
        log.info("No missing values found (continuous simulation export).")
    return out.reset_index(drop=True)


def iqr_clip_fences(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute IQR clipping fences (Q1-k*IQR, Q3+k*IQR) column-wise."""
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return q1 - config.IQR_K * iqr, q3 + config.IQR_K * iqr


def build_split(df: pd.DataFrame) -> SplitData:
    """Run the full Sec. 3.6 preprocessing and return the split container."""
    from .features import feature_matrix  # local import to avoid cycle

    X_all = feature_matrix(df).to_numpy(dtype=float)
    y_all = df[config.TARGET_COLUMN].to_numpy(dtype=int)

    idx = np.arange(len(df))
    X_train, X_test, y_train, y_test, tr_idx, te_idx = train_test_split(
        X_all,
        y_all,
        idx,
        test_size=config.TEST_SIZE,
        stratify=y_all,
        random_state=config.SEED,
    )
    log.info(
        "Stratified split: train=%d (%.1f%% kick), test=%d (%.1f%% kick).",
        len(y_train), 100 * y_train.mean(), len(y_test), 100 * y_test.mean(),
    )

    # IQR outlier handling: fences from TRAIN only, clipped on both partitions.
    lo, hi = iqr_clip_fences(X_train)
    n_clipped = int(((X_train < lo) | (X_train > hi)).sum()
                    + ((X_test < lo) | (X_test > hi)).sum())
    X_train = np.clip(X_train, lo, hi)
    X_test = np.clip(X_test, lo, hi)
    log.info("IQR clipping (k=%.1f): %d cell values winsorised.", config.IQR_K, n_clipped)

    # Scaling fitted on train only (identical transform applied at inference).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE on the training partition only.
    sm = SMOTE(random_state=config.SEED)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)
    log.info(
        "SMOTE (train only): %d -> %d rows (%.1f%% positive after resampling).",
        len(y_train), len(y_res), 100 * y_res.mean(),
    )

    return SplitData(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        X_train_res=X_res,
        y_train_res=y_res,
        feature_names=list(config.FEATURE_COLUMNS),
        scaler=scaler,
        train_index=tr_idx,
        test_index=te_idx,
    )
