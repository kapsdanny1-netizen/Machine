"""Row-local engineered features (report Sec. 3.6 feature priority).

All transforms here are functions of a single row (no fitted statistics),
so they are computed before the train/test split without leakage.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

_EPS = 1e-12


def d_exponent(rop: pd.Series, wob: pd.Series) -> pd.Series:
    """Simplified Jorden-Shirley d-exponent.

    The classical form is d = log10(ROP / (60 N)) / log10(12 WOB / (1e6 db)).
    The dataset has no rotary-speed (RPM) channel, so the depth/bit-size
    normalisation is applied with the available channels and the rotation
    term is absorbed into the constant (a standard simplification when N is
    unavailable -- documented in the README). Low d-exponent values are the
    classic indicator of abnormal pore pressure / kick conditions.
    """
    num = np.log10(rop.clip(lower=1e-6))
    den = np.log10(wob.clip(lower=1e-6))
    return (num / den.replace(0, np.nan)).fillna(0.0)


def ecd_proxy(wbo_press: pd.Series, depth: pd.Series) -> pd.Series:
    """Equivalent circulating density estimate in ppg.

    ECD = annular pressure / (0.052 * TVD), the standard oilfield relation,
    using wellbore pressure (psi) and bit depth (ft) as TVD proxy.
    """
    return wbo_press / (0.052 * depth.clip(lower=_EPS))


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with FlowDiff, DExponent and ECDProxy added."""
    out = df.copy()
    out["FlowDiff"] = out["FIn"] - out["FOut"]
    out["DExponent"] = d_exponent(out["RoPen"], out["WoBit"])
    out["ECDProxy"] = ecd_proxy(out["WBoPress"], out["BDepth"])
    log.info(
        "Derived features: FlowDiff, DExponent, ECDProxy -> %d total columns.",
        out.shape[1],
    )
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select the modelling feature matrix in the configured order."""
    missing = [c for c in config.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[config.FEATURE_COLUMNS].astype(float)
