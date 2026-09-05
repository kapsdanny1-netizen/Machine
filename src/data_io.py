"""Dataset loading, integrity verification and label derivation."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)


def verify_dataset(path: str | Path | None = None) -> bool:
    """Verify the local dataset against the official Zenodo checksum.

    Raises
    ------
    FileNotFoundError
        If the dataset file is missing.
    RuntimeError
        If size or MD5 do not match the published record.
    """
    path = Path(path) if path is not None else config.DATASET_PATH
    blob = path.read_bytes()
    size, md5 = len(blob), hashlib.md5(blob).hexdigest()
    if size != config.DATASET_SIZE_BYTES or md5 != config.DATASET_MD5:
        raise RuntimeError(
            f"Dataset integrity check failed: size={size} md5={md5}; "
            f"expected size={config.DATASET_SIZE_BYTES} "
            f"md5={config.DATASET_MD5}. Re-download from {config.DATASET_URL}."
        )
    log.info("Dataset integrity verified (md5=%s, %d bytes).", md5, size)
    return True


def load_dataset(path: str | None = None, verify: bool = True) -> pd.DataFrame:
    """Load the verified Kick_Detection.csv into a DataFrame.

    The file is UTF-8 with BOM and CRLF line endings; ``utf-8-sig`` strips
    the BOM so column names are clean.
    """
    path = path or str(config.DATASET_PATH)
    if verify:
        verify_dataset(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.shape[1] != 28:
        raise RuntimeError(f"Expected 28 columns, found {df.shape[1]}")
    log.info("Loaded dataset: %s", df.shape)
    return df


def derive_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the binary ``kick`` label from Active Gain/Loss.

    ``ActiveGL`` is the dataset's documented kick indicator: negative while
    the pit is losing fluid to the formation (normal circulation) and
    positive during an influx (kick). The sign change coincides with the
    formation-pressure break at t~12650.15 s (FPress steps 7265 -> 8223).
    """
    out = df.copy()
    out[config.TARGET_COLUMN] = (out[config.LABEL_SOURCE_COLUMN] > 0).astype(np.int64)
    n_pos = int(out[config.TARGET_COLUMN].sum())
    log.info(
        "Label derived from %s: %d kick / %d normal rows (%.1f%% positive).",
        config.LABEL_SOURCE_COLUMN, n_pos, len(out) - n_pos, 100 * n_pos / len(out),
    )
    return out
