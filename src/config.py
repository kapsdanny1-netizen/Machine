"""Central configuration: paths, seeds, dataset facts, feature lists.

Everything that influences reproducibility lives here.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATASET_PATH: Path = DATA_DIR / "Kick_Detection.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# ----------------------------------------------------------- determinism ---
SEED: int = 42

# --------------------------------------------------- dataset provenance ----
# Zenodo record 12759014 (Arifeen et al., 2024; CC-BY-4.0).
DATASET_URL: str = (
    "https://zenodo.org/records/12759014/files/Kick_Detection.csv?download=1"
)
DATASET_MD5: str = "8bbd611ca9d66c397d43c61a152bed93"
DATASET_SIZE_BYTES: int = 581_598
DATASET_DOI: str = "10.5281/zenodo.12759014"

TARGET_COLUMN: str = "kick"  # derived label, see data_io.derive_label

# Columns that are exact duplicates of a retained column (measured on the
# verified dataset: BTBR == BDepth == WellDepth on every row).
DUPLICATE_COLUMNS: list[str] = ["BTBR", "WellDepth"]

# Columns that are constant (or become constant after row 0) on the verified
# dataset -> zero variance, no discriminative signal, dropped for modelling.
CONSTANT_COLUMNS: list[str] = [
    "CPress",   # identically 0 (casing pressure; choke open throughout)
    "AMTD",     # 78.54546 throughout
    "STP",      # 13798.08 throughout
    "MPS1",     # 99.04626 throughout
    "MPS2",     # 99.04626 throughout
    "MPS3",     # 59.66036 throughout
    "BSize",    # 0 on the first row, 12.25 afterwards
]

# ActiveGL ("Active Gain/Loss") is the documented kick indicator of the
# dataset and is used to DERIVE the label -> excluded from features to avoid
# target leakage.
LABEL_SOURCE_COLUMN: str = "ActiveGL"

# Raw sensor columns retained as features (report Sec. 3.6 priority order is
# respected by putting flow/pit/pressure channels first).
RAW_FEATURE_COLUMNS: list[str] = [
    # flow in/out differential family (top priority in the report)
    "FIn", "FOut", "MRFlow",
    # pit volume totalizer
    "ATVolume",
    # standpipe / drill-pipe pressure family
    "DPPress", "FPress",
    # rate of penetration + drilling mechanics
    "RoPen", "WoBit", "SMSpeed", "CircFlow",
    # mud rheology / density
    "FDensity", "MVis", "ATMPV", "ATMYP",
    # depth & loads
    "BDepth", "HLoad", "WBoPress",
    # pump/flow rate channel
    "FRate",
]

# Engineered features derived in features.derive_features().
DERIVED_FEATURE_COLUMNS: list[str] = [
    "FlowDiff",     # FIn - FOut  (flow in/out differential, report priority #1)
    "DExponent",    # simplified Jorden-Shirley d-exponent from ROP/WOB
    "ECDProxy",     # equivalent circulating density estimate (ppg)
]

FEATURE_COLUMNS: list[str] = RAW_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS

# ------------------------------------------------------- preprocessing -----
IQR_K: float = 1.5              # IQR fence multiplier for outlier clipping
MAX_GAP_SAMPLES: int = 3        # gaps longer than this -> rows dropped
TEST_SIZE: float = 0.20         # stratified hold-out split

# ------------------------------------------------------------ modelling ----
CV_FOLDS: int = 5               # k-fold CV on the training partition only
N_JOBS: int = -1

# LSTM temporal windows (rows are ~0.01 s apart in simulation time)
LSTM_WINDOW: int = 50           # ~0.5 s of drilling history per window
LSTM_TRAIN_STRIDE: int = 1      # overlapping windows inside train partition
LSTM_TEST_STRIDE: int = 50      # non-overlapping, independent test windows
LSTM_EPOCHS: int = 60
LSTM_BATCH_SIZE: int = 32
