# Gas-Kick Detection While Drilling — ML System

Production-ready, fully reproducible machine-learning system for binary
gas-kick detection, implementing Chapters 2–3 of the undergraduate report
*"ML Application to Predict Gas Kick While Drilling"* on the public
**DataDRILL** dataset.

## Dataset (shipped, integrity-verified)

| Item | Value |
|---|---|
| File | `data/Kick_Detection.csv` |
| Source | [Zenodo record 12759014](https://zenodo.org/records/12759014), DOI `10.5281/zenodo.12759014` |
| Licence | CC-BY-4.0 |
| Size / MD5 | 581,598 bytes — `8bbd611ca9d66c397d43c61a152bed93` |
| Content | 2,337 data rows × 28 drilling channels (single simulated well, kick at t≈12650.15 s) |

The shipped CSV is **byte-identical** to the published record (the pipeline
verifies size + MD5 on every run and refuses to proceed on mismatch). To
re-download manually:

```bash
curl -L "https://zenodo.org/records/12759014/files/Kick_Detection.csv?download=1" -o data/Kick_Detection.csv
md5sum data/Kick_Detection.csv   # 8bbd611ca9d66c397d43c61a152bed93
```

**Secondary dataset (Forge 16B, Iorkyaa et al. 2025):** not used — the Forge
16B well logs are distributed via hosts that are not programmatically
reachable from this environment and no public bulk mirror exists; the system
therefore runs on DataDRILL only, as permitted by the task definition.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.run                 # full pipeline: verify -> preprocess -> tune -> train -> evaluate
python -m src.run --skip-train    # re-evaluate from saved artifacts (models/)
python inference.py --smoke       # quick inference demo around the kick onset
```

Single command `python -m src.run` does everything end-to-end (≈3 min on a
laptop CPU, all seeds fixed → bit-reproducible on the same versions).

## Inference on new data

```bash
python inference.py --input new_rows.csv                     # Random Forest (recommended)
python inference.py --input new_rows.csv --model lstm        # temporal model
python inference.py --input new_rows.csv --model xgboost --threshold 0.3
```

Input: CSV with the 28 DataDRILL channel names in the header (any subset of
rows, in time order). Output: per-row `P(kick)` and class; sequences shorter
than the 50-row LSTM window are zero-padded with a validity mask.

## Methodology (report mapping)

**§3.6 Preprocessing** (`src/preprocessing.py`)
- Linear interpolation for short gaps (≤3 samples); rows with extended
  dropouts are dropped (the verified export is gap-free; logic included for
  field data). The simulator's zero-depth initialisation row is removed.
- IQR outlier handling (k=1.5): fences computed **on the training partition
  only**, then clipped (winsorised) on both partitions.
- `StandardScaler` fitted on train only; per-algorithm use (scale-sensitive:
  SVM/KNN/LSTM; scale-invariant trees receive the same matrices for
  consistency).
- Derived features (`src/features.py`): `FlowDiff = FIn − FOut` (flow
  in/out differential, priority #1), simplified Jorden–Shirley
  **d-exponent** from ROP/WOB (no RPM channel in the dataset — the rotation
  term is absorbed into the constant), **ECD estimate** = WBoPress/(0.052·TVD)
  in ppg, pit-volume-totalizer channel `ATVolume` retained.
- Feature priority per report: flow in/out differential & pit volume
  totalizer & drill-pipe(standpipe) pressure first, then ROP, d-exponent,
  mud density (FDensity/AMTD), ECD, plus the remaining channels.
  Constant channels (CPress, AMTD, STP, MPS1-3, BSize) and exact duplicates
  (BTBR, WellDepth ≡ BDepth) are dropped; `ActiveGL` is the label source and
  is **excluded from features** (no target leakage).
- **Stratified 80/20 split** (seed 42): train 1,868 / test 468 rows, kick
  ratio 35.8 %/35.7 %. **SMOTE applied to the training partition only.**
  The LSTM keeps natural temporal structure (SMOTE cannot synthesise
  coherent time series) and uses class weights instead.

**§3.7 Models** (`src/models.py`, `src/train.py`)
- Decision Tree, Random Forest, SVM (RBF), KNN, XGBoost, Bidirectional LSTM
  (two stacked BiLSTM layers over 50-row windows, one window per test row).
- `GridSearchCV` (XGBoost: `RandomizedSearchCV`, 24 draws) with 5-fold
  stratified CV, `scoring="recall"`, on the **training partition only**.
- Fixed seeds everywhere (`SEED=42`, deterministic TF ops).

**§3.8 Evaluation** (`src/evaluate.py`) — Accuracy, Precision, **Recall
(primary)**, F1, ROC-AUC; confusion matrices; pairwise **McNemar tests**;
metric tables and plots in `results/`; explicit winner recommendation in
`results/summary.md` (copyable into the report).

## Results (see `results/` for full artifacts)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest *(recommended)* | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LSTM | 0.9744 | 1.0000 | 0.9281 | 0.9627 | 0.9994 |

All five tabular models separate the classes perfectly on the held-out test
partition — the kick signature (formation-pressure break at t≈12650 s plus
the flow/pit-volume response) is near-linearly separable in this dataset,
consistent with the ~100 % detection reported by the dataset authors
(arXiv:2409.19724). The LSTM is the only model that does not saturate
(0.974 accuracy / 0.928 recall) because it must detect from temporal context
rather than instantaneous row values. **Recommendation: Random Forest** —
perfect recall with interpretable feature importances, plus the cheapest
robust deployment among the tied-perfect models. `results/probability_timeline.png`
shows each model's confidence ramp around the true onset.

## Repository layout

```
├── data/Kick_Detection.csv      # verified dataset (md5-checked at runtime)
├── src/                         # package: config, data_io, features,
│                                #   preprocessing, models, train, evaluate, run
├── models/                      # best_*.pkl (classical), lstm.keras (LSTM)
├── results/                     # metrics.csv, mcnemar.csv, *.png,
│                                #   test_predictions.csv, summary.md
├── inference.py                 # CLI for new rows / sequences
└── requirements.txt             # pinned dependencies
```

## Citation

Arifeen, M., Petrovski, A., Hasan, M. J., Kotenko, I., Sletov, M., &
Hassard, P. (2024). *DataDRILL: Formation Pressure Prediction and Kick
Detection for Drilling Rigs* (arXiv:2409.19724). Dataset: Zenodo,
DOI 10.5281/zenodo.12759014, CC-BY-4.0.
