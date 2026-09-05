# Gas-kick detection - results summary

## Data
- DataDRILL `Kick_Detection.csv` (Zenodo record 12759014, DOI 10.5281/zenodo.12759014, CC-BY-4.0),
  integrity verified: md5 `8bbd611ca9d66c397d43c61a152bed93` (581,598 bytes).
- 2336 rows after preprocessing (initialisation row dropped), 28 channels, 21 modelling features (18 raw + 3 derived).
- Label: ActiveGL > 0 -> 836 kick rows (35.8% positive).
- Stratified 80/20 split (seed 42): train 1868, test 468; SMOTE applied to the training partition only.

## Test-partition metrics (sorted by recall)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Recall (train) |
|---|---|---|---|---|---|---|
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVM (RBF) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9992 |
| LSTM | 0.9744 | 1.0000 | 0.9281 | 0.9627 | 0.9994 | 1.0000 |

## Pairwise McNemar tests (test partition)
| model_a | model_b | statistic | p_value | significant (p<0.05) | note |
|---|---|---|---|---|---|
| Decision Tree | Random Forest | 0.0000 | 1.0000 | False |  |
| Decision Tree | SVM (RBF) | 0.0000 | 1.0000 | False |  |
| Decision Tree | KNN | 0.0000 | 1.0000 | False |  |
| Decision Tree | XGBoost | 0.0000 | 1.0000 | False |  |
| Decision Tree | LSTM | 10.0833 | 0.0015 | True | LSTM unit = overlapping windows |
| Random Forest | SVM (RBF) | 0.0000 | 1.0000 | False |  |
| Random Forest | KNN | 0.0000 | 1.0000 | False |  |
| Random Forest | XGBoost | 0.0000 | 1.0000 | False |  |
| Random Forest | LSTM | 10.0833 | 0.0015 | True | LSTM unit = overlapping windows |
| SVM (RBF) | KNN | 0.0000 | 1.0000 | False |  |
| SVM (RBF) | XGBoost | 0.0000 | 1.0000 | False |  |
| SVM (RBF) | LSTM | 10.0833 | 0.0015 | True | LSTM unit = overlapping windows |
| KNN | XGBoost | 0.0000 | 1.0000 | False |  |
| KNN | LSTM | 10.0833 | 0.0015 | True | LSTM unit = overlapping windows |
| XGBoost | LSTM | 10.0833 | 0.0015 | True | LSTM unit = overlapping windows |

- Statistically significant differences (p<0.05): 5 of 15 pairs.

## Recommendation
**Random Forest** is the recommended deployment model: highest recall (1.0000) on the held-out test partition with F1=1.0000 and ROC-AUC=1.0000; recall is the primary criterion because a missed kick (false negative) is the catastrophic error in drilling operations.
