# NEO-Hazard-Prediction
A machine learning project for predicting whether a Near-Earth Object (NEO) is hazardous to Earth, based on NASA's Near-Earth Object dataset. The final model is a LightGBM classifier packaged as a joblib bundle, ready for deployment or downstream analysis.

## Project Overview
The dataset contains ~90,836 records of NASA-certified near-Earth asteroids. The goal is to build a binary classifier that identifies which celestial bodies pose a threat to Earth — a safety-critical task where missing a hazardous object (False Negative) is far more costly than a false alarm (False Positive).

Four models were benchmarked (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM); LightGBM was selected as the final model because it offers the best combination of recall, ROC-AUC, and training speed.

Source: NASA Nearest Earth Objects — Kaggle [https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects/data]
The dataset is a snapshot of NASA's Near-Earth Object catalog. It contains physical and orbital characteristics of asteroids whose trajectories bring them close to Earth, together with a binary label indicating whether each object is classified as hazardous (i.e. potentially threatening to Earth).

⚠️ Lessons Learned
1. Feature order is a silent killer.
LightGBM stores feature_name_ in training order. If the inference code feeds a numpy array (no names) or a DataFrame in a different order, the model receives semantically wrong features and produces degenerate predictions (e.g. prob.min ≈ 0.82 → FN = 0). The bundle stores feature_cols = list(model.feature_name_) and the inference code passes a DataFrame so LightGBM can verify names. A mismatch now raises instead of silently failing.

2. Never double-weight the positive class.
Using both scale_pos_weight ≈ 9.28 and class_weight='balanced' multiplies the effective positive weight to ~47.7 and pushes all predictions to 1. This was the root cause of an earlier "FN = 0, FP = 16,400" degenerate run. Correct: use only scale_pos_weight.

3. eval_metric matters for early stopping.
With scale_pos_weight, the default binary_logloss stops early at ~10 trees and produces Recall = 0. Switching to eval_metric='auc' yields a proper stopping point (~328 trees) and Recall ≈ 0.97.

4. No test leakage.
Early stopping uses a validation set carved out of X_train; RandomizedSearchCV uses its own internal CV splits of X_train; X_test is touched only once, for the final evaluation.

5. The bundle must be saved with verified model state.
Before joblib.dump, the code asserts prob.min() < 0.1 on the correct feature order, so a degenerate model is never persisted. After saving, it reloads the bundle and re-verifies.

6. FP are a data limitation, not a bug.
With the current 6 features, false alarms are statistically indistinguishable from true positives. Reducing FP requires new features (orbital period, eccentricity, inclination, albedo, radial velocity) or a second-stage model.

## Future Work
1. New features: orbital period, eccentricity, inclination, radial velocity, albedo, chemical composition, time to closest approach, log(kinetic_energy), miss_distance / diameter.

2. Two-stage model: LightGBM as a high-recall first-stage filter, then a precise second-stage model (e.g. calibrated GBM or a small neural net) on the ~20% of objects that pass stage 1.

3. Probability calibration: Platt / Isotonic scaling — does not change ranking, but makes the probabilities interpretable (currently prob.mean = 0.244 at an actual positive rate of 0.097).

4. Threshold as a deployment parameter: the bundle already stores threshold; expose it via CLI / config to switch between safety-first (0.30) and balanced (0.50).

5. Cross-validation: replace the single train/test split with stratified k-fold to report mean ± std for all metrics.

## Acknowledgements
- Sameep Vani[https://www.kaggle.com/sameepvani] for publishing the dataset on Kaggle.

- NASA Open API & NEO Earth Close Approaches for the original data.

- LightGBM, scikit-learn, XGBoost, CatBoost maintainers.

- Reviewers who caught the feature-order and double-weighting bugs during the error-analysis phase.
