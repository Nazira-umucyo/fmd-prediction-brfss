# Predicting Frequent Mental Distress from Lifestyle and Social Factors
### Traditional Machine Learning vs Deep Learning on the BRFSS 2024 Survey

**Author:** Nazira Umucyo, BSc Software Engineering (Machine Learning), African Leadership University

---

## Problem
Frequent mental distress (FMD) means reporting 14 or more days of poor mental health in the past 30 days. This project asks whether FMD can be predicted from everyday lifestyle, social, and physical-health factors (age, income, employment, exercise, smoking, general health, and others), as a possible screening aid that points people toward support. It is not a diagnostic tool.

The project connects to my mission of raising awareness of mental well-being, especially among young people. In this dataset, adults aged 18–24 have the highest FMD rate of any age group (21.4%, compared with 13.9% overall).

## Dataset
- **Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html), a yearly US health survey run by the Centers for Disease Control and Prevention.
- **Size:** 457,670 respondents (449,514 after removing rows with an unknown target).
- **Target:** `_MENT14D`, converted to binary: 1 = 14+ poor mental-health days, 0 = otherwise (13.86% positive).
- **Features:** 14 survey variables plus 2 engineered features (34 columns after encoding).
- The raw data is **not** stored in this repository. The notebook downloads it directly from the CDC.

## Repository structure
```
├── fmd_prediction_summative.ipynb   # full pipeline: loading, cleaning, EDA, models, evaluation
├── requirements.txt                 # library versions
├── experiments.csv                  # all 28 experiment runs (settings + validation metrics)
├── test_results.csv                 # final test-set results
├── logs/                            # TensorBoard logs for every run
└── figures/                         # plots used in the report
```

## How to run
1. Open `fmd_prediction_summative.ipynb` in [Google Colab](https://colab.research.google.com/) (File → Upload notebook, or open it from GitHub).
2. Use a **CPU** runtime (a GPU is not needed for this tabular data).
3. Click **Runtime → Run all**. The notebook downloads the dataset from the CDC by itself.
4. A full run takes about 30 minutes on CPU. The last cell backs up outputs to Google Drive and is only needed in Colab.

To run locally instead: `pip install -r requirements.txt`, then open the notebook in Jupyter. The `wget`/`unzip` commands in the data-loading cell need a Linux or macOS shell.

**Reproducibility:** random seeds are fixed (seed 42) for Python, NumPy, TensorFlow, and every data split, and all results reproduced exactly when the notebook was re-run in a fresh session.

To view the experiment logs: `tensorboard --logdir logs` (the HParams tab compares all runs).

## Methods
- **Cleaning:** survey codes for "don't know / refused" set to missing; BMI and day-count codes converted to real values.
- **Feature engineering:** BMI categories (to capture a U-shaped relationship with FMD) and an income-missing indicator.
- **Split:** 70% train / 15% validation / 15% test, stratified.
- **Preprocessing:** median imputation and standardisation for numeric features; most-frequent imputation and one-hot encoding for categorical features; fitted on the training set only.
- **Class imbalance:** class weights in all main models.
- **Traditional ML (scikit-learn):** Logistic Regression, Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`).
- **Deep learning (TensorFlow/Keras):** multi-layer perceptrons (Sequential API) with a `tf.data` input pipeline, testing network size, dropout, batch normalisation, L2 regularisation, learning rate, and batch size.
- **28 logged experiments** in total, each recorded in TensorBoard and `experiments.csv`.

## Results (test set, threshold 0.5)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Train time |
|---|---|---|---|---|---|---|
| Majority baseline | 0.500 | 0.139 | 0.000 | 0.000 | 0.000 | 0 s |
| Gradient Boosting | **0.806** | **0.442** | 0.723 | 0.312 | 0.436 | 11 s |
| MLP (neural network) | 0.805 | 0.442 | 0.728 | 0.311 | 0.436 | 96 s |

## Key findings
- **Traditional ML and deep learning performed the same.** Gradient boosting matched the neural network while training about 9 times faster.
- **The features, not the model, set the limit.** Bigger models and more data barely helped, while adding one extra variable (a prior depression diagnosis, excluded from the final models as close to circular) raised ROC-AUC from 0.807 to 0.843.
- **Class weights mattered most for usefulness,** raising recall from about 0.20 to about 0.70.
- **Early stopping on validation loss was misleading with class weights;** monitoring validation AUC fixed it.
- **The model misses distress that comes without physical illness or economic hardship,** and it is least able to tell apart young adults with and without FMD (lowest ROC-AUC of any age group, 0.747).

## Limitations
The survey has no questions on sleep (not asked in 2024), loneliness, or social support; all data is self-reported; results come from US adults and may not transfer to other countries; survey weights were not applied; and the data shows associations, not causes.

## Saved models
The saved models are not stored in this repository; they are in the project's Google Drive folder (link in the report):
- `best_ml_gradient_boosting.joblib`: scikit-learn pipeline (preprocessor + gradient boosting)
- `best_dl_mlp.keras` / `best_dl_mlp.h5`: Keras neural network
- `preprocessor.joblib`: the fitted preprocessor needed before the neural network

```python
import joblib
from tensorflow import keras

ml_pipeline = joblib.load("best_ml_gradient_boosting.joblib")
prep = joblib.load("preprocessor.joblib")
dl_model = keras.models.load_model("best_dl_mlp.keras")

# X = cleaned survey data with engineered features (see clean_brfss and add_features in the notebook)
# ml_prob = ml_pipeline.predict_proba(X)[:, 1]
# dl_prob = dl_model.predict(prep.transform(X).astype("float32")).ravel()
```
