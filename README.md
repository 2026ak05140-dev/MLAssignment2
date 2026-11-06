---
title: Dry Bean Classification Assignment
description: Reproducible multiclass classification experiments and a Streamlit evaluation app for the UCI Dry Bean dataset.
ms.date: 2026-11-06
ms.topic: tutorial
---

## Problem Statement

This project classifies seven registered dry bean varieties from 16 numeric
measurements. It trains five classification models on the same training split,
compares their performance on held-out test data, and provides an interactive
Streamlit app for evaluating the saved models.

## Dataset Description

The [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
contains 13,611 instances, 16 input features, and the `Class` target. The seven
classes are Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira. UCI reports
no missing values.

The input features are Area, Perimeter, MajorAxisLength, MinorAxisLength,
AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity,
Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, and
ShapeFactor4. The project creates a stratified 80/20 split using random state
`20261106`. This produces 10,888 training rows and 2,723 test rows. The test CSV
contains all 16 features and the target column.

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Citation: Dry Bean [Dataset]. (2020). UCI Machine Learning Repository.
[DOI: 10.24432/C50S4B](https://doi.org/10.24432/C50S4B).

## GitHub Repository Link

[GitHub Repository](https://github.com/2026ak05140-dev/MLAssignment2)

The repository should contain the following files and artifacts:

```text
MLAssignment2/
|-- app.py
|-- prepare_data.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_metrics.csv
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- gaussian_naive_bayes.joblib
    |-- support_vector_machine.joblib
```

## Models Used

All five models use the same training rows and are evaluated against the same
held-out test rows. Precision, recall, and F1 use macro averaging. AUC uses
macro-averaged one-vs-rest scores calculated from class probabilities. MCC is
the multiclass Matthews correlation coefficient. Values below are from the
local dry run and are rounded to four decimal places.

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9207   | 0.9944 | 0.9343    | 0.9329 | 0.9335 | 0.9041 |
| Decision Tree          | 0.8957   | 0.9467 | 0.9116    | 0.9117 | 0.9115 | 0.8739 |
| KNN                    | 0.9170   | 0.9843 | 0.9304    | 0.9283 | 0.9292 | 0.8997 |
| Naive Bayes (Gaussian) | 0.7631   | 0.9658 | 0.7739    | 0.7687 | 0.7682 | 0.7142 |
| Support Vector Machine | 0.9254   | 0.9910 | 0.9399    | 0.9356 | 0.9376 | 0.9098 |

### Model Observations

* Logistic Regression achieved the highest AUC (0.9944) and strong overall scores.
* Decision Tree achieved 0.8957 accuracy and the lowest AUC (0.9467).
* KNN performed competitively, with 0.9170 accuracy and 0.9843 AUC.
* Gaussian Naive Bayes had the lowest accuracy (0.7631) and MCC (0.7142).
* Support Vector Machine led in accuracy, precision, recall, F1, and MCC.
* Overall winner: Support Vector Machine. Logistic Regression had the highest AUC.

Logistic Regression, KNN, and SVM include a `StandardScaler` in their saved
pipelines. The SVM uses five-fold sigmoid probability calibration so that the
same probability-based AUC method can be used for all five models.

## Reproduce the Local Dry Run

The preparation script downloads the official UCI archive, checks the row,
feature, class, and missing-value requirements, then writes stratified train
and test CSV files. The training script fits each classifier using only the
training CSV, evaluates it using the test CSV, saves all model pipelines under
`model/`, and writes `model_metrics.csv`.

```powershell
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe prepare_data.py
.venv\Scripts\python.exe train_models.py
.venv\Scripts\python.exe -m streamlit run app.py
```

To reproduce the BITS Virtual Lab run, install the pinned dependencies there,
run `prepare_data.py` and `train_models.py`, then start the app. Capture one
screenshot of assignment execution in BITS Virtual Lab for the submission PDF.
If the lab cannot download from UCI, transfer the generated train and test CSV
files to the lab and run `train_models.py` there.

## Streamlit App

The app loads the saved model pipelines and defaults to the included
`test_data.csv`. You can upload another test CSV containing the same 16 numeric
features and all seven `Class` labels. The app displays a comparison of all five
models, the six required metrics for the selected model, a confusion matrix,
and a classification report. It does not retrain models on uploaded data.

For deployment, push the repository to GitHub, sign in at
[Streamlit Community Cloud](https://streamlit.io/cloud), create a new app from
the repository, select the `main` branch and `app.py`, then deploy. Confirm the
live link opens the interactive app and loads without errors.

## Submission Checklist

Submit one PDF containing the following items in this order:

1. The GitHub repository link, including the source code, `requirements.txt`,
   this README, and experiment test CSV.
2. The live Streamlit app link.
3. One screenshot showing execution in BITS Virtual Lab.
4. The README content.

Before submission, check both links, confirm the app opens without errors, and
include this README in the PDF.
