# 🔥 Forest Fire Spread Classification

A Machine Learning classification project that predicts whether a forest fire in the Montesinho Natural Park will burn any area (`fire_occurred`: 0 = no spread, 1 = burned area) using weather conditions and the Canadian Fire Weather Index (FWI) system.

---

## 📌 Problem Statement & Overview

### The Problem
The Montesinho Natural Park in northern Portugal records hundreds of small fires every year. Most of them are put out before they consume any fuel, while a small number grow into large burns. Fire brigades cannot treat every report the same way, so a model that separates fires likely to spread from fires likely to die out helps decide where to send crews first.

### The Machine Learning Solution
This project implements a binary classification model comparing **Logistic Regression** and **Random Forest** classifiers. The dataset is cleaned by removing duplicate records, the burned `area` is turned into a binary `fire_occurred` target, the month is one-hot encoded, and the day is reduced to a weekend flag. Numerical features are standardized for Logistic Regression.

The dataset is known to be difficult. Weather conditions tell you a little about whether a fire spreads, but not much. The honest result is a model that beats the 52.4% majority-class baseline by only a few points, and the project documents that limit instead of hiding it.

---

## ⚙️ Pipeline Architecture

```
                 ┌──────────────────────────┐
                 │ dataset/raw_dataset.csv  │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │    preprocessing.py      │
                 │ • Drop Duplicate Records │
                 │ • area -> fire_occurred  │
                 │ • One-Hot Encode month   │
                 │ • Add weekend / rain     │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │     clean_fire.csv       │
                 └────────────┬─────────────┘
                              │
                              ▼
                 ┌──────────────────────────┐
                 │        train.py          │
                 │ • Stratified Train/Test  │
                 │ • StandardScaler         │
                 │ • Train LR & RF Models   │
                 └────────────┬─────────────┘
                              │
      ┌───────────────────────┴───────────────────────┐
      ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│         evaluate.py           │               │          predict.py           │
│ • Precision, Recall, F1       │               │ • Single Fire Event           │
│ • Overfitting Check           │               │   Inference                   │
│ • Confusion Matrix Plots      │               │                               │
└───────────────────────────────┘               └───────────────────────────────┘
```

---

## 📊 Dataset & Feature Description

### Input Features

| Feature | Description | Range / Values |
| :--- | :--- | :--- |
| `X` | x-axis coordinate in the Montesinho park map | 1 to 9 |
| `Y` | y-axis coordinate in the Montesinho park map | 2 to 9 |
| `FFMC` | Fine Fuel Moisture Code (FWI system) | 18.7 to 96.2 |
| `DMC` | Duff Moisture Code (FWI system) | 1.1 to 291.3 |
| `DC` | Drought Code (FWI system) | 7.9 to 860.6 |
| `ISI` | Initial Spread Index (FWI system) | 0.0 to 56.1 |
| `temp` | Temperature in Celsius | 2.2 to 33.3 |
| `RH` | Relative humidity in percent | 15 to 100 |
| `wind` | Wind speed in km/h | 0.4 to 9.4 |
| `rain_flag` | `1` if any rain fell, `0` otherwise | Only 8 of 513 rows have rain |
| `is_weekend` | `1` for Saturday or Sunday | Derived from `day` |
| `month_*` | One-hot encoded month (11 columns, April dropped) | `0` / `1` |

The raw `rain` column was replaced by `rain_flag` because only 8 rows have a non-zero value, so the exact amount adds noise without adding signal. The raw `day` column became `is_weekend` because weekday fires are mostly human-caused and that is the part a model can use.

### Target Variable
- **`fire_occurred`**: `1` = the fire burned an area greater than 0 ha, `0` = it burned no area.
- Raw class counts after cleaning: **269** fires spread, **244** fires did not (52.4% / 47.6%).

The original `area` column is dropped after the target is built so it cannot leak into the model.

### Outliers — checked, not removed

The dataset documentation warns that "it is not clear how many outliers are there". Both kinds were checked in `preprocessing.py`, and both were kept.

**Target.** 11 fires burned more than 100 ha, and the largest reached 1090.84 ha against a median of 0.54 ha. This skew is the reason the target is binarized at `area > 0`: once binarized, a 1000-ha fire and a 1-ha fire are both class 1, so the long tail no longer distorts the model. It matters for regression, not classification.

**Features.** 149 of 513 rows (29%) fall outside 1.5 × IQR on at least one weather index, mostly `Y`, `FFMC`, `DMC` and `DC`. They are real fire weather, not recording errors, so removing them deletes the cases the model exists to recognize:

| Handling | Rows | CV accuracy | CV AUC |
| :--- | ---: | ---: | ---: |
| **Keep (chosen)** | 513 | **0.567** | **0.579** |
| Drop 1.5 × IQR outliers | 364 | 0.563 | 0.546 |
| Drop 3 × IQR outliers | 487 | 0.567 | 0.576 |
| Winsorize at 1% / 99% | 513 | 0.567 | 0.579 |

Dropping the 11 largest fires lifts CV accuracy to 0.592, but that is not a real gain: it removes the rarest and hardest events from the evaluation set, which is exactly the population a fire model should care about.

---

## 🤖 Model & Feature Engineering

1. **Logistic Regression**
   - All features standardized with `StandardScaler` (fit on the training rows only).
   - Linear baseline that is easy to interpret.

2. **Random Forest Classifier**
   - 300 trees, `max_depth=2`, `min_samples_leaf=3`.
   - The depth limit is deliberate. An unrestricted forest reaches 0.99 training accuracy against 0.58 cross-validation accuracy, a 0.41 gap: it memorizes the training rows. Shallow trees keep the gap at 0.07 with the same validation accuracy.

### Why the model is kept small
This table comes from 5-fold stratified cross-validation on the cleaned data:

| Random Forest setting | Train acc | Valid acc | Gap | Valid AUC |
| :--- | ---: | ---: | ---: | ---: |
| No depth limit | 0.994 | 0.581 | 0.413 | 0.603 |
| `max_depth=6`, leaf 3 | 0.816 | 0.567 | 0.248 | 0.595 |
| `max_depth=4`, leaf 3 | 0.737 | 0.579 | 0.158 | 0.586 |
| `max_depth=3`, leaf 3 | 0.680 | 0.571 | 0.109 | 0.583 |
| **`max_depth=2`, leaf 3** | **0.634** | **0.567** | **0.067** | **0.581** |

Deeper forests only raise the training score. Validation accuracy stays flat around 0.57, so the extra depth is pure overfitting. Depth 4 buys 0.012 more cross-validation accuracy but reopens a 0.158 train/validation gap, which is not worth it.

---

## 📈 Evaluation & Diagnostics

Evaluated on a stratified 20% hold-out split (`random_state=42`) and with 5-fold stratified cross-validation.

**Hold-out test split (103 rows):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6311 | 0.6250 | 0.7407 | 0.6780 | 0.6924 |
| Random Forest | 0.6505 | 0.6125 | 0.9074 | 0.7313 | 0.7551 |

**Overfitting / underfitting check (5-fold CV):**

| Model | Train | Valid | Gap | Valid AUC |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6062 | 0.5361 | 0.0702 | 0.5542 |
| Random Forest | 0.6345 | 0.5673 | 0.0672 | 0.5811 |

Majority-class baseline: **0.5244**. Both models are above it, and neither shows a gap large enough to call overfitting. Logistic Regression is close to the baseline and is the weaker model; the forest is the one worth keeping.

### Why the accuracy is capped

The ~0.65 test accuracy is not a tuning mistake. Every model tried lands in the same narrow band on 5-fold cross-validation:

| Model | CV accuracy | CV ROC-AUC |
| :--- | ---: | ---: |
| Hist Gradient Boosting | 0.583 | 0.607 |
| Random Forest (no depth limit) | 0.581 | 0.603 |
| Extra Trees | 0.573 | 0.598 |
| **Random Forest (`max_depth=2`)** | **0.567** | **0.581** |
| K-Nearest Neighbors (k=25) | 0.562 | 0.576 |
| Logistic Regression (C=0.01) | 0.560 | 0.564 |
| Support Vector Classifier (RBF) | 0.557 | 0.583 |
| AdaBoost | 0.554 | 0.555 |
| Gradient Boosting | 0.548 | 0.569 |
| Gaussian Naive Bayes | 0.544 | 0.561 |
| **Shuffled labels (null test)** | **0.544** | **0.525** |

The last row is the key one. When the target is randomly permuted, the same forest still scores 0.544 because 52% of the rows are positive. Real models reach only 0.57. The gap between "learned something" and "guessed the majority class" is about 2 to 4 points.

Two further checks ruled out the usual suspects:

- **Engineered domain features.** Adding the Fire Weather Index's BUI (`0.8 * DMC * DC / (DMC + 0.4 * DC)`), an FWI approximation, `FFMC * ISI`, `temp / RH`, `wind * ISI`, month as an ordinal, seasons, or one-hot weekdays changed CV accuracy by at most 0.002.
- **Decision-threshold tuning.** The default 0.50 threshold is already the best for accuracy (0.567). Lowering it to 0.40 raises F1 to 0.691 but drops accuracy, which is just relabelling more rows as positive.
- **Probability resolution and calibration.** The depth-2 forest is not outputting one blunt number: it produces 171 distinct probabilities spanning 0.37 to 0.68. Sigmoid calibration made the ranking worse (CV ROC-AUC 0.48), so the raw probabilities are kept.

The dataset documentation reaches the same conclusion: it describes a "very difficult regression task" where large fires are rare and the best RMSE came from the naive mean predictor. Weather conditions alone do not decide whether a fire spreads — fuel, terrain, wind gusts and how fast crews arrive are not in the file.

Plots written to `outputs/`:
- `confusion_matrix_lr.png`
- `confusion_matrix_rf.png`
- `feature_importance.png` — temperature, humidity, wind, DMC and DC carry most of the signal
- `learning_curve.png` — training and validation scores as the training set grows

---

## 💻 How to Run

```bash
# 1. Clean dataset and build the binary target
python preprocessing.py

# 2. Train both classifiers and export model binaries
python train.py

# 3. Evaluate metrics, check overfitting, and generate plots
python evaluate.py

# 4. Run a single fire event prediction
python predict.py
```

---

## 🌐 Web Interface

The model is wrapped in a small browser UI called **Emberline**. It uses only the
Python standard library (`ThreadingHTTPServer`) plus the already-installed
scikit-learn and pandas — no Flask, no FastAPI, no build step.

```bash
python app.py
# Emberline running at http://127.0.0.1:8000  (Ctrl+C to stop)
```

Then open <http://127.0.0.1:8000>.

The page shows nine sliders (coordinates, the four fire-weather indices,
temperature, humidity, wind), a month selector, and weekday/weekend and
rain toggles. Submitting calls the API and renders the forest probability,
the logistic-regression probability, and the 52.44% historical base rate as
a dashed marker on both bars, so the estimate is always read against the
"coin flip" line.

The hero image is a real wildfire photograph, not a generated one:
**"Uncontrollable wildfire, forest fire"** from rawpixel, released under
**CC0 1.0** (public domain, no attribution required). It is served locally
from `web/assets/forest-fire.jpg` at 1024 px / 132 KB, so the page needs no
external network access.

### API

| Method | Route | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the estimator page |
| `GET` | `/static/*` | Serves `app.js` and `styles.css` |
| `GET` | `/api/health` | Liveness probe → `{"status": "ok"}` |
| `GET` | `/api/model-info` | Test-split metrics, row counts, base rate |
| `POST` | `/api/predict` | Validated prediction for one fire event |

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"X":7,"Y":5,"month":"aug","day_type":"weekend","rain":"no",
       "FFMC":92.5,"DMC":120,"DC":600,"ISI":12,"temp":30,"RH":25,"wind":5}'
```

```json
{"prediction":1,"probability":0.546,"label":"Likely to spread",
 "logistic_probability":0.6442,"logistic_prediction":1,
 "base_rate":0.5244,"month":"aug"}
```

The server validates every field against the documented dataset range before
touching the model: out-of-range or missing values return `422`, malformed
JSON returns `400`, and `/static/../app.py` returns `404` so the server cannot
be walked out of the `web/` directory. `month` is encoded the same way as
`preprocessing.py` — April maps to all-zero dummies because `drop_first=True`
dropped it.

### Web tests

```bash
python -m unittest discover -s tests
```

13 smoke tests cover the health route, the HTML page, static assets, the hero
image, path traversal, the sample prediction (matching `predict.py`), the base
rate, the dry-vs-wet direction, range and missing-field rejection, invalid
JSON, and the model-info metrics.

---

## 📁 Directory Structure

```
forest+fires/
├── dataset/
│   ├── raw_dataset.csv          # Raw UCI Forest Fires data
│   └── clean_fire.csv           # Cleaned & encoded dataset
├── models/                      # Saved binaries (*.pkl - ignored by Git)
│   ├── feature_columns.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── outputs/                     # Visual diagnostic plots
│   ├── confusion_matrix_lr.png
│   ├── confusion_matrix_rf.png
│   ├── feature_importance.png
│   └── learning_curve.png
├── preprocessing.py             # Cleaning, target building & encoding
├── train.py                     # Training script
├── evaluate.py                  # Metrics, overfitting check & plots
├── predict.py                   # Single fire event prediction script
├── app.py                       # Stdlib web server (UI + JSON API)
├── web/
│   ├── index.html               # Emberline estimator page
│   ├── styles.css               # Ember palette, sliders, comparison bars
│   ├── app.js                   # Form sync, validation, fetch, rendering
│   └── assets/
│       └── forest-fire.jpg      # Hero photo (rawpixel, CC0 1.0)
├── tests/
│   └── test_app.py              # 13 unittest smoke tests for the web layer
├── requirements.txt             # Requirements
└── README.md                    # Project documentation
```
