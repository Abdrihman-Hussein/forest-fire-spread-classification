import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    learning_curve,
    StratifiedKFold
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_PATH = "./dataset/clean_fire.csv"
df = pd.read_csv(DATA_PATH)

# 1) Define features and target
X = df.drop(columns=["fire_occurred"])
y = df["fire_occurred"]

# 2) Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3) Load models and scaler
lr = joblib.load(
    "./models/logistic_regression.pkl"
)

rf = joblib.load(
    "./models/random_forest.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

X_train_lr = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test_lr = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# 4) Predictions
p_lr = lr.predict(X_test_lr)
p_rf = rf.predict(X_test)

prob_lr = lr.predict_proba(X_test_lr)[:, 1]
prob_rf = rf.predict_proba(X_test)[:, 1]

# 5) Evaluation function
def evaluate_model(name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    p_lr,
    prob_lr
)

evaluate_model(
    "RANDOM FOREST",
    y_test,
    p_rf,
    prob_rf
)

# 6) Overfitting / underfitting check
# Two views are printed: the single 80/20 split and 5-fold
# cross-validation. Cross-validation is the more reliable one because
# it averages over five different train/validation splits.
baseline = max(y.mean(), 1 - y.mean())

train_acc_lr = accuracy_score(y_train, lr.predict(X_train_lr))
train_acc_rf = accuracy_score(y_train, rf.predict(X_train))

test_acc_lr = accuracy_score(y_test, p_lr)
test_acc_rf = accuracy_score(y_test, p_rf)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lr_cv_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

lr_cv = cross_validate(
    lr_cv_model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "roc_auc"],
    return_train_score=True
)

rf_cv = cross_validate(
    rf,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "roc_auc"],
    return_train_score=True
)

print("\n=== OVERFITTING / UNDERFITTING CHECK ===")
print(f"Majority-class baseline: {baseline:.4f}")

print("\nSingle 80/20 split:")
print(f"{'Model':<22}{'Train':>8}{'Test':>8}{'Gap':>8}")
print(
    f"{'Logistic Regression':<22}"
    f"{train_acc_lr:>8.4f}"
    f"{test_acc_lr:>8.4f}"
    f"{train_acc_lr - test_acc_lr:>8.4f}"
)
print(
    f"{'Random Forest':<22}"
    f"{train_acc_rf:>8.4f}"
    f"{test_acc_rf:>8.4f}"
    f"{train_acc_rf - test_acc_rf:>8.4f}"
)

lr_cv_train = lr_cv["train_accuracy"].mean()
lr_cv_valid = lr_cv["test_accuracy"].mean()
lr_cv_auc = lr_cv["test_roc_auc"].mean()

rf_cv_train = rf_cv["train_accuracy"].mean()
rf_cv_valid = rf_cv["test_accuracy"].mean()
rf_cv_auc = rf_cv["test_roc_auc"].mean()

print("\n5-fold cross-validation (shuffled):")
print(f"{'Model':<22}{'Train':>8}{'Valid':>8}{'Gap':>8}{'AUC':>8}")
print(
    f"{'Logistic Regression':<22}"
    f"{lr_cv_train:>8.4f}"
    f"{lr_cv_valid:>8.4f}"
    f"{lr_cv_train - lr_cv_valid:>8.4f}"
    f"{lr_cv_auc:>8.4f}"
)
print(
    f"{'Random Forest':<22}"
    f"{rf_cv_train:>8.4f}"
    f"{rf_cv_valid:>8.4f}"
    f"{rf_cv_train - rf_cv_valid:>8.4f}"
    f"{rf_cv_auc:>8.4f}"
)

def print_verdict(name, train_acc, valid_acc):
    gap = train_acc - valid_acc

    print(f"\n{name}:")

    if gap > 0.10:
        print(f"  Overfitting: train is {gap:.4f} above validation.")
    elif train_acc < baseline + 0.03 and valid_acc < baseline + 0.03:
        print(
            "  Underfitting: both scores sit near the "
            f"{baseline:.4f} majority baseline."
        )
    else:
        print(
            "  Balanced: train and validation are within "
            f"{abs(gap):.4f} of each other."
        )

print_verdict(
    "Logistic Regression",
    lr_cv_train,
    lr_cv_valid
)

print_verdict(
    "Random Forest",
    rf_cv_train,
    rf_cv_valid
)

# 7) Single-row sanity check
# Pick one unseen row from X_test and predict with both models.
i = 3

x_one_df = X_test.iloc[[i]]
y_true = y_test.iloc[i]

x_one_lr = pd.DataFrame(
    scaler.transform(x_one_df),
    columns=x_one_df.columns,
    index=x_one_df.index
)

p_lr_one = int(
    lr.predict(x_one_lr)[0]
)

p_rf_one = int(
    rf.predict(x_one_df)[0]
)

prob_lr_one = float(
    lr.predict_proba(x_one_lr)[0][1]
)

prob_rf_one = float(
    rf.predict_proba(x_one_df)[0][1]
)

print("\n=== SINGLE-ROW SANITY CHECK ===")
print(f"Actual Class : {y_true}")
print(f"LR Prediction: {p_lr_one} ({prob_lr_one:.2%})")
print(f"RF Prediction: {p_rf_one} ({prob_rf_one:.2%})")

# 8) Confusion Matrix - Logistic Regression
ConfusionMatrixDisplay.from_predictions(
    y_test,
    p_lr
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    "./outputs/confusion_matrix_lr.png"
)

plt.close()

# 9) Confusion Matrix - Random Forest
ConfusionMatrixDisplay.from_predictions(
    y_test,
    p_rf
)

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.tight_layout()

plt.savefig(
    "./outputs/confusion_matrix_rf.png"
)

plt.close()

# 10) Random Forest feature importance
importances = rf.feature_importances_
order = importances.argsort()

plt.figure(figsize=(8, 6))
plt.barh(
    X.columns[order],
    importances[order]
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()

plt.savefig(
    "./outputs/feature_importance.png"
)

plt.close()

# 11) Learning curve
# Shows whether more training rows would help and how wide the
# train/validation gap stays as the training set grows.
train_sizes, train_scores, val_scores = learning_curve(
    rf,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    train_sizes=np.linspace(0.2, 1.0, 5),
    shuffle=True,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(
    train_sizes,
    train_mean,
    marker="o",
    label="Training score"
)
plt.plot(
    train_sizes,
    val_mean,
    marker="o",
    label="Cross-validation score"
)
plt.xlabel("Training rows")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Random Forest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "./outputs/learning_curve.png"
)

plt.close()

print("\nSaved plots to ./outputs/")
