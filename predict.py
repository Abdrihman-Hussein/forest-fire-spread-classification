import joblib
import pandas as pd

# 1) Load saved objects
lr = joblib.load(
    "./models/logistic_regression.pkl"
)

rf = joblib.load(
    "./models/random_forest.pkl"
)

scaler = joblib.load(
    "./models/scaler.pkl"
)

feature_columns = joblib.load(
    "./models/feature_columns.pkl"
)

# The share of fires in the training data that spread. It gives the
# prediction below something to be compared against.
base_rate = pd.read_csv(
    "./dataset/clean_fire.csv"
)["fire_occurred"].mean()

# 2) New fire event
# Hot, dry afternoon in grid cell X=7, Y=5 with a long drought (DC) and
# a dry upper layer (DMC) - the conditions where fires usually spread.
new_data = pd.DataFrame({
    "X": [7],
    "Y": [5],
    "DMC": [120.0],
    "DC": [600.0],
    "temp": [30.0],
    "RH": [25],
    "wind": [5.0]
})

# 3) Put the columns in the exact order the models were trained on
new_data = new_data.reindex(
    columns=feature_columns,
    fill_value=0
)

# 4) Prepare Logistic Regression input
new_data_lr = pd.DataFrame(
    scaler.transform(new_data),
    columns=new_data.columns,
    index=new_data.index
)

# 5) Predict
lr_prediction = int(
    lr.predict(new_data_lr)[0]
)

rf_prediction = int(
    rf.predict(new_data)[0]
)

lr_probability = float(
    lr.predict_proba(new_data_lr)[0][1]
)

rf_probability = float(
    rf.predict_proba(new_data)[0][1]
)

print("\n=== FOREST FIRE SPREAD PREDICTION ===")
print(
    f"Logistic Regression: {lr_prediction} "
    f"({lr_probability:.2%})"
)
print(
    f"Random Forest      : {rf_prediction} "
    f"({rf_probability:.2%})"
)
print(f"Base rate          : {base_rate:.2%} of fires spread")
print("1 = fire burned area, 0 = no spread")
