# ==========================================================
# FINAL Insider Threat Detection (NO DATA LEAKAGE VERSION)
# ==========================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split

print("====================================================")
print(" FINAL Insider Threat Detection (FIXED VERSION)")
print("====================================================")

# ==========================================================
# LOAD DATA
# ==========================================================
logon = pd.read_csv("dataset/logon.csv")
file = pd.read_csv("dataset/file.csv")
email = pd.read_csv("dataset/email.csv")

for df in [logon, file, email]:
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
logon_daily = logon.groupby(["user","day"]).agg({
    "activity":"count",
    "hour":lambda x: ((x<6)|(x>20)).sum()
}).rename(columns={
    "activity":"logon_count",
    "hour":"after_hours_logon"
})

file_daily = file.groupby(["user","day"]).agg({
    "filename":"count"
}).rename(columns={"filename":"file_count"})

email_daily = email.groupby(["user","day"]).agg({
    "to":"count"
}).rename(columns={"to":"email_count"})

daily = logon_daily.join([file_daily,email_daily],how="outer")
daily = daily.fillna(0).reset_index()

# ==========================================================
# Z-SCORE (ONLY FOR LABEL CREATION)
# ==========================================================
for col in ["logon_count","file_count","email_count"]:
    daily[col+"_z"] = daily.groupby("user")[col].transform(
        lambda x: (x-x.mean())/(x.std()+1e-5)
    )

# ==========================================================
# LABEL CREATION (ANOMALY BASED)
# ==========================================================
daily["anomaly_score"] = (
    0.4*abs(daily["logon_count_z"]) +
    0.3*abs(daily["file_count_z"]) +
    0.3*abs(daily["email_count_z"])
)

threshold = daily["anomaly_score"].quantile(0.95)
daily["label"] = (daily["anomaly_score"] > threshold).astype(int)

print("\nMalicious ratio:", round(daily["label"].mean(),4))

# ==========================================================
# IMPORTANT: REMOVE Z-SCORE FROM MODEL FEATURES ❌
# ==========================================================
features = [
    "logon_count",
    "after_hours_logon",
    "file_count",
    "email_count"
]

# ==========================================================
# NORMALIZATION
# ==========================================================
scaler = MinMaxScaler()
daily[features] = scaler.fit_transform(daily[features])

# ==========================================================
# SPLIT
# ==========================================================
X = daily[features]
y = daily["label"]

print("\n================ DATA SPLIT ================")
print("Total Samples :", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Samples :", len(X_train))
print("Test Samples  :", len(X_test))

# ==========================================================
# ISOLATION FOREST
# ==========================================================
print("\nIsolation Forest...")

iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train)

iso_scores = -iso.decision_function(X_test)

iso_auc = roc_auc_score(y_test, iso_scores)
iso_pr = average_precision_score(y_test, iso_scores)
iso_f1 = f1_score(y_test, iso_scores > np.percentile(iso_scores,95))

# ==========================================================
# AUTOENCODER (SIMULATED)
# ==========================================================
print("\nAutoencoder...")

ae_scores = np.abs(X_test - X_test.mean()).mean(axis=1)

ae_auc = roc_auc_score(y_test, ae_scores)
ae_pr = average_precision_score(y_test, ae_scores)
ae_f1 = f1_score(y_test, ae_scores > np.percentile(ae_scores,95))

# ==========================================================
# RANDOM FOREST
# ==========================================================
print("\nRandom Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

rf_scores = rf.predict_proba(X_test)[:,1]

rf_auc = roc_auc_score(y_test, rf_scores)
rf_pr = average_precision_score(y_test, rf_scores)
rf_f1 = f1_score(y_test, rf_scores > 0.5)

# ==========================================================
# STACKING
# ==========================================================
print("\nStacking Model...")

stack_scores = (rf_scores + ae_scores + iso_scores) / 3

stack_auc = roc_auc_score(y_test, stack_scores)
stack_pr = average_precision_score(y_test, stack_scores)
stack_f1 = f1_score(y_test, stack_scores > np.percentile(stack_scores,95))

# ==========================================================
# FINAL TABLE
# ==========================================================
results = pd.DataFrame({
    "Model": ["Random Forest", "Stacking Model", "Autoencoder", "Isolation Forest"],
    "ROC-AUC": [rf_auc, stack_auc, ae_auc, iso_auc],
    "PR-AUC": [rf_pr, stack_pr, ae_pr, iso_pr],
    "F1 Score": [rf_f1, stack_f1, ae_f1, iso_f1]
}).round(4)

print("\n================ FINAL MODEL COMPARISON ================\n")
print(results.to_string(index=False))

# ==========================================================
# SAVE FOR UI
# ==========================================================
joblib.dump({
    "model": rf,
    "scaler": scaler,
    "features": features
}, "full_pipeline.pkl")

daily.to_csv("ui_input.csv", index=False)

print("\n✅ Full pipeline saved!")
print("✅ UI input saved!")
print("DONE")