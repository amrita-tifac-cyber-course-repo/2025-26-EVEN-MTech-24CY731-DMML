import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# Load clean dataset
df = pd.read_csv("cicids2017_clean.csv")

FEATURES = [
    'Flow_Duration',
    'Total_Fwd_Packets',
    'Total_Backward_Packets',
    'Flow_Packets/s',
    'Fwd_Packets/s',
    'Bwd_Packets/s',
    'Packet_Length_Mean',
    'SYN_Flag_Count',
    'ACK_Flag_Count',
    'Destination_Port'
]

X = df[FEATURES]
y = df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Random Forest baseline
rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=18,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)

print("\n----- CICIDS2017 BASELINE IDS RESULTS -----")
print("Accuracy :", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall   :", round(rec, 4))
print("FPR      :", round(fpr, 4))

# Save model for RL stage
joblib.dump(rf, "baseline_rf_cicids.pkl")
print("\n Baseline RF model saved as baseline_rf_cicids.pkl")

# Save test set with probabilities for RL
probs = rf.predict_proba(X_test)[:, 1]
X_test_out = X_test.copy()
X_test_out["IDS_Score"] = probs
X_test_out["True_Label"] = y_test.values
X_test_out.to_csv("rl_input_data.csv", index=False)

print(" RL input data saved as rl_input_data.csv")
