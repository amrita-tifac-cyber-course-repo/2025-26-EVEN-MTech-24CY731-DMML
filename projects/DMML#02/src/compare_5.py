import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print("\nBaseline Model Comparison")


# ===============================
# LOAD USER-LEVEL DATA
# ===============================

logon = pd.read_csv("dataset/logon.csv")
file = pd.read_csv("dataset/file.csv")
email = pd.read_csv("dataset/email.csv")
answers = pd.read_csv("dataset/answers.csv")

logon['date'] = pd.to_datetime(logon['date'])
file['date'] = pd.to_datetime(file['date'])
email['date'] = pd.to_datetime(email['date'])

logon['hour'] = logon['date'].dt.hour
file['hour'] = file['date'].dt.hour
email['hour'] = email['date'].dt.hour


# ===============================
# USER-LEVEL FEATURES (STATIC)
# ===============================

logon_feat = logon.groupby('user').agg({
    'activity':'count',
    'hour': lambda x: sum((x<6)|(x>20))
}).rename(columns={
    'activity':'logon_count',
    'hour':'after_hours_logon'
})

file_feat = file.groupby('user').agg({
    'filename':'count'
}).rename(columns={
    'filename':'file_count'
})

email_feat = email.groupby('user').agg({
    'to':'count'
}).rename(columns={
    'to':'email_count'
})

features = logon_feat.join(file_feat, how='outer')
features = features.join(email_feat, how='outer')
features = features.fillna(0)


# ===============================
# LABEL MERGE
# ===============================

answers['label'] = 1

data = features.merge(answers, on='user', how='left')
data['label'] = data['label'].fillna(0)

X = data.drop(['user','label'], axis=1)
y = data['label']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)


# ===============================
# 1. Logistic Regression
# ===============================

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict_proba(X_test)[:,1]
lr_auc = roc_auc_score(y_test, lr_pred)


# ===============================
# 2. Random Forest
# ===============================

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

rf_pred = rf.predict_proba(X_test)[:,1]
rf_auc = roc_auc_score(y_test, rf_pred)


# ===============================
# 3. Static Isolation Forest
# ===============================

iso = IsolationForest(contamination=0.05)
iso.fit(X_scaled)

iso_score = iso.decision_function(X_scaled)
iso_auc = roc_auc_score(y, -iso_score)


print("\nAUC Comparison Results:\n")

print("Logistic Regression AUC :", lr_auc)
print("Random Forest AUC      :", rf_auc)
print("Static IsolationForest :", iso_auc)

print("\nDynamic Model AUC      : 0.917 (from previous run)")