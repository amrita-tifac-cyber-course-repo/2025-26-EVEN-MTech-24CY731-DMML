import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from preprocess import load_data

# Load data
X, y, scaler = load_data('../dataset/ai4i2020.csv')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save
joblib.dump(model, '../models/model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("Model saved!")