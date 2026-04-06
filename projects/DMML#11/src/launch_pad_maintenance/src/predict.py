import os
import joblib
import numpy as np
from src.rl_agent import RLAgent

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Correct paths
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Load
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

rl = RLAgent()

def predict_system(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)

    prob = model.predict_proba(features)[0][1]
    decision = rl.decide(prob)

    return prob, decision