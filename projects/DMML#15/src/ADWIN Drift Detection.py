import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from river import drift
import warnings

# Suppress warnings for a clean output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BENIGN_PATH = "C:/Users/adity/Desktop/MTECH_2ND_SEM/DMML/Mini Project/Dataset/Total_CSVs/l2-benign.csv"
MALICIOUS_PATH = "C:/Users/adity/Desktop/MTECH_2ND_SEM/DMML/Mini Project/Dataset/Total_CSVs/l2-malicious.csv"

def load_and_multi_drift(b_path, m_path):
    print("[1/6] Loading Data and Creating Multi-Drift Scenario...")
    try:
        df_b = pd.read_csv(b_path)
        df_m = pd.read_csv(m_path)
    except FileNotFoundError:
        print("Error: CSV files not found. Please check your filenames.")
        return None, None, None, None

    # Feature Cleaning
    drop_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'Label']
    X_b = df_b.drop(columns=[col for col in drop_cols if col in df_b.columns]).fillna(0)
    X_m = df_m.drop(columns=[col for col in drop_cols if col in df_m.columns]).fillna(0)
    
    # --- PHASED MULTI-DRIFT CONSTRUCTION ---
    # Phase 1: Baseline (1000 Benign)
    X_train = X_b.iloc[:1000]
    y_train = np.zeros(1000) 
    
    # Building the Stream: Benign -> Malicious -> Benign -> Malicious -> Benign
    # This simulates a "pulsing" attack or intermittent tunneling activity
    s1_b = X_b.iloc[1000:2000]  # Normal
    s2_m = X_m.iloc[:1000]       # Attack 1 (Drift 1)
    s3_b = X_b.iloc[2000:3000]  # Back to Normal (Drift 2)
    s4_m = X_m.iloc[1000:2000]  # Attack 2 (Drift 3)
    s5_b = X_b.iloc[3000:4000]  # Back to Normal (Drift 4)
    
    X_stream = pd.concat([s1_b, s2_m, s3_b, s4_m, s5_b]).reset_index(drop=True)
    y_stream = np.concatenate([
        np.zeros(len(s1_b)), 
        np.ones(len(s2_m)), 
        np.zeros(len(s3_b)), 
        np.ones(len(s4_m)), 
        np.zeros(len(s5_b))
    ])
    
    return X_train, y_train, X_stream, y_stream

def run_simulation():
    X_train, y_train, X_stream, y_stream = load_and_multi_drift(BENIGN_PATH, MALICIOUS_PATH)
    if X_train is None: return
    
    print("[2/6] Training Initial Models...")
    static_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    adaptive_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    
    static_model.fit(X_train, y_train)
    adaptive_model.fit(X_train, y_train)
    
    # --- DRIFT DETECTION SETUP ---
    delta_val = 0.001
    adwin = drift.ADWIN(delta=delta_val) 
    batch_size = 50 
    
    drift_indices, a_accs, s_accs, indices = [], [], [], []
    
    print("[3/6] Processing Stream with Multi-Phase Traffic...")
    
    for i in range(0, len(X_stream), batch_size):
        X_batch = X_stream.iloc[i : i + batch_size]
        y_batch = y_stream[i : i + batch_size]
        if len(X_batch) < batch_size: break
        
        # Test Static (Stays trained only on the first Benign set)
        s_accs.append(accuracy_score(y_batch, static_model.predict(X_batch)))
        
        # Test Adaptive
        a_pred = adaptive_model.predict(X_batch)
        a_accs.append(accuracy_score(y_batch, a_pred))
        indices.append(i)
        
        # Monitor for multiple drifts
        for pred, actual in zip(a_pred, y_batch):
            error = 1 if pred != actual else 0
            adwin.update(error)
            
            if adwin.drift_detected:
                print(f"--> ALERT: Concept Drift Detected at Traffic Index {i}!")
                drift_indices.append(i)
                # Retrain on current data to adapt to the new pattern (Attack or Normal)
                adaptive_model.fit(X_batch, y_batch) 
                adwin = drift.ADWIN(delta=delta_val) # Fresh start for next drift
                break

    # --- VISUALIZATION ---
    print("[4/6] Generating Comparison Plot...")
    plt.figure(figsize=(14, 7))
    
    plt.plot(indices, s_accs, label='Static Model (Fixed Boundaries)', color='red', alpha=0.4, linestyle='--')
    plt.plot(indices, a_accs, label='Adaptive Model (ADWIN Controlled)', color='green', linewidth=2.5)
    
    # Vertical lines for every drift detection
    for d in drift_indices:
        plt.axvline(x=d, color='blue', linestyle=':', alpha=0.8, label='Drift Detected' if d == drift_indices[0] else "")
        
    plt.title('Multi-Drift DNS Tunneling Detection: Adaptability Analysis')
    plt.xlabel('Traffic Sample Index')
    plt.ylabel('Model Accuracy')
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='lower left', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n[5/6] Final Analysis Summary:")
    print("-" * 45)
    print(f"Total Drift Events Detected: {len(drift_indices)}")
    print(f"Avg Accuracy (Adaptive):     {np.mean(a_accs):.4f}")
    print(f"Avg Accuracy (Static):       {np.mean(s_accs):.4f}")
    print("-" * 45)
    print("[6/6] Simulation complete. Use this for your 'Drift Resilience' section.")

if __name__ == "__main__":
    run_simulation()
