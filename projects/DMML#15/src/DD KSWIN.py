import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from river import drift  # Using drift.KSWIN
import warnings
import random

# Suppress warnings for a clean output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BENIGN_PATH = "C:/Users/adity/Desktop/MTECH_2ND_SEM/DMML/Mini Project/Dataset/Total_CSVs/l2-benign.csv"
MALICIOUS_PATH = "C:/Users/adity/Desktop/MTECH_2ND_SEM/DMML/Mini Project/Dataset/Total_CSVs/l2-malicious.csv"

def load_and_multi_drift(b_path, m_path):
    print("[1/6] Loading Data and Randomizing Drift Points...")
    try:
        df_b = pd.read_csv(b_path)
        df_m = pd.read_csv(m_path)
    except FileNotFoundError:
        print("Error: CSV files not found. Please check your filenames.")
        return None, None, None, None, None

    # Feature Cleaning
    drop_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'Label']
    X_b = df_b.drop(columns=[col for col in drop_cols if col in df_b.columns]).fillna(0)
    X_m = df_m.drop(columns=[col for col in drop_cols if col in df_m.columns]).fillna(0)
    
    # Randomized Multi-Drift Construction
    lengths = [random.randint(500, 1500) for _ in range(5)]
    X_train = X_b.iloc[:800]
    y_train = np.zeros(800) 
    
    b_idx, m_idx = 800, 0
    segments_X, segments_y = [], []
    actual_drift_points = []
    current_total = 0

    for i, length in enumerate(lengths):
        if i % 2 == 0: # Benign Phases
            segments_X.append(X_b.iloc[b_idx : b_idx + length])
            segments_y.append(np.zeros(length))
            b_idx += length
        else: # Malicious Phases
            segments_X.append(X_m.iloc[m_idx : m_idx + length])
            segments_y.append(np.ones(length))
            m_idx += length
        
        if i > 0:
            actual_drift_points.append(current_total)
        current_total += length

    X_stream = pd.concat(segments_X).reset_index(drop=True)
    y_stream = np.concatenate(segments_y)
    
    return X_train, y_train, X_stream, y_stream, actual_drift_points

def run_simulation():
    X_train, y_train, X_stream, y_stream, real_drifts = load_and_multi_drift(BENIGN_PATH, MALICIOUS_PATH)
    if X_train is None: return
    
    print(f"Total Stream Length: {len(X_stream)} with {len(real_drifts)} random drift points.")
    
    print("[2/6] Training Initial Models...")
    static_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    adaptive_model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    
    static_model.fit(X_train, y_train)
    adaptive_model.fit(X_train, y_train)
    
    # --- KSWIN DETECTION SETUP ---
    # alpha: Probability threshold (default 0.005). Lower = more conservative.
    # window_size: Total window to monitor.
    # stat_size: The size of the "recent" sample to compare against the window.
    kswin_detector = drift.KSWIN(alpha=0.001, window_size=100, stat_size=30) 
    
    batch_size = 50 
    detected_indices, a_accs, s_accs, indices = [], [], [], []
    
    print("[3/6] Processing Stream with KSWIN Monitoring...")
    
    for i in range(0, len(X_stream), batch_size):
        X_batch = X_stream.iloc[i : i + batch_size]
        y_batch = y_stream[i : i + batch_size]
        if len(X_batch) < batch_size: break
        
        # Evaluation
        s_accs.append(accuracy_score(y_batch, static_model.predict(X_batch)))
        a_pred = adaptive_model.predict(X_batch)
        a_accs.append(accuracy_score(y_batch, a_pred))
        indices.append(i)
        
        # Drift Monitoring
        for pred, actual in zip(a_pred, y_batch):
            # KSWIN is usually fed the stream of errors (0 for correct, 1 for error)
            error = 1 if pred != actual else 0
            kswin_detector.update(error)
            
            if kswin_detector.drift_detected:
                print(f"--> ALERT: KSWIN Detected Statistical Drift near Index {i}!")
                detected_indices.append(i)
                
                # Adaptation: Retrain on the most recent context
                adaptive_model.fit(X_batch, y_batch) 
                # Reset KSWIN
                kswin_detector = drift.KSWIN(alpha=0.001, window_size=100, stat_size=30)
                break

    # --- VISUALIZATION ---
    print("[4/6] Generating Comparison Plot...")
    plt.figure(figsize=(15, 8))
    
    plt.plot(indices, s_accs, label='Static Model (Fixed)', color='red', alpha=0.3, linestyle='--')
    plt.plot(indices, a_accs, label='Adaptive Model (KSWIN)', color='purple', linewidth=2)
    
    for rd in real_drifts:
        plt.axvline(x=rd, color='gray', linestyle='-', alpha=0.4, label='Ground Truth Drift' if rd == real_drifts[0] else "")

    for dd in detected_indices:
        plt.axvline(x=dd, color='magenta', linestyle=':', alpha=0.9, label='KSWIN Detection' if dd == detected_indices[0] else "")
        
    plt.title('KSWIN Statistical Drift Detection on Randomized DNS Traffic')
    plt.xlabel('Traffic Sample Index')
    plt.ylabel('Model Accuracy')
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='lower left', ncol=2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n[5/6] Performance Analysis:")
    print("-" * 50)
    print(f"Real Drift Events:      {len(real_drifts)}")
    print(f"Detected Drift Events:  {len(detected_indices)}")
    print(f"Avg Accuracy (Adaptive): {np.mean(a_accs):.4f}")
    print(f"Avg Accuracy (Static):   {np.mean(s_accs):.4f}")
    print("-" * 50)
    print("[6/6] Simulation complete.")

if __name__ == "__main__":
    run_simulation()
