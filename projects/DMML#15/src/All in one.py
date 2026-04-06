import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from river import drift
import warnings
import random

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
BENIGN_PATH = "C:/Users/adity/Desktop/MTECH_2ND_SEM/DMML/Mini Project/Dataset/Total_CSVs/l2-benign.csv"
MALICIOUS_PATH = "C:/Users/adity/Desktop/MTECH_2ND_SEM/DMML/Mini Project/Dataset/Total_CSVs/l2-malicious.csv"

def load_and_multi_drift(b_path, m_path):
    print("[1/6] Loading Data and Creating Unified Stream...")
    try:
        df_b = pd.read_csv(b_path)
        df_m = pd.read_csv(m_path)
    except FileNotFoundError:
        print("Error: CSV files not found.")
        return None, None, None, None, None

    drop_cols = ['SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort', 'TimeStamp', 'Label']
    X_b = df_b.drop(columns=[col for col in drop_cols if col in df_b.columns]).fillna(0)
    X_m = df_m.drop(columns=[col for col in drop_cols if col in df_m.columns]).fillna(0)
    
    # Randomize Phase Lengths
    lengths = [random.randint(600, 1200) for _ in range(5)]
    X_train = X_b.iloc[:800]
    y_train = np.zeros(800) 
    
    b_idx, m_idx = 800, 0
    segments_X, segments_y, actual_drift_points = [], [], []
    current_total = 0

    for i, length in enumerate(lengths):
        if i % 2 == 0:
            segments_X.append(X_b.iloc[b_idx : b_idx + length])
            segments_y.append(np.zeros(length))
            b_idx += length
        else:
            segments_X.append(X_m.iloc[m_idx : m_idx + length])
            segments_y.append(np.ones(length))
            m_idx += length
        if i > 0: actual_drift_points.append(current_total)
        current_total += length

    return X_train, y_train, pd.concat(segments_X).reset_index(drop=True), np.concatenate(segments_y), actual_drift_points

def run_comparison():
    X_train, y_train, X_stream, y_stream, real_drifts = load_and_multi_drift(BENIGN_PATH, MALICIOUS_PATH)
    if X_train is None: return

    # Initialize 1 Static and 3 Adaptive Models
    models = {
        'Static': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        'ADWIN': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        'KSWIN': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        'PageHinkley': RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    }
    
    # Train all initially
    for name in models: models[name].fit(X_train, y_train)

    # Initialize Detectors
    detectors = {
        'ADWIN': drift.ADWIN(delta=0.002),
        'KSWIN': drift.KSWIN(alpha=0.001, window_size=100),
        'PageHinkley': drift.PageHinkley(delta=0.005, threshold=35)
    }

    # Tracking Results
    results = {name: [] for name in models}
    detections = {name: [] for name in detectors}
    indices = []
    batch_size = 50

    print("[2/6] Running Competitive Simulation...")
    for i in range(0, len(X_stream), batch_size):
        X_batch = X_stream.iloc[i : i + batch_size]
        y_batch = y_stream[i : i + batch_size]
        if len(X_batch) < batch_size: break
        indices.append(i)

        for name, model in models.items():
            preds = model.predict(X_batch)
            acc = accuracy_score(y_batch, preds)
            results[name].append(acc)

            # Only Update Adaptive Detectors
            if name in detectors:
                for p, a in zip(preds, y_batch):
                    err = 1 if p != a else 0
                    detectors[name].update(err)
                    if detectors[name].drift_detected:
                        detections[name].append(i)
                        models[name].fit(X_batch, y_batch) # Adapt
                        # Reset detector
                        if name == 'ADWIN': detectors[name] = drift.ADWIN(delta=0.002)
                        elif name == 'KSWIN': detectors[name] = drift.KSWIN(alpha=0.001)
                        else: detectors[name] = drift.PageHinkley(delta=0.005, threshold=35)
                        break

    # --- VISUALIZATION ---
    print("[3/6] Plotting Comparison Results...")
    plt.figure(figsize=(16, 9))
    
    # Plot Accuracies
    colors = {'Static': 'red', 'ADWIN': 'blue', 'KSWIN': 'purple', 'PageHinkley': 'green'}
    for name, acc_list in results.items():
        alpha_val = 0.3 if name == 'Static' else 0.8
        plt.plot(indices, acc_list, label=f'{name} Accuracy', color=colors[name], alpha=alpha_val, linewidth=2)

    # Plot Detections
    offset = 0.05
    for name, det_list in detections.items():
        plt.scatter(det_list, [1.05 + offset]*len(det_list), color=colors[name], marker='v', label=f'{name} Detection')
        offset += 0.05

    # Real Drift Lines
    for rd in real_drifts:
        plt.axvline(x=rd, color='black', linestyle='--', alpha=0.4)

    plt.title('Comparison of Concept Drift Detectors on DNS Tunneling Traffic')
    plt.ylabel('Accuracy / Detection Events')
    plt.xlabel('Traffic Index')
    plt.legend(loc='lower left', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n[Final Summary]")
    print(f"{'Algorithm':<15} | {'Avg Acc':<8} | {'Detections'}")
    print("-" * 40)
    for name in models:
        print(f"{name:<15} | {np.mean(results[name]):.4f}   | {len(detections.get(name, []))}")

if __name__ == "__main__":
    run_comparison()
