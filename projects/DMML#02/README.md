# 025-26-EVEN-MTech-24CY731-DMML

## 2025-26 EVEN MTech 24CY731 DMML

## AI & ML/DL Applications in Cybersecurity

## 1. Project Title

AI-Based Insider Threat Detection and Real-Time SOC Dashboard

## 2. Team Details

Team Number: 2

Members:
- CB.SC.P2CYS25005 - atara rushi vimalbhai


## 3. Problem Statement

This project addresses the problem of insider threat detection by analyzing enterprise user activity logs and identifying risky behavioral deviations. The system detects suspicious patterns such as unusual login times, spikes in file-copy events, and abrupt changes in email behavior, then provides real-time risk visualization through a SOC dashboard.

## 4. Objectives

- Build a user-level daily cybersecurity behavior dataset from raw logs.
- Detect anomalous and potentially malicious insider behavior using ML models.
- Compare multiple approaches (Random Forest, Isolation Forest, anomaly simulation, and stacking).
- Deploy a real-time Streamlit SOC dashboard for monitoring, alerting, and user-wise tracking.

## 5. Dataset Details

Dataset Name:
- CERT Insider Threat Dataset (Release 4.x style structure)

Source (Link):
- CERT Insider Threat Center: https://www.sei.cmu.edu/our-work/cert-insider-threat-center/
- CERT data overview page: https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099

Features Description:
- `logon_count`: total logon/logoff activity count per user per day
- `after_hours_logon`: number of user logons outside normal office hours
- `file_count`: total file-copy events to removable media per user per day
- `email_count`: total outgoing email activity count per user per day

Dataset Size:
- Depends on the downloaded CERT release and selected files.
- This project currently uses these input files:
  - `dataset/logon.csv`
  - `dataset/file.csv`
  - `dataset/email.csv`

## 6. Methodology

Techniques Used (ML/DL algorithms):
- Random Forest Classifier (supervised classification)
- Isolation Forest (unsupervised anomaly detection)
- Autoencoder-style anomaly proxy (simulated reconstruction deviation)
- Score-level Stacking/Ensemble

Model Architecture (for DL):
- No deep learning architecture is currently implemented in this codebase.
- A simulated autoencoder-style anomaly score is used for comparison.

Workflow:
1. Data Collection
   - Load `logon.csv`, `file.csv`, and `email.csv`.
2. Data Preprocessing
   - Convert date columns, derive day/hour fields, and aggregate per user-day.
3. Feature Selection/Engineering
   - Build `logon_count`, `after_hours_logon`, `file_count`, `email_count`.
   - Generate z-scores only for pseudo-label creation (to avoid leakage in model features).
4. Model Training
   - Train Isolation Forest and Random Forest on train split.
   - Compute autoencoder-style anomaly proxy scores.
5. Model Evaluation
   - Compare models using ROC-AUC, PR-AUC, and F1 score.

## 7. Tools & Technologies

Programming Language:
- Python

Libraries:
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit

Optional/Not currently used in this implementation:
- TensorFlow
- PyTorch
- Matplotlib

## 8. How to Run the Project

### A) Clone the Repository

```bash
git clone <repo-link>
cd <project-folder>
```

Or, for local setup in this machine:

```bash
cd c:\project\dmml_project
```

### B) Install dependencies

```bash
pip install pandas numpy scikit-learn joblib streamlit
```

### C) Train model and generate artifacts

```bash
python final_code.py
```

Expected generated files:
- `full_pipeline.pkl`
- `ui_input.csv`

### D) Run dashboard

```bash
streamlit run app.py
```

In the dashboard:
- Upload `ui_input.csv`
- Configure alert threshold and live mode

## Additional Notes

- Local documentation for dataset fields is available at `dataset/readme.txt`.
- License details are available at `dataset/license.txt`.
- If app shows "Model not found", run `python final_code.py` first.
