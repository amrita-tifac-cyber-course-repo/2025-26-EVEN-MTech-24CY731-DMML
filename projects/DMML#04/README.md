# DMML
# Log Anomaly Detection Using SLM 🛡️

## 📌 Project Overview
Modern enterprise systems generate millions of logs daily, making manual inspection impossible. While Large Language Models (LLMs) are powerful at understanding text, they are often too slow and resource-intensive for real-time security monitoring in a SOC (Security Operations Center).

This project, **Log Anomaly Detection Using SLM**, solves the **latency-accuracy tradeoff** by deploying a low-parameter **Small Language Model**. Our pipeline achieves deep semantic intelligence—the kind usually reserved for massive AI—but runs with the rapid execution speeds required for real-time threat hunting and system health monitoring.

### Key Technical Features:
* **Semantic Vectorization**: Uses the `all-MiniLM-L6-v2` transformer to turn raw logs into 384-dimensional math vectors.
* **Unsupervised Clustering**: Automatically groups logs into 5 distinct "normative" behavior states using K-Means.
* **Ensemble Anomaly Detection**: Runs three parallel algorithms—**Isolation Forest**, **One-Class SVM**, and **Local Outlier Factor (LOF)**—to find outliers.
* **Consensus Voting**: Implements a strict majority-rule filter to ensure a log is only flagged if at least two models agree, drastically reducing false alarms.

---

## 🏗️ System Architecture
The pipeline follows a professional Machine Learning workflow:
1. **Data Preprocessing**: Merges timestamp, component, and log content to preserve the full operational context.
2. **AI Vectorization**: Transforms text into dense "embeddings" that represent the meaning of the log.
3. **Cluster Optimization**: Uses the Davies-Bouldin index to mathematically determine the best number of system states.
4. **Multi-Model Filtering**: Passes data through three independent detectors simultaneously.
5. **Consensus Verdict**: Finalizes the anomaly list based on cross-model agreement.

---

## 📊 Performance Results
The system was tested on 2,000 real-world production logs from the **Loghub Windows dataset**:
* **Clustering Success**: Identified 5 distinct operational groups with a high separation score of **0.56**.
* **Detection Accuracy**: Successfully isolated **104 high-confidence anomalies** (5.2% of the total logs).
* **Efficiency**: The model footprint is only **~80MB**, making it light enough to run on edge devices or directly within a SIEM dashboard.

---

## 📁 Repository Structure
* **/src**: Contains the Python implementation (`main.py`), logic for consensus voting, and `requirements.txt`.
* **/images**: Visual proof of the project, including the architecture flowchart and UMAP cluster plots.
* **/dataset**: Documentation and direct links to the Loghub Windows and Linux datasets.

---

## 🏁 Conclusion
This project demonstrates that you don't need a massive, power-hungry LLM to understand system logs. By using a Small Language Model (SLM) and an ensemble of traditional algorithms, we can catch complex system failures and security incidents with very low computational overhead.

**Author**: Sonu George  
**Specialization**: M.Tech Cyber Security | Amrita Vishwa Vidyapeetham
