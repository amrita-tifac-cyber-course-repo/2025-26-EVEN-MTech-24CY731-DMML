# DMML
# Project Visualizations 🖼️

This directory contains the visual evidence and architectural diagrams for the SLM-Log-Anomaly pipeline.

## 1. System Architecture (`system_architecture.png`)
Provides a high-level overview of the data pipeline:
* **Preprocessing**: Log concatenation and cleaning.
* **Embedding**: Transformation via `all-MiniLM-L6-v2`.
* **Detection**: Parallel processing through the Ensemble Layer (Isolation Forest, One-Class SVM, and LOF).
* **Consensus**: Final anomaly filtering via majority voting.

## 2. UMAP Projection of Log Embeddings (`umap_clusters.png`)
This scatter plot visualizes the 384-dimensional latent space projected into 2D:
* **Clusters**: The 5 distinct colors represent normative system states identified by K-Means.
* **Separation**: A Davies-Bouldin score of 0.56 confirms that the SLM effectively grouped logs by semantic meaning.
* **Anomalies**: Anomalies typically appear as isolated points on the periphery of these dense clusters.
