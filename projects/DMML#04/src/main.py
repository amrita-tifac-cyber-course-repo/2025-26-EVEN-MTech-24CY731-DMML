import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity

# Configuration parameters
CONFIG = {
    "model_name": "all-MiniLM-L6-v2", # SLM
    "batch_size": 32,                 # Increased for faster processing
    "cluster_range": [3, 5, 7, 9],
    "anomaly_contamination": 0.05,
    "num_representative_logs": 3
}

# 1. Data Preparation (Mapped to your specific Log Format)
print("\n1. Loading and preprocessing data...")

# Load and immediately reset index to prevent the "duplicate labels" error
df = pd.read_csv("/content/Windows_2k.log_structured.csv")
df = df.loc[:, ~df.columns.duplicated()].copy() # Remove any duplicate columns
df = df.reset_index(drop=True)

print(f"Initial dataset size: {len(df):,} records")
original_size = len(df)

# Logic to handle different header names (DateTime vs Date/Time)
if "DateTime" in df.columns:
    # Use the combined DateTime column from your log format
    time_content = df["DateTime"].astype(str)
elif "Date" in df.columns and "Time" in df.columns:
    time_content = df["Date"].astype(str) + " " + df["Time"].astype(str)
else:
    # Fallback if headers are totally different
    time_content = ""
    print("Warning: Could not find standard Date/Time columns.")

# Map 'Content' or 'EventTemplate' to the text body
# Your format shows 'Content', but we'll check for both
body_col = "Content" if "Content" in df.columns else "EventTemplate"

# Create the text column using the headers from your specific format
# We use .values to avoid index alignment issues (the "reindex" error)
try:
    df["text"] = (
        time_content.values + " " +
        df["Level"].astype(str).values + " " +
        df["Component"].astype(str).values + " " +
        df[body_col].astype(str).values
    )

    # Clean up whitespace
    df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df.dropna(subset=["text"])

    print(f"Final dataset size: {len(df):,} records")
    print("Sample processed text:", df["text"].iloc[0][:100], "...")

except KeyError as e:
    print(f"Error: Missing expected column: {e}")
    print(f"Available columns in your file are: {df.columns.tolist()}")

# 2 & 3. Model Loading and Embedding Generation
print(f"\n2. Loading SLM ({CONFIG['model_name']}) and generating embeddings...")
# SentenceTransformer automatically handles tokenization and mean pooling
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(CONFIG["model_name"], device=device)

# Generate embeddings
log_embeddings = model.encode(
    df["text"].tolist(),
    batch_size=CONFIG["batch_size"],
    show_progress_bar=True,
    normalize_embeddings=True
)
print(f"Embedding generation complete. Final shape: {log_embeddings.shape}")

# 4. Optimized Clustering
print("\n4. Determining optimal cluster count...")
cluster_results = []
for n in CONFIG["cluster_range"]:
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10).fit(log_embeddings)
    score = davies_bouldin_score(log_embeddings, kmeans.labels_)
    cluster_results.append((n, score))
    print(f"Clusters: {n} \t Davies-Bouldin: {score:.2f} \t Inertia: {kmeans.inertia_:,.0f}")

best_n = min(cluster_results, key=lambda x: x[1])[0]
print(f"\nOptimal cluster count: {best_n} (Lowest Davies-Bouldin score)")
kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=10).fit(log_embeddings)
df['cluster'] = kmeans.labels_

# 5. UMAP Visualization
print("\n5. Creating UMAP visualization...")
reducer = UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(log_embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=df['cluster'], cmap='tab20', alpha=0.7,
                      s=10, edgecolor='none')
plt.title(f"UMAP Projection of MiniLM Log Embeddings")
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# 6. Anomaly Detection (Bug Fixed)
print("\n6. Running anomaly detection...")
anomaly_methods = {
    "IsolationForest": IsolationForest(contamination=CONFIG["anomaly_contamination"], random_state=42),
    "OneClassSVM": OneClassSVM(nu=CONFIG["anomaly_contamination"]),
    "LocalOutlierFactor": LocalOutlierFactor(novelty=True, contamination=CONFIG["anomaly_contamination"])
}

anomaly_cols = []
for name, ad_model in anomaly_methods.items():
    try:
        if name == "LocalOutlierFactor":
            # LOF with novelty=True requires fit() then predict()
            ad_model.fit(log_embeddings)
            preds = ad_model.predict(log_embeddings)
        else:
            preds = ad_model.fit_predict(log_embeddings)

        df[f'anomaly_{name}'] = np.where(preds == -1, 1, 0)
        print(f"{name:<20} Anomalies: {df[f'anomaly_{name}'].sum():>4} ({df[f'anomaly_{name}'].mean()*100:.1f}%)")
        anomaly_cols.append(f'anomaly_{name}')
    except Exception as e:
        print(f"Error with {name}: {str(e)}")

# Consensus anomalies
df['anomaly_consensus'] = df[anomaly_cols].sum(axis=1)
df['anomaly_consensus'] = np.where(df['anomaly_consensus'] >= 2, 1, 0)
print(f"\nConsensus anomalies (>=2 methods agree): {df['anomaly_consensus'].sum()} ({df['anomaly_consensus'].mean()*100:.1f}%)")

# 7. Final Results Saving
print("\n7. Saving results...")
df.to_csv("minilm_log_analysis.csv", index=False)
print("Analysis complete! Check minilm_log_analysis.csv for the tagged logs.")