
# Step 07: Hierarchical Clustering

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs" / "preprocessed_clustering_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# 2. Load preprocessed dataset
print("Loading preprocessed dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# 3. Prepare data for clustering
X = df.select_dtypes(include="number")

if X.shape[1] == 0:
    raise ValueError("No numerical features found in the dataset.")

print("\nFeatures used for clustering:", X.shape[1])

# 4. Generate linkage matrix
print("\nGenerating hierarchical linkage...")

linkage_matrix = linkage(
    X,
    method="ward",
    metric="euclidean"
)

# 5. Visualize dendrogram
print("\nCreating dendrogram...")

plt.figure(figsize=(14, 7))

dendrogram(
    linkage_matrix,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Employee Groups")
plt.ylabel("Euclidean Distance")
plt.tight_layout()

dendrogram_path = OUTPUT_DIR / "hierarchical_dendrogram.png"
plt.savefig(dendrogram_path, dpi=300)
plt.show()

# 6. Apply Agglomerative Clustering
print("\nApplying Agglomerative Clustering...")

model = AgglomerativeClustering(
    n_clusters=4,
    metric="euclidean",
    linkage="ward"
)

cluster_labels = model.fit_predict(X)

# 7. Add cluster labels
results = df.copy()
results["Hierarchical_Cluster"] = cluster_labels

# 8. Evaluate clustering
score = silhouette_score(X, cluster_labels)

print("\nHierarchical Clustering Results")
print("--------------------------------")
print("Number of clusters:", len(set(cluster_labels)))
print("Silhouette Score:", round(score, 4))

print("\nEmployee count in each cluster:")
print(results["Hierarchical_Cluster"].value_counts().sort_index())

# 9. Save clustering results
results_path = OUTPUT_DIR / "hierarchical_cluster_results.csv"

results.to_csv(results_path, index=False)

print("\nFiles saved successfully!")
print("Results:", results_path)
print("Dendrogram:", dendrogram_path)

print("\nStep 07 completed successfully!")
