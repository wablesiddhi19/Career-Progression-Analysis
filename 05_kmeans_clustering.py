
# =====================================================
# PROJECT: Career Progression and Promotion Gap Analysis
# STEP 5: K-MEANS CLUSTERING
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load preprocessed dataset
X = pd.read_csv("outputs/preprocessed_clustering_data.csv")

print("Preprocessed Dataset Shape:", X.shape)

# 2. Test different numbers of clusters
inertia_values = []
silhouette_values = []
k_values = range(2, 9)

print("\n========== CLUSTER EVALUATION ==========")

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    inertia_values.append(kmeans.inertia_)

    score = silhouette_score(X, labels)
    silhouette_values.append(score)

    print(f"K = {k}, Silhouette Score = {score:.4f}")

# 3. Plot Elbow Method
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.savefig("outputs/05_elbow_method.png")
plt.show()

# 4. Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)
plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.savefig("outputs/06_silhouette_scores.png")
plt.show()

# 5. Select K with the highest silhouette score
best_k = list(k_values)[silhouette_values.index(
    max(silhouette_values)
)]

print("\nBest K based on Silhouette Score:", best_k)

# 6. Train final K-Means model
final_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_kmeans.fit_predict(X)

# 7. Save cluster assignments
results = pd.DataFrame({
    "Employee_Index": range(len(X)),
    "Career_Cluster": cluster_labels
})

# Add Attrition only for later analysis, not clustering
original_df = pd.read_csv("dataset/Palo Alto Networks.csv")
results["Attrition"] = original_df["Attrition"].values

results.to_csv(
    "outputs/kmeans_cluster_results.csv",
    index=False
)

# 8. Display cluster sizes
print("\n========== CLUSTER DISTRIBUTION ==========")
print(results["Career_Cluster"].value_counts().sort_index())

print("\nCluster Results Shape:", results.shape)

print("\nK-Means Clustering Completed Successfully!")
print("Results saved in outputs/kmeans_cluster_results.csv")
