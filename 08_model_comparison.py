
# Step 08: K-Means vs Hierarchical Clustering Comparison

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 1. Set project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# 2. Load datasets
print("Loading clustering results...")

X = pd.read_csv(OUTPUT_DIR / "preprocessed_clustering_data.csv")
kmeans = pd.read_csv(OUTPUT_DIR / "kmeans_cluster_results.csv")
hierarchical = pd.read_csv(
    OUTPUT_DIR / "hierarchical_cluster_results.csv"
)

# 3. Verify dataset alignment
if not (
    len(X) == len(kmeans) == len(hierarchical) == 1470
):
    raise ValueError("Dataset record counts do not match.")

if not np.array_equal(
    kmeans["Employee_Index"].to_numpy(),
    np.arange(len(X))
):
    raise ValueError("K-Means employee indexes are not aligned.")

if (
    kmeans["Career_Cluster"].isna().any()
    or hierarchical["Hierarchical_Cluster"].isna().any()
):
    raise ValueError("Missing cluster labels detected.")

# Hierarchical labels follow the original preprocessed row order.
# Both clustering scripts use the same preprocessed dataset.
comparison = pd.DataFrame({
    "Employee_Index": kmeans["Employee_Index"],
    "KMeans_Cluster": kmeans["Career_Cluster"],
    "Hierarchical_Cluster": hierarchical["Hierarchical_Cluster"]
})

print("Employee alignment verified!")
print("Total employees:", len(comparison))

# 4. Calculate silhouette scores
print("\nCalculating evaluation metrics...")

kmeans_score = silhouette_score(
    X, comparison["KMeans_Cluster"]
)

hierarchical_score = silhouette_score(
    X, comparison["Hierarchical_Cluster"]
)

# 5. Calculate Adjusted Rand Index
ari_score = adjusted_rand_score(
    comparison["KMeans_Cluster"],
    comparison["Hierarchical_Cluster"]
)

# 6. Create model comparison table
metrics = pd.DataFrame({
    "Model": [
        "K-Means",
        "Hierarchical Clustering"
    ],
    "Number_of_Clusters": [
        comparison["KMeans_Cluster"].nunique(),
        comparison["Hierarchical_Cluster"].nunique()
    ],
    "Silhouette_Score": [
        round(kmeans_score, 4),
        round(hierarchical_score, 4)
    ],
    "Total_Employees": [
        len(comparison),
        len(comparison)
    ]
})

# 7. Create cluster distribution
kmeans_distribution = (
    comparison["KMeans_Cluster"].value_counts().sort_index()
)

hierarchical_distribution = (
    comparison["Hierarchical_Cluster"].value_counts().sort_index()
)

# 8. Create cross-tabulation
cross_tab = pd.crosstab(
    comparison["KMeans_Cluster"],
    comparison["Hierarchical_Cluster"],
    margins=True
)

# 9. Save comparison results
metrics.to_csv(
    OUTPUT_DIR / "08_model_comparison_metrics.csv",
    index=False
)

cross_tab.to_csv(
    OUTPUT_DIR / "08_cluster_cross_tabulation.csv"
)

comparison.to_csv(
    OUTPUT_DIR / "08_employee_cluster_comparison.csv",
    index=False
)

# 10. Display results
print("\nMODEL COMPARISON")
print("-----------------------------")
print(metrics.to_string(index=False))

print("\nADJUSTED RAND INDEX:", round(ari_score, 4))

print("\nK-MEANS CLUSTER DISTRIBUTION")
print(kmeans_distribution)

print("\nHIERARCHICAL CLUSTER DISTRIBUTION")
print(hierarchical_distribution)

print("\nCLUSTER CROSS-TABULATION")
print(cross_tab)

print("\nAll comparison files saved successfully!")
print("\nStep 08 completed successfully!")
