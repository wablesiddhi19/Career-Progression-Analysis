
# =====================================================
# PROJECT: Career Progression and Promotion Gap Analysis
# STEP 6: CLUSTER PROFILING AND INTERPRETATION
# =====================================================

import pandas as pd
import os

# 1. Load original dataset
df = pd.read_csv("dataset/Palo Alto Networks.csv")

# 2. Load K-Means cluster results
cluster_results = pd.read_csv(
    "outputs/kmeans_cluster_results.csv"
)

# 3. Add cluster labels to original dataset
df["Career_Cluster"] = cluster_results["Career_Cluster"]

# 4. Select important career-related features
profile_features = [
    "Age",
    "JobLevel",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear",
    "YearsWithCurrManager",
    "Promotion_Gap_Ratio",
    "Role_Stagnation_Index",
    "Training_Intensity",
    "Manager_Stability_Indicator"
]

# Load engineered features to include calculated ratios
engineered_df = pd.read_csv(
    "outputs/career_progression_features.csv"
)

df["Promotion_Gap_Ratio"] = engineered_df["Promotion_Gap_Ratio"]
df["Role_Stagnation_Index"] = engineered_df["Role_Stagnation_Index"]
df["Training_Intensity"] = engineered_df["Training_Intensity"]
df["Manager_Stability_Indicator"] = engineered_df[
    "Manager_Stability_Indicator"
]

# 5. Calculate average career characteristics
cluster_profile = df.groupby("Career_Cluster")[
    profile_features
].mean().round(2)

print("\n========== CAREER CLUSTER PROFILE ==========")
print(cluster_profile)

# 6. Calculate number of employees in each cluster
cluster_size = df["Career_Cluster"].value_counts().sort_index()

print("\n========== CLUSTER SIZE ==========")
print(cluster_size)

# 7. Calculate attrition rate by cluster
attrition_rate = (
    df.groupby("Career_Cluster")["Attrition"]
    .mean()
    .mul(100)
    .round(2)
)

print("\n========== ATTRITION RATE BY CLUSTER (%) ==========")
print(attrition_rate)

# 8. Find most common department in each cluster
print("\n========== MOST COMMON DEPARTMENT ==========")

for cluster in sorted(df["Career_Cluster"].unique()):
    department = (
        df[df["Career_Cluster"] == cluster]["Department"]
        .mode()
        .iloc[0]
    )
    print(f"Cluster {cluster}: {department}")

# 9. Save cluster profile
os.makedirs("outputs", exist_ok=True)

cluster_profile.to_csv(
    "outputs/cluster_profile.csv"
)

# 10. Save detailed employee data with cluster labels
df.to_csv(
    "outputs/employee_clustered_data.csv",
    index=False
)

print("\nCluster profiling completed successfully!")
print("Files saved:")
print("1. outputs/cluster_profile.csv")
print("2. outputs/employee_clustered_data.csv")
