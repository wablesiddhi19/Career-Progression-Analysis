
# =====================================================
# PROJECT: Career Progression and Promotion Gap Analysis
# STEP 3: FEATURE ENGINEERING
# =====================================================

import pandas as pd
import numpy as np
import os

# 1. Load the original dataset
df = pd.read_csv("dataset/Palo Alto Networks.csv")

print("Original Dataset Shape:", df.shape)

# 2. Identify employees with zero company tenure
df["Zero_Tenure_Flag"] = (df["YearsAtCompany"] == 0).astype(int)

# 3. Create Promotion Gap Ratio
df["Promotion_Gap_Ratio"] = np.where(
    df["YearsAtCompany"] > 0,
    df["YearsSinceLastPromotion"] / df["YearsAtCompany"],
    0
)

# 4. Create Role Stagnation Index
df["Role_Stagnation_Index"] = np.where(
    df["YearsAtCompany"] > 0,
    df["YearsInCurrentRole"] / df["YearsAtCompany"],
    0
)

# 5. Create Training Intensity
df["Training_Intensity"] = np.where(
    df["YearsAtCompany"] > 0,
    df["TrainingTimesLastYear"] / df["YearsAtCompany"],
    0
)

# 6. Create Manager Stability Indicator
df["Manager_Stability_Indicator"] = np.where(
    df["YearsAtCompany"] > 0,
    df["YearsWithCurrManager"] / df["YearsAtCompany"],
    0
)

# 7. Display newly created features
new_features = [
    "Zero_Tenure_Flag",
    "Promotion_Gap_Ratio",
    "Role_Stagnation_Index",
    "Training_Intensity",
    "Manager_Stability_Indicator"
]

print("\n========== NEW FEATURES ==========")
print(df[new_features].head(10))

# 8. Display statistical summary
print("\n========== FEATURE SUMMARY ==========")
print(df[new_features].describe())

# 9. Save the updated dataset
os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/career_progression_features.csv",
    index=False
)

print("\nOriginal Dataset Shape:", (1470, 31))
print("Updated Dataset Shape:", df.shape)
print("\nFeature Engineering Completed Successfully!")
print("Updated dataset saved in outputs folder.")
