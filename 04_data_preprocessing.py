
# =====================================================
# PROJECT: Career Progression and Promotion Gap Analysis
# STEP 4: DATA PREPROCESSING
# =====================================================

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Load feature-engineered dataset
df = pd.read_csv("outputs/career_progression_features.csv")

print("Original Dataset Shape:", df.shape)

# 2. Select numerical career-related features
numeric_features = [
    "Age",
    "JobLevel",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear",
    "PerformanceRating",
    "Promotion_Gap_Ratio",
    "Role_Stagnation_Index",
    "Training_Intensity",
    "Manager_Stability_Indicator",
    "Zero_Tenure_Flag"
]

# 3. Select categorical features
categorical_features = [
    "Department",
    "JobRole",
    "EducationField",
    "BusinessTravel",
    "OverTime"
]

# 4. Create a copy of selected data
X = df[numeric_features + categorical_features].copy()

# 5. Handle extreme values using IQR capping
# Apply only to selected continuous career variables
iqr_features = [
    "Age",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear"
]

for col in iqr_features:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    X[col] = X[col].clip(
        lower=lower_limit,
        upper=upper_limit
    )

print("\nOutlier Treatment Completed.")

# 6. Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

# 7. Transform the selected data
X_processed = preprocessor.fit_transform(X)

# 8. Get processed feature names
feature_names = preprocessor.get_feature_names_out()

# 9. Convert processed data into DataFrame
processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# 10. Save preprocessed dataset
os.makedirs("outputs", exist_ok=True)

processed_df.to_csv(
    "outputs/preprocessed_clustering_data.csv",
    index=False
)

# 11. Display results
print("\n========== PREPROCESSING SUMMARY ==========")

print("Original Selected Features:", X.shape[1])
print("Processed Features:", processed_df.shape[1])
print("Number of Records:", processed_df.shape[0])

print("\nFirst Five Processed Records:")
print(processed_df.head())

print("\nMissing Values:", processed_df.isnull().sum().sum())

print("\nPreprocessed Dataset Saved Successfully!")
print("File: outputs/preprocessed_clustering_data.csv")
