
# =====================================================
# PROJECT: Career Progression and Promotion Gap Analysis
# STEP 1: DATA UNDERSTANDING
# =====================================================

import pandas as pd
import numpy as np

# 1. Load the dataset
file_path = "dataset/Palo Alto Networks.csv"

df = pd.read_csv(file_path)

# 2. Display basic dataset information
print("\n========== DATASET OVERVIEW ==========")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

# 3. Display first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# 4. Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 5. Check data types and non-null values
print("\n========== DATASET INFORMATION ==========")
df.info()

# 6. Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 7. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 8. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate Rows:", df.duplicated().sum())

# 9. Check attrition distribution
if "Attrition" in df.columns:
    print("\n========== ATTRITION DISTRIBUTION ==========")
    print(df["Attrition"].value_counts())

# 10. Check dataset structure
print("\n========== DATA UNDERSTANDING COMPLETED ==========")
