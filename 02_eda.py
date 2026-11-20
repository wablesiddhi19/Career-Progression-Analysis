
# =====================================================
# PROJECT: Career Progression and Promotion Gap Analysis
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("dataset/Palo Alto Networks.csv")

# Create folder to save charts
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Attrition Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition (0 = Stayed, 1 = Left)")
plt.ylabel("Number of Employees")
plt.savefig("outputs/01_attrition_distribution.png")
plt.show()

# 2. Years at Company Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="YearsAtCompany", bins=20, kde=True)
plt.title("Distribution of Years at Company")
plt.xlabel("Years at Company")
plt.ylabel("Number of Employees")
plt.savefig("outputs/02_years_at_company.png")
plt.show()

# 3. Years Since Last Promotion
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="YearsSinceLastPromotion", bins=15, kde=True)
plt.title("Distribution of Years Since Last Promotion")
plt.xlabel("Years Since Last Promotion")
plt.ylabel("Number of Employees")
plt.savefig("outputs/03_promotion_gap.png")
plt.show()

# 4. Average Promotion Gap by Department
department_data = df.groupby("Department")[
    "YearsSinceLastPromotion"
].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(
    data=department_data,
    x="Department",
    y="YearsSinceLastPromotion"
)
plt.title("Average Years Since Last Promotion by Department")
plt.xlabel("Department")
plt.ylabel("Average Years Since Last Promotion")
plt.xticks(rotation=15)
plt.savefig("outputs/04_department_promotion_gap.png")
plt.show()

# 5. Important Statistical Information
print("\n========== CAREER PROGRESSION SUMMARY ==========")

print("\nAverage Years at Company:")
print(df["YearsAtCompany"].mean())

print("\nAverage Years Since Last Promotion:")
print(df["YearsSinceLastPromotion"].mean())

print("\nAverage Years in Current Role:")
print(df["YearsInCurrentRole"].mean())

print("\nAverage Promotion Gap by Department:")
print(department_data)

print("\n========== EDA COMPLETED ==========")
