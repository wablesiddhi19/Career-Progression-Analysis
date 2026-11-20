# Career Progression and Promotion Gap Analysis for Retention Optimization at Palo Alto Networks

## Project Overview

This project analyzes employee career progression, promotion gaps, and attrition patterns to identify employee segments that may need retention-focused attention. It uses exploratory data analysis, feature engineering, clustering, Excel analysis, and Tableau dashboards.

**Project tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Microsoft Excel, Tableau Public, and GitHub.

> **Data note:** The analysis uses the employee dataset supplied for this project. The dataset fields and analysis do not independently verify that the records represent actual Palo Alto Networks employees. The project title is retained as assigned.

## Objectives

- Understand employee attrition and career progression patterns.
- Examine promotion waiting periods and promotion-gap indicators.
- Engineer features that describe career stagnation and manager stability.
- Segment employees using K-Means and Hierarchical Clustering.
- Compare clustering models and summarize retention-related findings.
- Present key results through Excel and Tableau dashboards.

## Dataset

- **Records:** 1,470 employees
- **Original fields:** 31
- **Prepared analysis fields:** 36
- **Missing values:** None reported in the initial data check
- **Attrition:** 237 employees marked as attrition = 1 and 1,233 as attrition = 0
- **Overall attrition rate:** Approximately 16.12%

The analysis uses fields including Department, JobRole, MonthlyIncome, OverTime, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager, and Attrition.

## Methodology

1. **Data understanding and exploratory analysis**
   - Inspected dataset dimensions, columns, data types, and missing values.
   - Examined attrition distribution and career progression measures.
   - Compared average promotion waiting period across departments.

2. **Feature engineering**
   - Created `Zero_Tenure_Flag`.
   - Created `Promotion_Gap_Ratio`.
   - Created `Role_Stagnation_Index`.
   - Created `Training_Intensity`.
   - Created `Manager_Stability_Indicator`.

3. **Preprocessing**
   - Selected clustering features.
   - Applied preprocessing and outlier treatment.
   - Prepared the employee-level clustering dataset.

4. **Clustering**
   - Applied K-Means clustering and evaluated multiple values of K.
   - Applied Hierarchical Clustering.
   - Compared the two approaches using Silhouette Score and Adjusted Rand Index.

5. **Visualization and reporting**
   - Built Excel PivotTable analysis and dashboard charts.
   - Created Tableau worksheets and two dashboards for attrition, career clusters, and promotion gaps.

## Key Findings

### Overall employee and attrition profile

- The dataset contains 1,470 employee records.
- 237 employees have `Attrition = 1`, giving an overall attrition rate of approximately **16.12%**.
- Average `YearsAtCompany`: approximately **7.01 years**.
- Average `YearsSinceLastPromotion`: approximately **2.19 years**.
- Average `YearsInCurrentRole`: approximately **4.23 years**.

### Department-level promotion waiting period

Average `YearsSinceLastPromotion` by department:

| Department | Average years since last promotion |
|---|---:|
| Human Resources | 1.78 |
| Research & Development | 2.14 |
| Sales | 2.35 |

Sales has the highest average waiting period among the three departments in this dataset.

### K-Means clustering

The selected K-Means solution used **4 clusters**, with a Silhouette Score of **0.1845**.

| Cluster | Employees | Attrition rate |
|---:|---:|---:|
| 0 | 169 | 34.32% |
| 1 | 432 | 10.42% |
| 2 | 44 | 36.36% |
| 3 | 825 | 14.30% |

Cluster 2 has the highest observed attrition rate, but it contains only 44 employees. Cluster 0 also has a comparatively high attrition rate. These segments may warrant further investigation; rates should be interpreted alongside cluster size.

### Model comparison

| Model | Number of clusters | Silhouette Score |
|---|---:|---:|
| K-Means | 4 | 0.1845 |
| Hierarchical Clustering | 4 | 0.1732 |

- Adjusted Rand Index between the two cluster assignments: **0.7297**.
- K-Means was selected as the primary clustering result because it achieved the higher Silhouette Score in this comparison.
- The silhouette values indicate modest cluster separation. The clusters should therefore be treated as exploratory employee segments, not definitive employee categories.
- Cluster labels from different algorithms are arbitrary and should not be interpreted as matching categories.

### Tableau observations

- Attrition rate for employees without overtime: approximately **10.4%**.
- Attrition rate for employees with overtime: approximately **30.5%**.
- The overtime group therefore shows a substantially higher observed attrition rate in this dataset.
- Tableau dashboards also compare attrition by career cluster and promotion-gap ratio by career cluster.

## Project Files

The project folder contains Python analysis scripts, processed data, model outputs, visualizations, Excel analysis, and Tableau workbooks. Important generated outputs include:

- `employee_clustered_data.csv`
- `career_progression_features.csv`
- `cluster_profile.csv`
- `kmeans_cluster_results.csv`
- `hierarchical_cluster_results.csv`
- `08_model_comparison_metrics.csv`
- `08_cluster_cross_tabulation.csv`
- `08_employee_cluster_comparison.csv`
- EDA and clustering charts in the `outputs` folder
- Excel analysis workbook
- Tableau packaged workbook: `Career_Progression_Analysis.twbx`

Keep the actual folder and filenames in the repository consistent with the files you upload. Do not upload duplicate temporary files or personal/system files.

## How to Run the Python Analysis

1. Install Python and the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
2. Place the source dataset in the project data location expected by the scripts.
3. Run the scripts in sequence from the project root using the Python launcher on Windows:
   ```bash
   py 01_data_understanding.py
   py 02_eda.py
   py 03_feature_engineering.py
   py 04_data_preprocessing.py
   py 05_kmeans_clustering.py
   py 06_cluster_profiling.py
   py 07_hierarchical_clustering.py
   py 08_model_comparison.py
   ```
4. Open the Excel workbook to review the PivotTables and dashboard.
5. Open `Career_Progression_Analysis.twbx` in Tableau Public/Desktop to review the dashboards.

The scripts may require the dataset to be placed at the file path configured in the code. Update that path if the project is moved to another computer.

## Limitations

- Clustering is exploratory; the Silhouette Scores are relatively low and indicate limited separation between segments.
- A high attrition percentage in a small cluster should be interpreted with its employee count.
- The observed associations are descriptive and do not establish that overtime or promotion delays cause attrition.
- Dataset provenance and permission to redistribute the raw employee records should be checked before publishing them publicly. If redistribution is not permitted or is unclear, exclude the raw dataset and document how an authorized copy can be obtained.

## Conclusion

The analysis identifies differences in attrition and promotion-gap patterns across departments, overtime groups, and employee clusters. The K-Means results provide exploratory employee segments for further review. The findings can help frame retention questions and guide deeper investigation into career progression and employee experience.

---
**Project:** Career Progression and Promotion Gap Analysis for Retention Optimization at Palo Alto Networks  
**Project type:** Internship project
