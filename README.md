# Clustering Indian States and Union Territories Based on Forest Cover and Its Changes Using Machine Learning

## 1. Project Overview

This project applies unsupervised machine learning techniques to group Indian States and Union Territories according to their forest-cover composition and changes between 2019 and 2021.

The study uses forest-cover information published in the India State of Forest Report (ISFR) by the Forest Survey of India (FSI).

The project analyses three forest-cover categories:

- Very Dense Forest (VDF)
- Moderately Dense Forest (MDF)
- Open Forest (OF)

The objective is to identify groups of States/UTs with similar forest-cover characteristics, examine changes in forest density and identify regions experiencing declines in combined dense forest cover.

## 2. Problem Statement

India aims to increase its forest and tree cover. Understanding differences in forest density and changes across States and Union Territories is important for environmental monitoring and planning.

This project uses clustering techniques to group States/UTs based on their forest-cover composition and recent changes in forest-cover categories.

## 3. Project Objectives

1. Analyse forest-cover patterns across Indian States and Union Territories.
2. Calculate forest-cover percentages relative to geographical area.
3. Apply K-means clustering to group States/UTs with similar forest-cover characteristics.
4. Compare K-means with hierarchical clustering.
5. Evaluate clustering quality using the Silhouette Score.
6. Use Principal Component Analysis (PCA) for two-dimensional visualisation.
7. Identify States/UTs experiencing declines in combined VDF and MDF.
8. Evaluate clustering stability using different random seeds and bootstrap resampling.
9. Compare clustering performance with baseline approaches.
10. Perform feature ablation and transformation sensitivity analysis.
11. Examine the relationship of the findings with SDG 15 (Life on Land) and SDG 13 (Climate Action).

## 4. Intended Users and Decision Context

The findings may be useful to:

- Forest Survey of India (FSI).
- State Forest Departments.
- Environmental researchers.
- Policymakers and conservation planners.

The clusters can help identify States/UTs with similar forest-cover characteristics and indicate where further district-level investigation or forest monitoring may be useful.

The results are exploratory and should not be used independently for funding allocation, State rankings or policy decisions.

## 5. Dataset Description

The dataset is based on State/UT-wise forest-cover information from the India State of Forest Report 2019 and 2021.

**Dataset source:** Open Government Data (OGD) Platform India.

**Dataset URL:**

https://ap.data.gov.in/resource/stateuts-wise-forest-cover-india-state-forest-report-isfr-during-2021

**Original publishing organisation:** Forest Survey of India (FSI), Ministry of Environment, Forest and Climate Change, Government of India.

The prepared dataset contains 36 observations:

- 28 States
- 8 Union Territories

### 5.1 Dataset Licence and Attribution

The dataset-specific metadata identifies the applicable licence as the Government Open Data License – India (GODL-India).

The data are based on the India State of Forest Report published by the Forest Survey of India.

**Dataset access date:** [INSERT ACTUAL ACCESS DATE]

**Licence:** https://data.gov.in/Godl

Appropriate attribution to the original data source has been maintained.

### 5.2 Clustering Features

Six numerical features are used for the main clustering analysis.

| Feature | Description |
|---|---|
| VDF_2021_pct_GA | Very Dense Forest percentage of geographical area in 2021 |
| MDF_2021_pct_GA | Moderately Dense Forest percentage of geographical area in 2021 |
| OF_2021_pct_GA | Open Forest percentage of geographical area in 2021 |
| VDF_change_pct_GA | Change in Very Dense Forest percentage of geographical area |
| MDF_change_pct_GA | Change in Moderately Dense Forest percentage of geographical area |
| OF_change_pct_GA | Change in Open Forest percentage of geographical area |

The features are normalised relative to geographical area and standardised before clustering.

## 6. Methodology

### 6.1 Data Preprocessing

- Loaded the prepared dataset using pandas.
- Checked for missing values and duplicate observations.
- Selected six numerical clustering features.
- Normalised forest-cover measurements relative to geographical area.
- Standardised the selected features using StandardScaler.

### 6.2 K-means Clustering

K-means clustering was applied to identify groups of States/UTs with similar forest-cover profiles.

The number of clusters was evaluated for K = 2 to 6.

The main configuration used:

- Number of clusters: K = 3
- Random state: 42
- Number of initialisations: 20

### 6.3 Hierarchical Clustering

Agglomerative hierarchical clustering using Ward linkage was performed as a comparative method.

The resulting Silhouette Scores were compared with those obtained using K-means.

### 6.4 Silhouette Score

The Silhouette Score was used to evaluate cluster cohesion and separation.

A higher score generally indicates better-defined clusters.

### 6.5 Principal Component Analysis

PCA reduced the six standardised features to two principal components for visualisation.

PCA was used only to display the clustering results in two dimensions. The main clustering analysis used all six standardised features.

### 6.6 Dense-Forest Change Analysis

Combined dense forest change was calculated separately:

**Combined Dense Forest Change = VDF Change + MDF Change**

Negative values indicate a decline in combined Very Dense Forest and Moderately Dense Forest area.

### 6.7 Baseline and Validation Analysis

The selected clustering solution was evaluated using:

- Silhouette Score.
- Davies-Bouldin Index.
- Calinski-Harabasz Index.
- Adjusted Rand Index.
- Rule-based baseline comparison.
- Permutation-based null baseline.

### 6.8 Cluster Stability Analysis

Cluster stability was examined through repeated K-means runs with different random seeds.

The Adjusted Rand Index (ARI) was used to compare cluster assignments.

Bootstrap resampling was also performed to examine the consistency of cluster membership across resampled datasets.

### 6.9 Exploratory Data Analysis and Feature Skewness

Exploratory analysis examined:

- Descriptive statistics of the six clustering features.
- Feature distributions.
- Correlation between clustering features.
- Differences between States and Union Territories.
- Skewness of the selected features.

### 6.10 Feature Ablation Analysis

Feature ablation was performed to examine how different feature groups affect the clustering solution.

The configurations included:

- All six features.
- Forest-cover level features only.
- Forest-cover change features only.
- Features excluding Open Forest.

Cluster memberships were compared using ARI.

### 6.11 Transformation Sensitivity Analysis

The original feature representation was compared with alternative transformations:

- Original features.
- Logarithmic transformation.
- Quantile/rank-based transformation.

Silhouette Scores, selected cluster counts and cluster membership were examined to assess sensitivity to data transformation.

## 7. Experimental Results

### 7.1 Silhouette Score Comparison

| K | K-means | Hierarchical Ward |
|---:|---:|---:|
| 2 | 0.4401 | 0.4337 |
| 3 | 0.4570 | 0.4611 |
| 4 | 0.4567 | 0.3677 |
| 5 | 0.3881 | 0.3768 |
| 6 | 0.3942 | 0.3942 |

Both methods achieved their highest tested Silhouette Score at K = 3.

The main reported solution uses K-means with three clusters and a Silhouette Score of 0.4570.

### 7.2 K-means Cluster Distribution

| Cluster | Number of States/UTs |
|---|---:|
| Cluster 1 | 8 |
| Cluster 2 | 27 |
| Cluster 3 | 1 |

Cluster 3 contains Andaman and Nicobar Islands, which has a distinctive Very Dense Forest share.

This singleton cluster is interpreted cautiously.

### 7.3 Cluster Profiles

| Cluster | VDF 2021 (%) | MDF 2021 (%) | OF 2021 (%) |
|---|---:|---:|---:|
| 1 | 6.27 | 34.35 | 32.91 |
| 2 | 3.36 | 9.88 | 10.26 |
| 3 | 68.83 | 8.28 | 4.64 |

The cluster profiles show differences in forest-cover composition across the identified groups.

### 7.4 Dense-Forest Change Findings

The prepared dataset indicates:

- Very Dense Forest change: +501 sq. km.
- Moderately Dense Forest change: approximately −1,582.74 sq. km.
- Open Forest change: approximately +2,621.14 sq. km.
- Combined VDF and MDF change: approximately −1,081.74 sq. km.
- 25 out of 36 States/UTs show a negative combined VDF and MDF change.

These are descriptive changes in reported forest-density categories. They do not independently establish the causes of forest-cover change.

### 7.5 Stability Analysis

Seed-based and bootstrap stability analyses were performed using ARI.

The bootstrap analysis reported:

- Number of bootstrap resamples: 100.
- Mean ARI: approximately 0.661.
- Standard deviation: approximately 0.170.

These results indicate that cluster membership can vary across initialisations and resampled datasets.

### 7.6 Baseline Comparison

The selected K-means solution achieved:

- Silhouette Score: 0.457.
- Davies-Bouldin Index: 0.930.

The rule-based baseline achieved a Silhouette Score of 0.054.

The permutation null baseline reported:

- Mean Silhouette Score: approximately 0.246.
- 95th percentile: approximately 0.344.
- Permutation p-value: 0.020.

### 7.7 Feature Ablation

Feature ablation examined the effect of using different combinations of forest-cover level and change features.

The results showed that clustering membership and evaluation scores were sensitive to the feature configuration.

ARI was used to compare memberships across the feature subsets.

### 7.8 Transformation Sensitivity

Alternative feature transformations changed the selected number of clusters and cluster memberships.

This demonstrates that the clustering solution is sensitive to the representation and distribution of the input features.

### 7.9 Interpretation

The results provide an exploratory comparison of forest-cover composition and changes across Indian States and Union Territories.

The clusters are not rankings of environmental performance. Changes in forest-density categories may reflect canopy-density transitions, regrowth, measurement differences or reporting-related factors.

## 8. Technologies and Libraries

- Python 3
- pandas
- NumPy
- Matplotlib
- scikit-learn
- SciPy
- openpyxl

The exact dependencies required to execute the scripts are listed in `requirements.txt`.

## 9. Project Folder Structure

The repository is organised into the following main directories:

```text
Forest-Cover-Clustering/
│
├── Code/
│   ├── clustering_analysis.py
│   ├── baseline_analysis.py
│   ├── extra_analysis.py
│   └── dense_forest_analysis.py
│
├── Dataset/
│   ├── forest_cover_clustering_features.csv
│   └── forest_cover_full_dataset.csv
│
├── Results/
│   ├── cluster_assignments.csv
│   ├── cluster_profiles.csv
│   ├── silhouette_scores.csv
│   ├── run_summary.txt
│   ├── seed_stability.csv
│   ├── exploratory_statistics.txt
│   ├── feature_skewness.csv
│   ├── ablation_results.csv
│   ├── ablation_membership.csv
│   ├── transform_results.csv
│   ├── transform_membership.csv
│   ├── dense_forest_change_by_state.csv
│   ├── dense_forest_summary.txt
│   ├── kmeans_pca_plot.png
│   ├── cluster_feature_boxplots.png
│   ├── silhouette_score_comparison.png
│   └── permutation_null_comparison.png
│
├── Report/
│   └── Forest_Cover_Clustering_Report.pdf
│
├── README.md
└── requirements.txt
```

The structure above documents the intended project organisation. Filenames should match the actual files committed to the repository.

## 10. Installation and Execution

### Step 1: Install Python

Install Python 3 on your system.

### Step 2: Clone the Repository

```bash
git clone https://github.com/anshipandey446/Forest-Cover-Clustering.git
cd Forest-Cover-Clustering
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Main Clustering Analysis

```bash
python Code/clustering_analysis.py
```

This performs the main K-means and hierarchical clustering analysis, evaluation, cluster profiling and visualisation.

### Step 5: Run Baseline Analysis

```bash
python Code/baseline_analysis.py
```

This performs the baseline and stability analyses implemented in the script.

### Step 6: Run Extra Analysis

```bash
python Code/extra_analysis.py
```

This performs the additional exploratory statistics, feature skewness, feature ablation and transformation sensitivity analyses.

### Step 7: Run Dense-Forest Analysis

```bash
python Code/dense_forest_analysis.py
```

This calculates State/UT-wise dense-forest changes and generates the corresponding summary outputs.

### Step 8: View Results

The generated CSV files, text summaries and figures are saved in the `Results/` directory.

Run the scripts from the repository root directory so that relative paths resolve correctly.

## 11. Limitations and Responsible Interpretation

1. The dataset contains only 36 State/UT observations.
2. The analysis covers the 2019–2021 reporting period.
3. Clustering results depend on feature selection, standardisation and algorithm settings.
4. One cluster contains only a single observation.
5. Aggregate State/UT-level data hide local differences in forest conditions.
6. Changes in forest-density categories do not necessarily represent direct forest destruction.
7. Small geographical areas can produce relatively large percentage changes.
8. Satellite-data, measurement and boundary differences may affect comparisons between reporting cycles.
9. Stability analysis indicates variation across initialisations and resampling.
10. Cluster labels should not be used independently for policy rankings or funding decisions.

## 12. Future Scope

Future improvements may include:

- Adding more ISFR reporting years.
- Including climate and land-use variables.
- Extending the analysis to district-level observations.
- Comparing additional clustering algorithms.
- Exploring spatial clustering and geographical visualisation.
- Investigating forest fragmentation and biodiversity indicators.

## 13. Conclusion

This project demonstrates how unsupervised machine learning can be used to explore forest-cover patterns across Indian States and Union Territories.

K-means and hierarchical clustering were compared, and the selected K-means solution contained three clusters.

Additional analyses examined stability, baseline performance, feature ablation, skewness and transformation sensitivity.

A separate dense-forest analysis identified States/UTs showing declines in combined Very Dense Forest and Moderately Dense Forest area.

The study provides an exploratory, data-driven approach to comparing forest-cover composition and change.

## 14. AI Use Declaration

AI tools were used to support conceptual understanding, report drafting, language editing and code debugging. The project author reviewed the outputs and takes responsibility for the final dataset, code, calculations and report.

## 15. References

1. Forest Survey of India. (2019). *India State of Forest Report 2019*. Ministry of Environment, Forest and Climate Change, Government of India.
2. Forest Survey of India. (2021). *India State of Forest Report 2021*. Ministry of Environment, Forest and Climate Change, Government of India.
3. Open Government Data Platform India. *State/UT-wise Forest Cover as per India State of Forest Report (ISFR) during 2021*. https://ap.data.gov.in/resource/stateuts-wise-forest-cover-india-state-forest-report-isfr-during-2021
4. Government Open Data License – India. https://data.gov.in/Godl
5. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
6. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236–244.
7. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
8. Jolliffe, I. T., and Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374, 20150202.
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

**Project:** Clustering Indian States and Union Territories Based on Forest Cover and Its Changes Using Machine Learning

**Institution:** Ramanujan College, University of Delhi

**Course:** B.Sc. (Hons.) Computer Science
