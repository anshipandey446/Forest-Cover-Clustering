
# Clustering Indian States and Union Territories Based on Forest Cover and Its Changes Using Machine Learning

## 1. Project Overview

This project applies unsupervised machine learning techniques to group Indian States and Union Territories according to their forest-cover composition and changes between 2019 and 2021.

The study uses forest-cover data from the India State of Forest Report (ISFR) published by the Forest Survey of India (FSI).

India aims to increase its forest and tree cover. Therefore, understanding differences in forest density and changes across States and Union Territories is important for environmental monitoring and planning.

The project focuses on three forest-cover categories:

- Very Dense Forest (VDF)
- Moderately Dense Forest (MDF)
- Open Forest (OF)

The objective is to identify groups of States/UTs with similar forest-cover characteristics and examine changes in dense forest cover.

## 2. Problem Statement

India aims to bring one-third of its geographical area under forest and tree cover.

This project clusters States and Union Territories using their shares of Very Dense Forest, Moderately Dense Forest and Open Forest, along with recent changes in these categories.

The analysis also identifies States/UTs experiencing a decline in combined dense forest cover.

## 3. Objectives

The main objectives are:

1. To analyse forest-cover patterns across Indian States and Union Territories.
2. To calculate forest-density category percentages relative to geographical area.
3. To apply K-means clustering to group States/UTs with similar forest-cover characteristics.
4. To compare K-means with hierarchical clustering.
5. To evaluate clustering solutions using the Silhouette Score.
6. To use Principal Component Analysis (PCA) for two-dimensional visualisation.
7. To identify States/UTs showing a decline in combined Very Dense Forest and Moderately Dense Forest.

## 4. Dataset Description

The dataset is based on the India State of Forest Report 2019 and 2021.

**Primary data source:** Forest Survey of India (FSI), Ministry of Environment, Forest and Climate Change, Government of India.

**Dataset portal:**

https://ap.data.gov.in/resource/stateuts-wise-forest-cover-india-state-forest-report-isfr-during-2021

The prepared dataset contains 36 observations:

- 28 States
- 8 Union Territories

The full dataset contains forest-cover category areas and their changes between 2019 and 2021.

### Input Features

Six numerical features are used for clustering:

| Feature | Description |
|---|---|
| VDF_2021_pct_GA | Very Dense Forest percentage of geographical area in 2021 |
| MDF_2021_pct_GA | Moderately Dense Forest percentage of geographical area in 2021 |
| OF_2021_pct_GA | Open Forest percentage of geographical area in 2021 |
| VDF_change_pct_GA | Change in Very Dense Forest as a percentage of geographical area |
| MDF_change_pct_GA | Change in Moderately Dense Forest as a percentage of geographical area |
| OF_change_pct_GA | Change in Open Forest as a percentage of geographical area |

All six features are normalised relative to geographical area before standardisation.

**Geographical-area adjustment:** The prepared dataset uses 54,624 sq. km. for Jammu and Kashmir and 168,055 sq. km. for Ladakh as geographical-area denominators. The original downloaded area fields are retained separately in the prepared data because they are not suitable as direct denominators for these observations.

## 5. Methodology

The project follows these steps:

### 5.1 Data Collection

Forest-cover data for 2019 and 2021 were compiled from the Forest Survey of India reports and the Open Government Data Platform.

### 5.2 Data Preprocessing

- Loaded the prepared dataset using pandas.
- Checked for duplicate State/UT observations.
- Checked for missing values in the clustering features.
- Used forest-cover percentages relative to geographical area.
- Selected six numerical features for clustering.

### 5.3 Feature Standardisation

StandardScaler from scikit-learn was used to standardise the six numerical features.

Standardisation makes the features comparable by centring them around their mean and scaling them according to their standard deviation.

### 5.4 K-means Clustering

K-means clustering was applied using the following settings:

- Number of clusters tested: K = 2 to 6
- Random state: 42
- Number of initialisations: 20

The K-means solution with the highest Silhouette Score among the tested values was selected.

### 5.5 Hierarchical Clustering

Agglomerative hierarchical clustering using Ward linkage was also performed for K = 2 to 6.

Its Silhouette Scores were compared with those of K-means.

### 5.6 Silhouette Score

The Silhouette Score was used to evaluate cluster cohesion and separation.

A higher score generally indicates better separation between clusters and better grouping within clusters.

### 5.7 Principal Component Analysis

PCA was applied to reduce the six standardised features to two principal components.

It was used only for visualisation, not as the input to the clustering algorithms.

### 5.8 Dense-Forest Change Analysis

Combined dense forest change was calculated separately:

**Dense Forest Change = VDF Change + MDF Change**

A negative value indicates a decline in combined dense forest area.

This descriptive calculation is separate from the machine-learning clustering process.

### 5.9 Cluster Stability Analysis

K-means cluster stability is evaluated by repeating the selected clustering configuration with different random seeds.

The Adjusted Rand Index (ARI) is used to compare the resulting cluster assignments. The results are saved in:

`Results/stability_results.csv`

## 6. Experimental Results

The analysis uses 36 State/UT observations and six input features.

The best K-means solution among the tested values was K = 3.

| Evaluation | Result |
|---|---:|
| Number of observations | 36 |
| Number of features | 6 |
| Selected K-means clusters | 3 |
| K-means Silhouette Score | 0.4570 |
| Hierarchical Ward Silhouette Score at K = 3 | 0.4611 |

### K-means Cluster Sizes

| Cluster | Number of States/UTs |
|---|---:|
| Cluster 1 | 8 |
| Cluster 2 | 27 |
| Cluster 3 | 1 |

Cluster numbers are arbitrary identifiers and do not represent rankings.

Cluster 3 contains one observation, Andaman and Nicobar Islands. This small cluster is treated as a limitation and is interpreted cautiously.

### Dense-Forest Change Findings

Combined dense forest refers to Very Dense Forest and Moderately Dense Forest.

The prepared dataset indicates:

- Very Dense Forest change: +501 sq. km.
- Moderately Dense Forest change: approximately -1,582.74 sq. km.
- Open Forest change: approximately +2,621.14 sq. km.
- Combined VDF and MDF change: approximately -1,081.74 sq. km.
- States/UTs showing a decline in combined dense forest: 25 out of 36.

These findings are descriptive and do not establish the causes of forest-cover changes.

## 7. Figures and Visualisations

The project generates the following visualisations:

1. **PCA Cluster Visualisation:** Shows the distribution of States/UTs in a two-dimensional PCA representation, coloured by their K-means cluster.
2. **Cluster Feature Boxplots:** Shows the distribution of the six input features across the K-means clusters.
3. **Silhouette Score Comparison:** Compares K-means and hierarchical Ward clustering for K = 2 to 6.

The generated figures are saved in the Results folder.

## 8. Technologies and Libraries Used

- Python 3
- pandas
- NumPy
- Matplotlib
- scikit-learn
- openpyxl

### Machine Learning Techniques

- K-means Clustering
- Agglomerative Hierarchical Clustering
- Ward Linkage
- StandardScaler
- Principal Component Analysis
- Silhouette Score Evaluation

## 9. Project Folder Structure

```text
Forest-Cover-Clustering/
│
├── Code/
│   ├── clustering_analysis.py
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
│   ├── kmeans_pca_plot.png
│   ├── cluster_feature_boxplots.png
│   ├── silhouette_score_comparison.png
│   ├── dense_forest_change_by_state.csv
│   └── dense_forest_summary.txt
│
├── Report/
│   └── Forest_Cover_Clustering_Report.pdf
│
├── README.md
└── requirements.txt
```

## 10. Installation and Execution

### Step 1: Install Python

Install Python 3 on your system.

### Step 2: Install Required Libraries

Open Command Prompt or a terminal in the project root directory and run:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Clustering Analysis

Execute:

```bash
python Code/clustering_analysis.py
```

This script performs:

- Dataset loading and validation
- Feature standardisation
- K-means clustering
- Hierarchical clustering
- Silhouette Score calculation
- Cluster profile generation
- PCA visualisation
- Feature distribution boxplots
- Silhouette comparison graph

### Step 4: Run Dense-Forest Change Analysis

Execute:

```bash
python Code/dense_forest_analysis.py
```

This script calculates State/UT-wise combined dense-forest changes and saves the results in the Results folder.

### Step 5: Check Generated Results

Open the Results folder to view the generated CSV files, text summaries and figures.

## 11. Limitations

1. The dataset contains only 36 State/UT observations.
2. The analysis covers the reporting period 2019–2021.
3. Clustering results depend on the selected features, standardisation and algorithm settings.
4. Cluster 3 contains only one observation, which requires cautious interpretation.
5. Silhouette Score measures mathematical cluster separation and does not establish ecological validity.
6. Aggregate State/UT-level data cannot explain local forest conditions, biodiversity, fragmentation or the causes of forest-cover change.
7. The study is exploratory and does not establish causal relationships.

## 12. Future Scope

Future improvements may include:

- Incorporating additional ISFR reporting years.
- Including rainfall, land-use and climate-related variables.
- Studying forest fragmentation and biodiversity indicators.
- Comparing additional clustering algorithms.
- Testing cluster stability with alternative initialisations and sampling approaches.
- Developing a geographical visualisation of the resulting State/UT clusters.

## 13. Conclusion

This project demonstrates the application of unsupervised machine learning to the analysis of forest-cover patterns across Indian States and Union Territories.

K-means and hierarchical clustering were compared using the Silhouette Score. The selected K-means solution contained three clusters, and PCA was used to visualise the resulting groups.

A separate dense-forest change analysis identified States/UTs experiencing reductions in combined Very Dense Forest and Moderately Dense Forest area.

The project provides an exploratory, data-driven approach to comparing forest-cover composition and change across Indian States and Union Territories.

## 14. References

1. Forest Survey of India. (2019). *India State of Forest Report 2019*. Ministry of Environment, Forest and Climate Change, Government of India.

2. Forest Survey of India. (2021). *India State of Forest Report 2021*. Ministry of Environment, Forest and Climate Change, Government of India.

3. Open Government Data Platform India. State/UT-wise forest cover, India State of Forest Report 2021.  
   https://ap.data.gov.in/resource/stateuts-wise-forest-cover-india-state-forest-report-isfr-during-2021

4. MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.

5. Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236–244.

6. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.

7. Jolliffe, I. T., and Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374, 20150202.

8. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

---

**Project:** Clustering Indian States and Union Territories Based on Forest Cover and Its Changes Using Machine Learning

**Course:** Machine Learning (CSA), B.Sc. (H) Computer Science

**Institution:** Ramanujan College, University of Delhi
