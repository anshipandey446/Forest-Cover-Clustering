# Forest-Cover-Clustering
Machine Learning project on clustering Indian States and Union Territories based on forest cover and its changes using ISFR data.
## How to Run the Project

### Requirements
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- openpyxl

### Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Run the Clustering Analysis

From the project root directory, execute:

```bash
python Code/clustering_analysis.py
```

This performs data validation, standardisation, K-means clustering, hierarchical clustering, silhouette evaluation and PCA visualisation.

### Run the Dense-Forest Change Analysis

Execute:

```bash
python Code/dense_forest_analysis.py
```

This calculates state/UT-wise changes in combined Very Dense Forest and Moderately Dense Forest, identifies observations with a decline, and saves the results.

### Generated Results

The Results folder contains:
- Cluster assignments
- Cluster profiles
- Silhouette scores
- Run summary
- PCA visualisation
- Feature distribution boxplots
- Silhouette comparison graph
- Dense-forest change by State/UT
- Dense-forest analysis summary

### Reproducibility

The clustering experiment uses StandardScaler, K-means with random_state=42 and n_init=20, and Ward hierarchical clustering. The silhouette score is used to compare cluster configurations for K values from 2 to 6.
