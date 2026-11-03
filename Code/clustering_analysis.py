# Forest Cover Clustering of Indian States and Union Territories
# Run: python clustering_analysis.py
#
# Input:
#   ../Dataset/forest_cover_clustering_features.csv
#
# Output:
#   ../Results/cluster_assignments.csv
#   ../Results/silhouette_scores.csv
#   ../Results/cluster_profiles.csv
#   ../Results/kmeans_pca_plot.png
#   ../Results/cluster_feature_boxplots.png

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "forest_cover_clustering_features.csv"
RESULTS_DIR = ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "VDF_2021_pct_GA",
    "MDF_2021_pct_GA",
    "OF_2021_pct_GA",
    "VDF_change_pct_GA",
    "MDF_change_pct_GA",
    "OF_change_pct_GA",
]

# 1. Load and check the dataset
data = pd.read_csv(DATA_PATH)
if data["State/UT"].duplicated().any():
    raise ValueError("Duplicate State/UT names found.")
if data[FEATURES].isnull().any().any():
    raise ValueError("Missing values found in clustering features.")

X = data[FEATURES]

# 2. Standardise features so that all six variables are on a comparable scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Compare K-means solutions for K = 2 to 6 using silhouette score.
score_rows = []
kmeans_models = {}

for k in range(2, 7):
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    kmeans_models[k] = (model, labels)
    score_rows.append({
        "Algorithm": "K-Means",
        "K": k,
        "Silhouette_Score": score
    })

# 4. Compare hierarchical clustering for the same K values.
hierarchical_models = {}
for k in range(2, 7):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    hierarchical_models[k] = (model, labels)
    score_rows.append({
        "Algorithm": "Hierarchical_Ward",
        "K": k,
        "Silhouette_Score": score
    })

scores = pd.DataFrame(score_rows)
scores.to_csv(RESULTS_DIR / "silhouette_scores.csv", index=False)

# 5. Select K-means K with the highest silhouette score.
kmeans_scores = scores[scores["Algorithm"] == "K-Means"]
best_k = int(kmeans_scores.loc[kmeans_scores["Silhouette_Score"].idxmax(), "K"])
best_kmeans, kmeans_labels = kmeans_models[best_k]

# Use the same K for the hierarchical comparison.
best_hierarchical, hierarchical_labels = hierarchical_models[best_k]

# 6. Save state-wise cluster assignments.
output = data.copy()
output["KMeans_Cluster"] = kmeans_labels + 1
output["Hierarchical_Cluster"] = hierarchical_labels + 1
output.to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)

# 7. Create cluster profiles using the original, interpretable percentage values.
profile_data = output.groupby("KMeans_Cluster")[FEATURES].mean().round(4)
profile_data["Number_of_States_UTs"] = output.groupby("KMeans_Cluster").size()
profile_data.to_csv(RESULTS_DIR / "cluster_profiles.csv")

# 8. PCA: reduce six standardised features to two components for visualisation.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
for cluster in sorted(set(kmeans_labels)):
    mask = kmeans_labels == cluster
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cluster {cluster + 1}",
        s=65
    )
for i, state in enumerate(data["State/UT"]):
    plt.annotate(state, (X_pca[i, 0], X_pca[i, 1]), fontsize=6, alpha=0.8)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.title(f"State/UT Forest-Cover Clusters (K-means, K={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "kmeans_pca_plot.png", dpi=300)
plt.close()

# 9. Feature boxplots by K-means cluster.
plot_data = output.copy()
plot_data["KMeans_Cluster"] = plot_data["KMeans_Cluster"].astype(str)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, feature in zip(axes.flatten(), FEATURES):
    groups = [
        plot_data.loc[plot_data["KMeans_Cluster"] == cluster, feature].values
        for cluster in sorted(plot_data["KMeans_Cluster"].unique())
    ]
    ax.boxplot(groups, tick_labels=sorted(plot_data["KMeans_Cluster"].unique()))
    ax.set_title(feature.replace("_", " "), fontsize=8)
    ax.set_xlabel("K-means cluster")
    ax.set_ylabel("Percent")
plt.suptitle("Feature Distribution Across K-means Clusters")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "cluster_feature_boxplots.png", dpi=300)
plt.close()

# 10. Save a concise run summary for reproducibility.
summary_lines = [
    f"Observations: {len(data)}",
    f"Features: {len(FEATURES)}",
    f"Selected K for K-means (highest tested silhouette): {best_k}",
    f"K-means silhouette score: {silhouette_score(X_scaled, kmeans_labels):.4f}",
    f"Hierarchical silhouette score at K={best_k}: {silhouette_score(X_scaled, hierarchical_labels):.4f}",
    "",
    "K-means cluster sizes:",
    output["KMeans_Cluster"].value_counts().sort_index().to_string(),
    "",
    "Mean feature values by K-means cluster:",
    profile_data.to_string(),
]
(RESULTS_DIR / "run_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

# 11. Print a concise report for the terminal.
print("\n--- MODEL SUMMARY ---")
print(f"Number of observations: {len(data)}")
print(f"Number of input features: {len(FEATURES)}")
print(f"Best K-means K by silhouette score: {best_k}")
print(f"K-means silhouette score: {silhouette_score(X_scaled, kmeans_labels):.4f}")
print(f"Hierarchical silhouette score (K={best_k}): "
      f"{silhouette_score(X_scaled, hierarchical_labels):.4f}")
print("\nK-means cluster sizes:")
print(output["KMeans_Cluster"].value_counts().sort_index())
print("\nCluster profiles (mean feature percentages):")
print(profile_data)
print(f"\nResults saved in: {RESULTS_DIR}")
