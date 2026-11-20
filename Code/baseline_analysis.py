# Baselines, validity and stability checks for the forest-cover clustering
# Run (from the Code folder): python baseline_analysis.py
#
# Input : ../Dataset/forest_cover_clustering_features.csv
# Output: ../Results/baseline_comparison.csv, ../Results/validity_indices.csv,
#         ../Results/bootstrap_stability.csv, ../Results/sdg_indicators.csv,
#         ../Results/cluster_names.csv, ../Results/baseline_summary.txt,
#         ../Results/null_baseline_plot.png

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "forest_cover_clustering_features.csv"
RESULTS_DIR = ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)
#FIXED SEED SO MY RESULT REPEAT EVRY TIME I RUN IT
SEED = 42

# FOREST SHARES IN 2021+3 CHANGES FROM 2019 TO 2021 , ALL AS % OF AREA
FEATURES = ["VDF_2021_pct_GA", "MDF_2021_pct_GA", "OF_2021_pct_GA",
            "VDF_change_pct_GA", "MDF_change_pct_GA", "OF_change_pct_GA"]

data = pd.read_csv(DATA_PATH)
#put all the features on the same scale so no one feature dominates
X = StandardScaler().fit_transform(data[FEATURES])
n = len(data)
lines = []          # text summary written at the end


#helper that runs k-means and returns cluster lables
def kmeans_labels(X, k, seed=SEED, n_init=20):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit_predict(X)

# ------------------------------------------------------------------
# 1. Extra validity indices for K = 2..6 (silhouette alone can mislead)
# ------------------------------------------------------------------

#silhouette alone can mislead, so i also check davies-bouldin 
rows = []
for k in range(2, 7):
    lab = kmeans_labels(X, k)
    rows.append({"K": k,
                 "Silhouette": silhouette_score(X, lab),
                 "Davies_Bouldin": davies_bouldin_score(X, lab),
                 "Calinski_Harabasz": calinski_harabasz_score(X, lab),
                 "Smallest_cluster_size": int(np.bincount(lab).min())})
validity = pd.DataFrame(rows)
validity.to_csv(RESULTS_DIR / "validity_indices.csv", index=False)

# ------------------------------------------------------------------
# 2. BASELINE 1: permutation null.
#    Shuffle every feature column independently. This keeps each feature's
#    distribution but destroys any joint (cluster) structure. If real data
#    does not score above this, the clusters are not stronger than chance.
# ------------------------------------------------------------------
BEST_K = 3
rng = np.random.default_rng(SEED)
real_sil = silhouette_score(X, kmeans_labels(X, BEST_K))
null_scores = []
for _ in range(200):
    Xn = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    null_scores.append(silhouette_score(Xn, kmeans_labels(Xn, BEST_K, n_init=5)))
null_scores = np.array(null_scores)
p_value = (1 + (null_scores >= real_sil).sum()) / (1 + len(null_scores))

plt.figure(figsize=(7, 4.5))
plt.hist(null_scores, bins=25, alpha=0.8, label="Permuted (null) data")
plt.axvline(real_sil, color="red", linewidth=2, label=f"Real data = {real_sil:.3f}")
plt.xlabel("Silhouette score (K-means, K=3)")
plt.ylabel("Count of 200 shuffles")
plt.title("Real silhouette score against the permutation null")
plt.legend(); plt.tight_layout()
plt.savefig(RESULTS_DIR / "null_baseline_plot.png", dpi=300)
plt.close()

# ------------------------------------------------------------------
# 3. BASELINE 2: simple rule with no machine learning.
#    Group states by total forest cover (VDF+MDF+OF as % of area).
# ------------------------------------------------------------------
total_cover = data[["VDF_2021_pct_GA", "MDF_2021_pct_GA", "OF_2021_pct_GA"]].sum(axis=1)
rule_labels = np.select([total_cover >= 33, total_cover >= 15], [2, 1], default=0)
kmeans_lab = kmeans_labels(X, BEST_K)

baseline_table = pd.DataFrame([
    {"Method": "K-means (K=3)", "Silhouette": real_sil,
     "Davies_Bouldin": davies_bouldin_score(X, kmeans_lab)},
    {"Method": "Rule on total cover (<15, 15-33, >=33 %)",
     "Silhouette": silhouette_score(X, rule_labels),
     "Davies_Bouldin": davies_bouldin_score(X, rule_labels)},
    {"Method": "Permutation null, mean of 200",
     "Silhouette": null_scores.mean(), "Davies_Bouldin": np.nan},
    {"Method": "Permutation null, 95th percentile",
     "Silhouette": np.percentile(null_scores, 95), "Davies_Bouldin": np.nan},
])
baseline_table.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
lines += [f"Real K-means (K=3) silhouette: {real_sil:.4f}",
          f"Null mean / 95th percentile: {null_scores.mean():.4f} / {np.percentile(null_scores, 95):.4f}",
          f"Permutation p-value: {p_value:.4f}",
          f"Rule-based baseline silhouette: {silhouette_score(X, rule_labels):.4f}",
          f"ARI between rule and K-means: {adjusted_rand_score(rule_labels, kmeans_lab):.4f}", ""]

# ------------------------------------------------------------------
# 4. Stability: bootstrap resampling (stronger than changing the seed only)
# ------------------------------------------------------------------
boot = []
for i in range(100):
    idx = rng.choice(n, n, replace=True)
    m = KMeans(n_clusters=BEST_K, random_state=i, n_init=5).fit(X[idx])
    boot.append(adjusted_rand_score(kmeans_lab, m.predict(X)))
pd.DataFrame({"Bootstrap_ARI": boot}).to_csv(RESULTS_DIR / "bootstrap_stability.csv", index=False)
lines += [f"Bootstrap ARI (100 resamples): mean {np.mean(boot):.4f}, sd {np.std(boot):.4f}"]

# Sensitivity: remove singleton outlier(s) and re-cluster
single = [c for c in range(BEST_K) if (kmeans_lab == c).sum() == 1]
keep = ~np.isin(kmeans_lab, single)
Xk = StandardScaler().fit_transform(data.loc[keep, FEATURES])
for k in (2, 3):
    lab = kmeans_labels(Xk, k)
    lines.append(f"Without singleton outlier(s) {list(data.loc[~keep, 'State/UT'])}: "
                 f"K={k} silhouette {silhouette_score(Xk, lab):.4f}")
lines.append("")

# ------------------------------------------------------------------
# 5. SDG link: indicators per cluster (SDG 15 forest area; SDG 13 forest quality)
# ------------------------------------------------------------------
data["Total_cover_pct_GA"] = total_cover
data["Dense_2021_pct_GA"] = data["VDF_2021_pct_GA"] + data["MDF_2021_pct_GA"]
data["Dense_change_pp"] = data["VDF_change_pct_GA"] + data["MDF_change_pct_GA"]
data["Gap_to_33pct_pp"] = (33 - total_cover).clip(lower=0)   # National Forest Policy 1988 goal
data["Cluster"] = kmeans_lab + 1
sdg = data.groupby("Cluster").agg(
    States_UTs=("State/UT", "size"),
    Mean_total_cover_pct=("Total_cover_pct_GA", "mean"),
    Mean_dense_share_pct=("Dense_2021_pct_GA", "mean"),
    Mean_dense_change_pp=("Dense_change_pp", "mean"),
    Units_with_dense_decline=("Dense_change_pp", lambda s: int((s < 0).sum())),
    Units_meeting_33pct=("Total_cover_pct_GA", lambda s: int((s >= 33).sum())),
    Mean_gap_to_33pct_pp=("Gap_to_33pct_pp", "mean")).round(3)

def name(r):   # naming convention (state it in the report)
    lvl = ("Forest-rich" if r.Mean_total_cover_pct >= 60 else
           "Moderately forested" if r.Mean_total_cover_pct >= 33 else "Low-forest")
    return lvl + (" with dense-forest loss" if r.Mean_dense_change_pp < -0.1 else "")
sdg["Suggested_name"] = [name(r) for r in sdg.itertuples()]
sdg["Members"] = data.groupby("Cluster")["State/UT"].apply(lambda s: "; ".join(sorted(s)))
sdg.to_csv(RESULTS_DIR / "sdg_indicators.csv")
sdg[["Suggested_name"]].to_csv(RESULTS_DIR / "cluster_names.csv")
lines += [f"Units with total cover >= 33%: {(total_cover >= 33).sum()} of {n}",
          f"Units with a negative VDF+MDF change (percentage-point basis): {(data['Dense_change_pp'] < 0).sum()}", ""]

(RESULTS_DIR / "baseline_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(validity.round(4).to_string(index=False)); print(); print("\n".join(lines))
print(baseline_table.round(4).to_string(index=False)); print(); print(sdg.drop(columns="Members").to_string())
