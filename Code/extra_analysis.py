# Extra analyses requested in review: seed stability, exploratory statistics,
# feature ablation and skew/transform sensitivity.
# Run (from the Code folder): python extra_analysis.py
# Input : ../Dataset/forest_cover_clustering_features.csv
# Output: ../Results/seed_stability.csv, exploratory_statistics.txt, feature_skewness.csv,
#         ablation_results.csv, ablation_membership.csv, transform_results.csv, transform_membership.csv

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "forest_cover_clustering_features.csv"
RESULTS_DIR = ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

LEVELS = ["VDF_2021_pct_GA", "MDF_2021_pct_GA", "OF_2021_pct_GA"]
CHANGES = ["VDF_change_pct_GA", "MDF_change_pct_GA", "OF_change_pct_GA"]
FEATURES = LEVELS + CHANGES
UTS = ["Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
       "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"]   # the 8 UTs in the dataset

data = pd.read_csv(DATA_PATH)
K = 3

def cluster(X, k=K, seed=42):
    return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)

X_full = StandardScaler().fit_transform(data[FEATURES])
base = cluster(X_full)

# ---------------------------------------------------------------
# 1. Seed stability: same setup as clustering_analysis.py (seed 10 is the reference run)
# ---------------------------------------------------------------
seeds = [10, 20, 30, 40, 50]
runs = {s: cluster(X_full, seed=s) for s in seeds}
seed_df = pd.DataFrame({"Random_Seed": seeds[1:],
                        "Adjusted_Rand_Index": [adjusted_rand_score(runs[10], runs[s]) for s in seeds[1:]]})
seed_df.to_csv(RESULTS_DIR / "seed_stability.csv", index=False)

# ---------------------------------------------------------------
# 2. Exploratory statistics quoted in the report
# ---------------------------------------------------------------
lines = ["EXPLORATORY STATISTICS", f"Observations: {len(data)}",
         f"States: {(~data['State/UT'].isin(UTS)).sum()}, UTs: {data['State/UT'].isin(UTS).sum()}", ""]
lines.append("Descriptive statistics (mean, std, min, median, max):")
d = data[FEATURES].agg(["mean", "std", "min", "median", "max"]).T
lines.append(d.round(3).to_string()); lines.append("")
for col, nm in zip(LEVELS, ["VDF", "MDF", "OF"]):
    r = data.loc[data[col].idxmax()]; lines.append(f"Highest {nm} 2021 share: {r['State/UT']} ({r[col]:.3f}%)")
corr = data[FEATURES].corr()
lines += ["", f"Corr MDF2021-OF2021: {corr.loc['MDF_2021_pct_GA','OF_2021_pct_GA']:.2f}",
          f"Corr OF2021-MDF change: {corr.loc['OF_2021_pct_GA','MDF_change_pct_GA']:.2f}",
          f"Corr MDF2021-MDF change: {corr.loc['MDF_2021_pct_GA','MDF_change_pct_GA']:.2f}", ""]
grp = np.where(data["State/UT"].isin(UTS), "UT", "State")
med = data.groupby(grp)[LEVELS].median().round(2)
lines += ["Median 2021 shares, States vs UTs:", med.to_string(), ""]
dense_pp = data["VDF_change_pct_GA"] + data["MDF_change_pct_GA"]
lines.append(f"Units with negative VDF+MDF change (percentage-point basis): {(dense_pp < 0).sum()} of {len(data)}")
lines.append("(The km2-based count and the ten largest declines come from dense_forest_analysis.py.)")
(RESULTS_DIR / "exploratory_statistics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------
# 3. Skewness of each feature
# ---------------------------------------------------------------
pd.DataFrame({"Feature": FEATURES, "Skewness": [skew(data[c]) for c in FEATURES]}).round(3)\
  .to_csv(RESULTS_DIR / "feature_skewness.csv", index=False)

# ---------------------------------------------------------------
# 4. Feature ablation (K = 3, and K chosen by best silhouette for each set)
# ---------------------------------------------------------------
sets = {"All six features (reference)": FEATURES,
        "Levels only (3 shares in 2021)": LEVELS,
        "Changes only (3 changes 2019-2021)": CHANGES,
        "Drop open forest (VDF, MDF levels + changes)": ["VDF_2021_pct_GA", "MDF_2021_pct_GA", "VDF_change_pct_GA", "MDF_change_pct_GA"]}
rows, member = [], pd.DataFrame({"State/UT": data["State/UT"], "Reference_K3": base + 1})
for name, cols in sets.items():
    X = StandardScaler().fit_transform(data[cols]); lab = cluster(X)
    sil = {k: silhouette_score(X, cluster(X, k)) for k in range(2, 7)}
    kbest = max(sil, key=sil.get)
    rows.append({"Feature_set": name, "n_features": len(cols), "Silhouette_K3": silhouette_score(X, lab),
                 "Best_K_by_silhouette": kbest, "Best_silhouette": sil[kbest],
                 "ARI_vs_reference_K3": adjusted_rand_score(base, lab),
                 "Cluster_sizes_K3": "/".join(map(str, sorted(np.bincount(lab), reverse=True)))})
    member[name] = lab + 1
pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
member.to_csv(RESULTS_DIR / "ablation_membership.csv", index=False)

# ---------------------------------------------------------------
# 5. Skew handling: log1p on the (non-negative) 2021 shares, and a rank (quantile) transform of all features
# ---------------------------------------------------------------
variants = {}
logged = data[FEATURES].copy(); logged[LEVELS] = np.log1p(logged[LEVELS])
variants["Original (reference)"] = data[FEATURES]
variants["log1p on 2021 shares"] = logged
variants["Rank (quantile-normal) transform, all six"] = pd.DataFrame(
    QuantileTransformer(n_quantiles=len(data), output_distribution="normal", random_state=42).fit_transform(data[FEATURES]), columns=FEATURES)
rows, member = [], pd.DataFrame({"State/UT": data["State/UT"], "Reference_K3": base + 1})
for name, df in variants.items():
    X = StandardScaler().fit_transform(df); lab = cluster(X)
    sil = {k: silhouette_score(X, cluster(X, k)) for k in range(2, 7)}; kbest = max(sil, key=sil.get)
    rows.append({"Version": name, "Silhouette_K3": silhouette_score(X, lab), "Best_K_by_silhouette": kbest,
                 "Best_silhouette": sil[kbest], "ARI_vs_reference_K3": adjusted_rand_score(base, lab),
                 "Cluster_sizes_K3": "/".join(map(str, sorted(np.bincount(lab), reverse=True))),
                 "Singleton_state": ", ".join(data.loc[[i for c in range(K) if (lab == c).sum() == 1 for i in np.where(lab == c)[0]], "State/UT"])})
    member[name] = lab + 1
pd.DataFrame(rows).round(4).to_csv(RESULTS_DIR / "transform_results.csv", index=False)
member.to_csv(RESULTS_DIR / "transform_membership.csv", index=False)

print(seed_df.round(4).to_string(index=False), "Mean:", round(seed_df.Adjusted_Rand_Index.mean(), 4), "\n")
print("\n".join(lines)); print()
print(pd.read_csv(RESULTS_DIR / "feature_skewness.csv").to_string(index=False)); print()
print(pd.read_csv(RESULTS_DIR / "ablation_results.csv").to_string(index=False)); print()
print(pd.read_csv(RESULTS_DIR / "transform_results.csv").to_string(index=False))
