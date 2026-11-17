
from pathlib import Path
import pandas as pd

# Project folders
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Dataset" / "forest_cover_full_dataset.csv"
RESULTS_DIR = ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load the full state-wise dataset
data = pd.read_csv(DATA_PATH)

# Check required columns
required_columns = [
    "State/UT",
    "VDF_change_sq_km",
    "MDF_change_sq_km",
    "OF_change_sq_km"
]

missing = [col for col in required_columns if col not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if data["State/UT"].duplicated().any():
    raise ValueError("Duplicate State/UT observations found.")

# Calculate combined dense forest change
# Dense forest = Very Dense Forest + Moderately Dense Forest
data["Dense_Forest_Change_sq_km"] = (
    data["VDF_change_sq_km"] + data["MDF_change_sq_km"]
)

# Save state-wise changes
result = data[
    [
        "State/UT",
        "VDF_change_sq_km",
        "MDF_change_sq_km",
        "Dense_Forest_Change_sq_km"
    ]
].copy()

result = result.sort_values("Dense_Forest_Change_sq_km")

result.to_csv(
    RESULTS_DIR / "dense_forest_change_by_state.csv",
    index=False
)

# Count States/UTs with a decline in dense forest
declines = result[
    result["Dense_Forest_Change_sq_km"] < 0
]

# Calculate total category-wise changes
total_vdf_change = data["VDF_change_sq_km"].sum()
total_mdf_change = data["MDF_change_sq_km"].sum()
total_of_change = data["OF_change_sq_km"].sum()

total_dense_change = total_vdf_change + total_mdf_change

# Save summary
summary = [
    "DENSE FOREST CHANGE ANALYSIS",
    f"Total observations: {len(data)}",
    f"States/UTs with dense forest decline: {len(declines)}",
    f"Total VDF change: {total_vdf_change:.2f} sq. km.",
    f"Total MDF change: {total_mdf_change:.2f} sq. km.",
    f"Total OF change: {total_of_change:.2f} sq. km.",
    f"Combined VDF + MDF change: {total_dense_change:.2f} sq. km.",
    "",
    "TEN LARGEST DENSE FOREST DECLINES:",
    declines.head(10).to_string(index=False)
]

summary_text = "\n".join(summary)

(RESULTS_DIR / "dense_forest_summary.txt").write_text(
    summary_text,
    encoding="utf-8"
)

# Display results
print(summary_text)
print("\nFiles saved successfully in the Results folder.")
