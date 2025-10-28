import pandas as pd
from pathlib import Path

SRC = Path("backend/data/results.csv")
OUT = Path("backend/data/features.csv")

df = pd.read_csv(SRC)
df.sort_values(["season", "round"], inplace=True)

# Target variable: did the driver win?
df["win"] = (df["position"] == 1).astype(int)

# Build a DNF (Did Not Finish) flag safely
status_lower = df["status"].astype(str).str.lower()

# Mark as finished if contains "finish", "+N laps", or "lap"/"lapped"
finished_mask = (
    status_lower.str.contains("finish", regex=False)
    | status_lower.str.contains("+", regex=False)   # literal plus sign
    | status_lower.str.contains("lap", regex=False)
)

df["dnf"] = (~finished_mask).astype(int)

# Rolling and cumulative driver stats
g_driver = df.groupby(["season", "driver"], sort=False)
df["drv_pts_to_date"]  = g_driver["points"].cumsum() - df["points"]
df["drv_roll_pts_5"]   = g_driver["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)
df["drv_roll_pos_5"]   = g_driver["position"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)
df["drv_roll_dnf_5"]   = g_driver["dnf"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)

# Rolling and cumulative constructor stats
g_team = df.groupby(["season", "constructor"], sort=False)
df["con_pts_to_date"]  = g_team["points"].cumsum() - df["points"]
df["con_roll_pts_5"]   = g_team["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)

# Context features
df["grid"] = df["grid"].fillna(df["grid"].median())
df["round_norm"] = df["round"] / df.groupby("season")["round"].transform("max")

feat_cols = [
    "grid",
    "drv_pts_to_date", "drv_roll_pts_5", "drv_roll_pos_5", "drv_roll_dnf_5",
    "con_pts_to_date", "con_roll_pts_5",
    "round_norm",
]

features = df[feat_cols + ["win", "season"]].fillna(0)
features.to_csv(OUT, index=False)
print(f"Saved {OUT} with {len(features)} rows and {len(feat_cols)} features.")
