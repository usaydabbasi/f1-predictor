from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json
import requests
import pandas as pd
import numpy as np
import joblib
import os
import certifi

app = FastAPI(title="F1 Predictor API")

# Allow local frontend (Vite default port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Model + feature order
# ----------------------------
MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"
FEATS_PATH = Path(__file__).parent / "models" / "features.json"

model = joblib.load(MODEL_PATH)
with open(FEATS_PATH, "r") as f:
    feature_order = json.load(f)

# Ergast via Jolpica
BASE = "https://api.jolpi.ca/ergast/f1"


# ----------------------------
# Helpers (HTTP + next race)
# ----------------------------
def fetch_json(url: str) -> dict:
    # Use certifi for SSL certificate bundle
    verify_path = certifi.where()
    r = requests.get(url, timeout=30, verify=verify_path)
    r.raise_for_status()
    return r.json()

def fetch_next_race_metadata() -> dict:
    data = fetch_json(f"{BASE}/current/next.json")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return {}
    r = races[0]
    return {
        "season": int(r["season"]),
        "round": int(r["round"]),
        "raceName": r["raceName"],
        "date": r.get("date"),
        "time": r.get("time"),
        "circuit": r["Circuit"]["circuitName"],
        "location": r["Circuit"]["Location"],
    }

def infer_grid_for_next_race() -> list[dict]:
    """
    If qualifying for next race exists, return:
      [{"driver": "...", "constructor": "...", "grid": 1}, ...]
    Else return [].
    """
    data = fetch_json(f"{BASE}/current/next/qualifying.json?limit=1000")
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    quali = races[0].get("QualifyingResults", [])
    if not quali:
        return []
    out = []
    for q in quali:
        driver = f'{q["Driver"]["givenName"]} {q["Driver"]["familyName"]}'
        constructor = q["Constructor"]["name"]
        grid = int(q["position"])  # quali pos -> grid (ignores penalties)
        out.append({"driver": driver, "constructor": constructor, "grid": grid})
    return out

def fetch_current_driver_standings() -> list[dict]:
    """
    Fallback roster (drivers + team) if qualifying is not available.
    """
    data = fetch_json(f"{BASE}/current/driverStandings.json?limit=1000")
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return []
    standings = lists[0].get("DriverStandings", [])
    out = []
    for s in standings:
        d = s["Driver"]
        driver = f'{d.get("givenName","")} {d.get("familyName","")}'.strip()
        constructors = s.get("Constructors", [])
        constructor = constructors[0]["name"] if constructors else ""
        out.append({
            "driver": driver,
            "constructor": constructor,
            "standing_position": int(s.get("position", "0")),
        })
    return out

def fetch_next_entry_map() -> dict:
    """
    Canonical driver -> constructor map.
    Tries next qualifying first; falls back to standings if quali not available.
    """
    # Try qualifying
    try:
        data = fetch_json(f"{BASE}/current/next/qualifying.json?limit=1000")
        races = data["MRData"]["RaceTable"]["Races"]
        if races and races[0].get("QualifyingResults"):
            m = {}
            for q in races[0]["QualifyingResults"]:
                name = f'{q["Driver"]["givenName"]} {q["Driver"]["familyName"]}'
                m[name] = {"constructor": q["Constructor"]["name"]}
            return m
    except Exception:
        pass

    # Fallback standings
    try:
        data = fetch_json(f"{BASE}/current/driverStandings.json?limit=1000")
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if lists:
            m = {}
            for s in lists[0].get("DriverStandings", []):
                d = s["Driver"]
                name = f'{d.get("givenName","")} {d.get("familyName","")}'.strip()
                constructors = s.get("Constructors", [])
                team = constructors[0]["name"] if constructors else ""
                m[name] = {"constructor": team}
            return m
    except Exception:
        pass

    return {}


# ----------------------------
# Feature building (live)
# ----------------------------
def fetch_current_season_results_until_round(season: int, last_round_inclusive: int) -> pd.DataFrame:
    """
    Returns all driver results in the season up to last_round_inclusive.
    Columns: season, round, race, driver, constructor, position, grid, points, status
    """
    data = fetch_json(f"{BASE}/{season}/results.json?limit=1000")
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        rnd = int(race["round"])
        if rnd > last_round_inclusive:
            continue
        race_name = race["raceName"]
        for res in race["Results"]:
            driver = f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}'
            constructor = res["Constructor"]["name"]
            pos = int(res["position"])
            grid = int(res["grid"])
            points = float(res["points"])
            status = res.get("status", "")
            rows.append([season, rnd, race_name, driver, constructor, pos, grid, points, status])
    cols = ["season","round","race","driver","constructor","position","grid","points","status"]
    return pd.DataFrame(rows, columns=cols)

def status_to_dnf(status: str) -> int:
    s = str(status).lower()
    finished = ("finish" in s) or ("+" in s) or ("lap" in s)
    return 0 if finished else 1

def build_live_features_for_next_race(season: int, next_round: int, grid_rows: list[dict]) -> pd.DataFrame:
    """
    Build the numeric feature rows your model expects for each driver in grid_rows.
    grid_rows must include: driver, constructor, grid.
    """
    hist = fetch_current_season_results_until_round(season, next_round - 1)

    # If no history, default zeros
    if hist.empty:
        df = pd.DataFrame(grid_rows)
        df["drv_pts_to_date"] = 0.0
        df["drv_roll_pts_5"] = 0.0
        df["drv_roll_pos_5"] = 0.0
        df["drv_roll_dnf_5"] = 0.0
        df["con_pts_to_date"] = 0.0
        df["con_roll_pts_5"] = 0.0
        df["round_norm"] = 0.0
        df["season"] = season
        cols = feature_order + ["driver","constructor","season"]
        return df[[*cols]]

    hist = hist.sort_values(["season", "round"])
    hist["dnf"] = hist["status"].apply(status_to_dnf)

    # Driver stats (prior-only)
    g_driver = hist.groupby(["season", "driver"], sort=False)
    hist["drv_pts_to_date"] = g_driver["points"].cumsum() - hist["points"]
    hist["drv_roll_pts_5"] = g_driver["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)
    hist["drv_roll_pos_5"] = g_driver["position"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)
    hist["drv_roll_dnf_5"] = g_driver["dnf"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)

    # Team stats (prior-only)
    g_team = hist.groupby(["season", "constructor"], sort=False)
    hist["con_pts_to_date"] = g_team["points"].cumsum() - hist["points"]
    hist["con_roll_pts_5"] = g_team["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=[0,1], drop=True)

    # Round normalization for next race
    max_round = int(hist.groupby("season")["round"].max().get(season, next_round - 1))
    round_norm = (next_round / max_round) if max_round > 0 else 0.0

    # Latest per driver/team prior to next race
    last_driver = hist.sort_values("round").groupby("driver", as_index=False).tail(1)
    last_team = hist.sort_values("round").groupby("constructor", as_index=False).tail(1)

    df_grid = pd.DataFrame(grid_rows)

    df = df_grid.merge(
        last_driver[["driver","drv_pts_to_date","drv_roll_pts_5","drv_roll_pos_5","drv_roll_dnf_5"]],
        on="driver", how="left"
    )
    df = df.merge(
        last_team[["constructor","con_pts_to_date","con_roll_pts_5"]],
        on="constructor", how="left"
    )

    df["round_norm"] = round_norm
    df["season"] = season
    df = df.fillna(0.0)

    cols = feature_order + ["driver","constructor","season"]
    return df[[*cols]]


# ----------------------------
# API endpoints (for frontend)
# ----------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "features": feature_order}

@app.get("/api/next-race")
def next_race():
    meta = fetch_next_race_metadata()
    return meta if meta else {"error": "No upcoming race found"}

@app.get("/api/next-grid")
def next_grid():
    try:
        grid = infer_grid_for_next_race()
        return {"grid": grid}
    except Exception:
        return {"grid": []}

@app.get("/api/next-roster")
def next_roster():
    meta = fetch_next_race_metadata()
    if not meta:
        return {"error": "No upcoming race found", "drivers": []}
    roster = fetch_current_driver_standings()
    drivers = [{"driver": r["driver"], "constructor": r["constructor"], "standing_position": r["standing_position"]} for r in roster]
    return {"race": meta, "drivers": drivers}

@app.get("/api/predict-next-race")
def predict_next_race():
    """
    If qualifying exists: return predictions for all drivers automatically.
    """
    meta = fetch_next_race_metadata()
    if not meta:
        return {"error": "No upcoming race found"}

    season = meta["season"]
    next_round = meta["round"]

    grid = infer_grid_for_next_race()
    if not grid:
        return {
            "race": meta,
            "message": "Qualifying not available yet; use /api/next-roster and POST /api/predict-next-race-with-grid",
            "predictions": []
        }

    feats = build_live_features_for_next_race(season, next_round, grid_rows=grid)
    X = feats[feature_order].to_numpy(dtype=float)
    probs = model.predict_proba(X)[:, 1]

    out = feats[["driver", "constructor"]].copy()
    out["grid"] = [g["grid"] for g in grid]
    out["win_probability"] = np.round(probs, 3)
    out = out.sort_values("win_probability", ascending=False).reset_index(drop=True)

    return {"race": meta, "predictions": out.to_dict(orient="records")}

@app.post("/api/predict-next-race-with-grid")
def predict_next_race_with_grid(payload: dict):
    """
    Before qualifying: frontend provides only driver + grid.
    Team is auto-picked (not editable, not accepted from client).
    Body:
    {
      "grid": [
        {"driver": "Max Verstappen", "grid": 1},
        {"driver": "Charles Leclerc", "grid": 2}
      ]
    }
    """
    meta = fetch_next_race_metadata()
    if not meta:
        return {"error": "No upcoming race found"}

    season = meta["season"]
    next_round = meta["round"]

    user_rows = payload.get("grid", [])
    if not user_rows:
        return {"error": "Missing grid", "required_format": {"grid": [{"driver": "...", "grid": 1}] }}

    canon = fetch_next_entry_map()
    if not canon:
        return {"error": "Could not fetch roster; try again later"}

    validated = []
    errors = []

    for i, row in enumerate(user_rows):
        drv = str(row.get("driver", "")).strip()
        if drv not in canon:
            errors.append({"index": i, "driver": drv, "issue": "unknown_driver"})
            continue
        try:
            grid = int(row.get("grid"))
        except Exception:
            errors.append({"index": i, "driver": drv, "issue": "invalid_grid"})
            continue

        team = canon[drv]["constructor"] or ""
        validated.append({"driver": drv, "constructor": team, "grid": grid})

    if errors:
        return {"error": "validation_failed", "errors": errors}

    feats = build_live_features_for_next_race(season, next_round, grid_rows=validated)
    X = feats[feature_order].to_numpy(dtype=float)
    probs = model.predict_proba(X)[:, 1]

    out = feats[["driver", "constructor"]].copy()
    out["grid"] = [r["grid"] for r in validated]
    out["win_probability"] = np.round(probs, 3)
    out = out.sort_values("win_probability", ascending=False).reset_index(drop=True)

    return {"race": meta, "predictions": out.to_dict(orient="records")}



