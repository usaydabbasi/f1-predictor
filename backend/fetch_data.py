from datetime import datetime
import requests
import pandas as pd

BASE = "https://api.jolpi.ca/ergast/f1"

def get_results(season: int) -> pd.DataFrame:
    url = f"{BASE}/{season}/results.json?limit=1000"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    races = r.json()["MRData"]["RaceTable"]["Races"]
    rows = []
    for race in races:
        round_no = int(race["round"])
        race_name = race["raceName"]
        for res in race["Results"]:
            driver = res["Driver"]["familyName"]
            constructor = res["Constructor"]["name"]
            pos = int(res["position"])
            grid = int(res["grid"])
            points = float(res["points"])
            status = res.get("status", "")
            rows.append([
                season, round_no, race_name, driver, constructor,
                pos, grid, points, status
            ])
    return pd.DataFrame(rows, columns=[
        "season", "round", "race", "driver", "constructor",
        "position", "grid", "points", "status"
    ])

if __name__ == "__main__":
    start_year = 2018
    current_year = datetime.utcnow().year
    years = list(range(start_year, current_year + 1))

    frames = []
    for y in years:
        print(f"Fetching {y} …")
        try:
            df_y = get_results(y)
        except requests.HTTPError as e:
            print(f"Skipping {y}: {e}")
            continue
        if not df_y.empty:
            frames.append(df_y)

    if frames:
        df_all = pd.concat(frames, ignore_index=True)
        df_all.to_csv("backend/data/results.csv", index=False)
        print("Saved backend/data/results.csv")
    else:
        print("No data was fetched.")

