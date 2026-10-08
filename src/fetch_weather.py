"""Fetch daily weather for Singapore from Open-Meteo ERA5 archive and aggregate to ISO weeks.

Why Open-Meteo / ERA5 instead of MSS (data.gov.sg)?
The MSS v1 environment API (api.data.gov.sg/v1/environment/rainfall) returns
no data before 2018, but the dengue cluster series starts July 2015. ERA5
reanalysis via the free Open-Meteo archive API covers the full window with no
API key. Singapore is small enough (~50 km) that island-wide weekly weather is
the standard unit in published dengue work (e.g. NEA's own models).
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "weather_weekly.csv"

# 4 grid points spanning the island; averaged to one island-wide series.
GRID = [(1.42, 103.70), (1.42, 103.95), (1.28, 103.70), (1.28, 103.95)]
START, END = "2015-06-01", "2020-12-31"


def fetch_point(lat, lon):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START}&end_date={END}"
        "&daily=temperature_2m_mean,precipitation_sum"
        "&timezone=Asia%2FSingapore"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "dengue-risk-research/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)["daily"]
    return pd.DataFrame(
        {"date": pd.to_datetime(d["time"]),
         "tmean": d["temperature_2m_mean"],
         "precip": d["precipitation_sum"]}
    )


def main():
    frames = []
    for lat, lon in GRID:
        df = fetch_point(lat, lon)
        df["lat"], df["lon"] = lat, lon
        frames.append(df)
        print(f"fetched {len(df)} days @ ({lat},{lon})")
    daily = pd.concat(frames)
    island = daily.groupby("date").agg(tmean_c=("tmean", "mean"),
                                       precip_mm=("precip", "mean")).reset_index()
    island["iso_year"] = island["date"].dt.isocalendar().year.astype(int)
    island["iso_week"] = island["date"].dt.isocalendar().week.astype(int)
    weekly = (island.groupby(["iso_year", "iso_week"])
              .agg(week_start=("date", "min"),
                   temp_mean_c=("tmean_c", "mean"),
                   rain_total_mm=("precip_mm", "sum"))
              .reset_index())
    weekly.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(weekly)} weeks, "
          f"{weekly['week_start'].min().date()}..{weekly['week_start'].max().date()})")


if __name__ == "__main__":
    main()
