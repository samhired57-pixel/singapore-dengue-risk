"""Build the modeling table: planning-area x week with lagged features.

Target: has_cluster_next = 1 if the area has >=1 active cluster location
in week t+1 (shifted within each planning area; last week per area dropped).

Features at week t (nothing from the future):
- weather lags: island-wide weekly rainfall total / mean temperature,
  lagged 1/2/4/8 weeks (dengue responds to weather with a delay:
  rainfall -> breeding sites, ~1-2 wk mosquito development +
  extrinsic incubation -> cases)
- prior activity: has_cluster, n_locations, total_cases at lags 1/2/4
- static: population density, area size
- seasonality: cyclical month encoding
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def add_lags(df, cols, lags, group="PLN_AREA_N"):
    df = df.sort_values([group, "iso_year", "iso_week"]).copy()
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df.groupby(group)[c].shift(L)
    return df


def main():
    panel = pd.read_csv(DATA / "panel_weekly.csv")
    # week_date from the ISO calendar (area-weeks without clusters have no
    # snapshot date, so derive the week start deterministically)
    panel["week_date"] = pd.to_datetime(
        panel["iso_year"].astype(str) + "-W" +
        panel["iso_week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u")
    panel = panel.sort_values(["PLN_AREA_N", "week_date"]).reset_index(drop=True)

    weather = pd.read_csv(DATA / "weather_weekly.csv")
    weather["week_start"] = pd.to_datetime(weather["week_start"])
    panel = panel.merge(weather[["iso_year", "iso_week", "temp_mean_c", "rain_total_mm"]],
                        on=["iso_year", "iso_week"], how="left")
    assert panel[["temp_mean_c", "rain_total_mm"]].notna().all().all()

    # target: next week's status within each area
    panel["has_cluster_next"] = panel.groupby("PLN_AREA_N")["has_cluster"].shift(-1)
    panel = panel.dropna(subset=["has_cluster_next"])
    panel["has_cluster_next"] = panel["has_cluster_next"].astype(int)

    panel = add_lags(panel, ["temp_mean_c", "rain_total_mm"], [1, 2, 4, 8])
    panel = add_lags(panel, ["has_cluster", "n_locations", "total_cases"], [1, 2, 4])

    areas = pd.read_csv(DATA / "areas.csv")
    panel = panel.merge(areas[["PLN_AREA_N", "area_km2", "pop_density_km2"]],
                        on="PLN_AREA_N", how="left")
    panel["log_popdens"] = np.log1p(panel["pop_density_km2"])

    month = panel["week_date"].dt.month
    panel["month_sin"] = np.sin(2 * np.pi * month / 12)
    panel["month_cos"] = np.cos(2 * np.pi * month / 12)

    panel = panel.dropna().reset_index(drop=True)
    panel.to_csv(DATA / "modeling.csv", index=False)
    print(f"modeling rows: {len(panel)} | positives: {panel['has_cluster_next'].mean():.1%} "
          f"| weeks {panel['week_date'].min().date()}..{panel['week_date'].max().date()}")
    feats = [c for c in panel.columns if "_lag" in c or c in
             ("log_popdens", "area_km2", "month_sin", "month_cos")]
    print(f"features: {len(feats)}")


if __name__ == "__main__":
    main()
