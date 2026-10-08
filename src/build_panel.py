"""Build planning-area x week panel from sgcharts dengue cluster snapshots.

Input:  data/sgcharts.zip (256 weekly snapshots, 2015-07-03..2020-11-06,
         each row = one cluster location; schema documented on
         outbreak.sgcharts.com/data — see README)
         data/planning_areas.geojson (URA planning areas)
         data/population_by_planning_area.csv (SingStat population)

Output: data/panel_weekly.csv  (one row per planning area x week)
        data/areas.csv         (planning-area static attributes)

Label:  has_cluster = 1 if >=1 active cluster location in the area that week.
Weeks with no snapshot at all are dropped (label unknown) -- documented.
"""
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
COLS = ["n_cases", "address", "lat", "lon", "cluster_no",
        "recent_cases", "total_cases", "yymmdd", "month"]


def parse_snapshots():
    z = zipfile.ZipFile(DATA / "sgcharts.zip")
    names = sorted(n for n in z.namelist()
                   if n.startswith("sgcharts/csv/") and n.endswith(".csv"))
    frames = []
    for n in names:
        s = z.read(n).decode("utf-8", errors="replace")
        rows = [ln.split(",", 8) for ln in s.splitlines() if ln.strip()]
        df = pd.DataFrame(rows, columns=COLS)
        df["snapshot"] = n.split("/")[-1].replace("-clusters.csv", "")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    for c in ["n_cases", "lat", "lon", "recent_cases", "total_cases", "month"]:
        allp[c] = pd.to_numeric(allp[c], errors="coerce")
    allp["date"] = pd.to_datetime(allp["yymmdd"], format="%y%m%d", errors="coerce")
    allp = allp.dropna(subset=["date", "lat", "lon"])
    allp = allp[(allp["lat"].between(1.15, 1.48)) & (allp["lon"].between(103.6, 104.12))]
    iso = allp["date"].dt.isocalendar()
    allp["iso_year"] = iso.year.astype(int)
    allp["iso_week"] = iso.week.astype(int)
    return allp


def main():
    pts = parse_snapshots()
    print(f"parsed {len(pts)} cluster-location records from "
          f"{pts['snapshot'].nunique()} snapshots "
          f"({pts['date'].min().date()}..{pts['date'].max().date()})")

    areas = gpd.read_file(DATA / "planning_areas.geojson")
    areas = areas[["PLN_AREA_N", "REGION_N", "geometry"]].copy()
    areas["area_km2"] = areas.to_crs(epsg=3414).geometry.area / 1e6

    gpts = gpd.GeoDataFrame(
        pts, geometry=[Point(xy) for xy in zip(pts["lon"], pts["lat"])], crs="EPSG:4326")
    joined = gpd.sjoin(gpts, areas[["PLN_AREA_N", "geometry"]],
                       how="left", predicate="within")
    unmatched = joined["PLN_AREA_N"].isna().sum()
    print(f"points outside all planning areas: {unmatched} "
          f"({unmatched/len(joined):.1%})")
    joined = joined.dropna(subset=["PLN_AREA_N"])

    # one snapshot per ISO week: keep the latest snapshot date in each week
    joined = joined.sort_values("date")
    weekly = (joined.groupby(["PLN_AREA_N", "iso_year", "iso_week"], as_index=False)
              .agg(n_locations=("address", "size"),
                   total_cases=("n_cases", "sum"),
                   week_date=("date", "max")))

    # complete panel over weeks that have at least one snapshot
    weeks = weekly[["iso_year", "iso_week"]].drop_duplicates()
    print(f"weeks with snapshots: {len(weeks)}")
    panel = (areas[["PLN_AREA_N"]].merge(weeks, how="cross")
             .merge(weekly, on=["PLN_AREA_N", "iso_year", "iso_week"], how="left"))
    panel["n_locations"] = panel["n_locations"].fillna(0).astype(int)
    panel["total_cases"] = panel["total_cases"].fillna(0).astype(int)
    panel["has_cluster"] = (panel["n_locations"] > 0).astype(int)

    # static attributes: population density
    pop = pd.read_csv(DATA / "population_by_planning_area.csv")
    pop_rows = pop[pop.iloc[:, 0].str.endswith("- Total", na=False)].copy()
    pop_rows["area_key"] = (pop_rows.iloc[:, 0].str.replace("- Total", "", regex=False)
                            .str.strip().str.upper())
    pop_map = dict(zip(pop_rows["area_key"],
                       pd.to_numeric(pop_rows.iloc[:, 1], errors="coerce")))
    areas["pop_2020"] = areas["PLN_AREA_N"].map(pop_map)
    # 6 non-residential areas (islands, water catchment, industrial) report
    # 'na' in SingStat -- 0 resident population is the honest value.
    areas["pop_2020"] = areas["pop_2020"].fillna(0)
    matched = (areas["pop_2020"] > 0).sum()
    print(f"planning areas matched to population: {matched}/{len(areas)}")
    areas["pop_density_km2"] = areas["pop_2020"] / areas["area_km2"]

    areas_out = areas[["PLN_AREA_N", "REGION_N", "area_km2",
                       "pop_2020", "pop_density_km2"]].copy()
    areas_out.to_csv(DATA / "areas.csv", index=False)
    panel.to_csv(DATA / "panel_weekly.csv", index=False)
    print(f"wrote panel: {len(panel)} rows, "
          f"positives: {panel['has_cluster'].sum()} "
          f"({panel['has_cluster'].mean():.1%})")


if __name__ == "__main__":
    main()
