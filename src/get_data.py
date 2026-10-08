"""Fetch (or verify) all raw data sources. Idempotent: skips files already present.

1. Dengue cluster snapshots (sgcharts, NEA-sourced, weekly 2015-07..2020-11):
     curl -L -o data/sgcharts.zip http://outbreak.sgcharts.com/sgcharts.zip
2. Current NEA dengue clusters (GeoJSON, daily snapshot):
     URL=$(curl -s "https://api-open.data.gov.sg/v1/public/api/datasets/d_dbfabf16158d1b0e1c420627c0819168/poll-download" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['url'])")
     curl -s "$URL" -o data/dengue_clusters_<YYYY-MM-DD>.geojson
3. URA Master Plan 2019 planning-area boundaries:
     URL=$(curl -s "https://api-open.data.gov.sg/v1/public/api/datasets/d_4765db0e87b9c86336792efe8a1f7a66/poll-download" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['url'])")
     curl -s "$URL" -o data/planning_areas.geojson
4. SingStat resident population by planning area:
     URL=$(curl -s "https://api-open.data.gov.sg/v1/public/api/datasets/d_ae8f6f7e9184361a187dd335fa04b684/poll-download" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['url'])")
     curl -s "$URL" -o data/population_by_planning_area.csv
5. Weather: fetched by src/fetch_weather.py (Open-Meteo ERA5 archive, no key).

All sources retrieved 2026-10-07. See README for licences/attribution.
"""
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

SOURCES = [
    ("sgcharts.zip",
     "http://outbreak.sgcharts.com/sgcharts.zip"),
]


def poll_download(dataset_id, out_name):
    meta = subprocess.run(
        ["curl", "-s",
         f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"],
        capture_output=True, text=True, check=True).stdout
    import json
    url = json.loads(meta)["data"]["url"]
    out = DATA / out_name
    if out.exists():
        print(f"exists, skipping: {out_name}")
        return
    subprocess.run(["curl", "-s", "-m", 180, url, "-o", str(out)], check=True)
    print(f"downloaded: {out_name}")


def main():
    DATA.mkdir(exist_ok=True)
    for name, url in SOURCES:
        out = DATA / name
        if out.exists():
            print(f"exists, skipping: {name}")
            continue
        subprocess.run(["curl", "-sL", "-m", 180, url, "-o", str(out)], check=True)
        print(f"downloaded: {name}")
    poll_download("d_dbfabf16158d1b0e1c420627c0819168", "dengue_clusters_latest.geojson")
    poll_download("d_4765db0e87b9c86336792efe8a1f7a66", "planning_areas.geojson")
    poll_download("d_ae8f6f7e9184361a187dd335fa04b684", "population_by_planning_area.csv")


if __name__ == "__main__":
    main()
