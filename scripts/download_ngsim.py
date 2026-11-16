"""Download NGSIM US-101 trajectories (main lanes 1-5, 10 Hz) from data.transportation.gov.

Writes one CSV per page into OUT_DIR so an interrupted download resumes where it stopped.

    python scripts/download_ngsim.py --out data/raw/us101 --count-only
    python scripts/download_ngsim.py --out data/raw/us101
"""

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.transportation.gov/resource/8ect-6jqj.csv"
COLUMNS = ["vehicle_id", "frame_id", "total_frames", "global_time", "local_y",
           "v_length", "v_class", "v_vel", "lane_id", "preceding"]
WHERE = "location='us-101' AND lane_id >= 1 AND lane_id <= 5"
HEADERS = {"User-Agent": "DSE511-capstone-car-following/1.0 (educational use)"}
PAGE_SIZE = 200_000
DELAY_S = 1.0


def get(params: dict) -> requests.Response:
    for attempt in range(3):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=300)
            r.raise_for_status()
            return r
        except requests.RequestException as error:
            if attempt == 2:
                raise
            print(f"  request failed ({error}); retrying in 10 s")
            time.sleep(10)


def count_rows() -> int:
    r = get({"$select": "count(*) AS n", "$where": WHERE})
    return int(pd.read_csv(io.StringIO(r.text))["n"].iloc[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()

    total = count_rows()
    print(f"rows to download: {total:,}")
    if args.count_only:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    n_pages = -(-total // PAGE_SIZE)
    for page in range(n_pages):
        path = args.out / f"page_{page:03d}.csv"
        if path.exists():
            continue
        start = time.time()
        r = get({"$select": ",".join(COLUMNS), "$where": WHERE, "$order": ":id",
                 "$limit": PAGE_SIZE, "$offset": page * PAGE_SIZE})
        tmp = path.with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(path)
        print(f"page {page + 1}/{n_pages}: {len(r.content) / 1e6:.1f} MB in {time.time() - start:.0f} s")
        time.sleep(DELAY_S)

    rows = sum(len(pd.read_csv(p, usecols=["vehicle_id"])) for p in sorted(args.out.glob("page_*.csv")))
    print(f"downloaded rows: {rows:,} (expected {total:,})")
    if rows != total:
        raise SystemExit("row count mismatch; delete nothing, rerun to fetch missing pages")


if __name__ == "__main__":
    main()
