"""Download HYG v4 star catalog and extract naked-eye stars for the frontend.

Data source: HYG Database (astronexus/HYG-Database), CC-BY-SA 4.0
https://github.com/astronexus/HYG-Database
"""

import csv
import io
import json
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/master/hyg/CURRENT/hygdata_v41.csv"
MAG_LIMIT = 5.0
OUTPUT = Path(__file__).resolve().parent.parent / "public" / "data" / "stars.json"


def main() -> None:
    print(f"Downloading HYG catalog from {URL}...")
    with urllib.request.urlopen(URL) as resp:
        text = resp.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    stars: list[list[float]] = []
    for row in reader:
        try:
            mag = float(row["mag"])
        except (ValueError, KeyError):
            continue
        if mag > MAG_LIMIT:
            continue
        try:
            ra = float(row["ra"])
            dec = float(row["dec"])
        except (ValueError, KeyError):
            continue
        try:
            ci = float(row["ci"]) if row.get("ci") else 0.6
        except ValueError:
            ci = 0.6
        stars.append([round(ra, 4), round(dec, 4), round(mag, 2), round(ci, 2)])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(stars, f, separators=(",", ":"))

    print(f"Wrote {len(stars)} stars to {OUTPUT} ({OUTPUT.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
