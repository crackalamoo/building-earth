"""Download Natural Earth populated places and export compact binary for frontend city lights."""

import struct
import urllib.request
import json
from pathlib import Path


GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_populated_places_simple.geojson"
)

OUTPUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "data" / "cities.bin"


def main() -> None:
    print(f"Downloading populated places from Natural Earth...")
    with urllib.request.urlopen(GEOJSON_URL) as resp:
        data = json.loads(resp.read())

    features = data["features"]
    cities: list[tuple[float, float, float]] = []
    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]
        pop = props.get("pop_max") or props.get("pop_min") or 0
        if pop <= 0:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        cities.append((lon, lat, float(pop)))

    print(f"Extracted {len(cities)} cities with population > 0")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "wb") as fp:
        fp.write(struct.pack("<I", len(cities)))
        for lon, lat, pop in cities:
            fp.write(struct.pack("<fff", lon, lat, pop))

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
