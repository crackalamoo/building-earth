import rioxarray
from pathlib import Path
import urllib.request

def download_etopo(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif"
    dest = dest_dir / "etopo_60s.tif" # 60 arc-second resolution

    if dest.exists():
        print(f"File already exists at {dest}, skipping download.")
        return dest
    
    print(f"Downloading ETOPO data from {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")
    return dest

if __name__ == "__main__":
    data_dir = Path("./data")
    print(data_dir)
    etopo_path = download_etopo(data_dir)

    # Load the ETOPO data using rioxarray
    etopo_ds = rioxarray.open_rasterio(etopo_path)
    print(etopo_ds)

    # Example: Access elevation data
    elevation_data = etopo_ds[0]  # Assuming single band
    print(elevation_data)
    print(f"Elevation at (29 N, 86 E): {elevation_data.sel(y=29, x=86, method='nearest').values} meters")
    print(f"Elevation at (5 N, 86 E): {elevation_data.sel(y=5, x=86, method='nearest').values} meters")
