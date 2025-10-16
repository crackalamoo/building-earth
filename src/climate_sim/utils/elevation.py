from pathlib import Path
import os
from dotenv import load_dotenv
import urllib.request

import numpy as np
import rioxarray
from PIL import Image
from functools import lru_cache
import xarray as xr

def download_etopo(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_bed_elev_gtif/ETOPO_2022_v1_60s_N90W180_bed.tif"
    url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/60s/60s_surface_elev_gtif/ETOPO_2022_v1_60s_N90W180_surface.tif"
    dest = dest_dir / "etopo_60s.tif" # 60 arc-second resolution

    if dest.exists():
        print(f"File already exists at {dest}, skipping download.")
        return dest
    
    print(f"Downloading ETOPO data from {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")
    return dest

def _wrap_longitudes(lon_deg: np.ndarray) -> np.ndarray:
    """Wrap longitudes to the [-180, 180) range."""

    return ((lon_deg + 180.0) % 360.0) - 180.0


@lru_cache(maxsize=1)
def load_elevation_data(path: str | Path | None = None) -> xr.DataArray | None:
    """Return an elevation dataset registered to WGS84 coordinates."""

    if path is None:
        data_dir = os.getenv("DATA_DIR")
        if data_dir is None:
            raise ValueError("Please set the DATA_DIR environment variable.")
        data_dir = Path(data_dir)

        path = data_dir / "etopo_60s.tif"

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Elevation data file not found at {dataset_path}")

    data = rioxarray.open_rasterio(dataset_path).squeeze()
    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326", inplace=False)
    return data

def compute_cell_elevation(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    *,
    data: xr.DataArray | None = None,
    sample_method: str = "center",
) -> np.ndarray:
    """Return the elevation (m) for the provided grid cells."""

    if lon2d.shape != lat2d.shape:
        raise ValueError("Longitude and latitude grids must share the same shape")

    lon_array = np.asarray(lon2d, dtype=float)
    lat_array = np.asarray(lat2d, dtype=float)

    dataset = data if data is not None else load_elevation_data()
    if dataset is None:
        return np.zeros_like(lon_array)

    if sample_method != "center":
        raise ValueError("Only 'center' sampling is currently supported")

    if dataset.rio.crs is None:
        dataset = dataset.rio.write_crs("EPSG:4326", inplace=False)

    lon_wrapped = _wrap_longitudes(lon_array)
    lat_clamped = np.clip(lat_array, -90.0, 90.0)

    lon_flat = lon_wrapped.ravel()
    lat_flat = lat_clamped.ravel()

    # Sample the dataset at the cell centre using bilinear interpolation when possible.
    coords = {
        "x": ("points", lon_flat),
        "y": ("points", lat_flat),
    }

    try:
        sampled = dataset.interp(
            x=coords["x"], y=coords["y"], method="linear", kwargs={"fill_value": None}
        )
    except TypeError:
        sampled = dataset.interp(x=coords["x"], y=coords["y"], method="linear")

    values = np.asarray(sampled.values, dtype=float).reshape(lon_array.shape)

    if np.any(~np.isfinite(values)):
        sampled_nearest = dataset.interp(
            x=coords["x"], y=coords["y"], method="nearest"
        )
        nearest_values = np.asarray(sampled_nearest.values, dtype=float).reshape(
            lon_array.shape
        )
        mask = ~np.isfinite(values)
        values[mask] = nearest_values[mask]

    return np.nan_to_num(values, nan=0.0)


def pressure_from_temperature_elevation(
    temperature_K: np.ndarray,
    elevation_m: np.ndarray,
    *,
    sea_level_pressure_pa: float = 101_325.0,
    gravity_m_s2: float = 9.81,
    gas_constant_J_kgK: float = 287.0,
) -> np.ndarray:
    """Compute surface pressure (Pa) from temperature and elevation using a hydrostatic profile."""

    if temperature_K.shape != elevation_m.shape:
        raise ValueError("Temperature and elevation fields must share the same shape")

    temp_safe = np.maximum(np.asarray(temperature_K, dtype=float), 1.0)
    elev = np.asarray(elevation_m, dtype=float)

    exponent = -gravity_m_s2 * elev / (gas_constant_J_kgK * temp_safe)
    pressure = sea_level_pressure_pa * np.exp(exponent)
    return pressure

def write_elevation_png_1deg(tif_path: Path, out_dir: Path | None = None, resolution: float = 0.1, contrast_water: bool = True) -> Path:
    if out_dir is None:
        out_dir = tif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and ensure geographic CRS
    da = rioxarray.open_rasterio(tif_path).squeeze()
    if da.rio.crs is None:
        # Assume geographic WGS84 if not encoded
        da = da.rio.write_crs("EPSG:4326", inplace=False)

    # Target 1° grid bounds: use the data bounds from the raster itself
    # Build a 1-degree regular grid in x (lon) and y (lat)
    xmin, ymin, xmax, ymax = da.rio.bounds()
    # Clamp to plausible geographic bounds
    xmin = max(-180.0, xmin)
    xmax = min(180.0, xmax)
    ymin = max(-90.0, ymin)
    ymax = min(90.0, ymax)

    lons = np.arange(np.floor(xmin), np.ceil(xmax) + 1e-9, resolution)
    lats = np.arange(np.floor(ymin), np.ceil(ymax) + 1e-9, resolution)
    # Use cell centers for resampling (shift by 0.5 degree)
    lons_centers = lons[:-1] + resolution / 2
    lats_centers = lats[:-1] + resolution / 2

    # xarray wants coordinate arrays assigned to the DataArray dims
    # Ensure dimensions are named x/y
    da = da.rename({da.dims[-1]: "x", da.dims[-2]: "y"})

    # Reproject to itself (to be safe) and resample by nearest or average.
    # Use rio.reproject to a coarser resolution grid by specifying transform/coords.
    da_coarse = da.interp(
        x=("x", lons_centers), y=("y", lats_centers), method="linear"
    )

    data = da_coarse.values.astype(np.float64)
    # Compute min/max excluding NaNs
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Elevation array contains no finite values after resampling.")
    
    vmin = float(data[finite].min())
    vmax = float(data[finite].max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    norm = (data - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    img_arr = (norm * 255.0).astype(np.uint8)

    # make color if contrast water (green for land, blue for water)
    if contrast_water:
        img_arr_color = np.zeros((img_arr.shape[0], img_arr.shape[1], 3), dtype=np.uint8)
        # Land: greenish
        land = data >= 0.0
        water = data < 0.0
        img_arr_color[..., 0][land] = np.floor(img_arr[land] * 0.7)  # R
        img_arr_color[..., 1][land] = img_arr[land]       # G
        img_arr_color[..., 2][land] = np.floor(img_arr[land] * 0.7)  # B
        # Water: bluish
        img_arr_color[..., 0][water] = np.floor(img_arr[water] * 0.7)    # R
        img_arr_color[..., 1][water] = np.floor(img_arr[water] * 0.7)    # G
        img_arr_color[..., 2][water] = img_arr[water]  # B
        img_arr_color = np.flipud(img_arr_color)
        img = Image.fromarray(img_arr_color)
    else:
        # y increases upward in data, but image origin is top-left; flip vertically
        img_arr = np.flipud(img_arr)
        img = Image.fromarray(img_arr)

    out_path = out_dir / f"elevation_{resolution}deg.png"
    img.save(out_path)
    return out_path

if __name__ == "__main__":
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    if data_dir is None:
        raise ValueError("Please set the DATA_DIR environment variable.")
    data_dir = Path(data_dir)

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

    out = write_elevation_png_1deg(etopo_path)
    print(f"Wrote: {out}")
