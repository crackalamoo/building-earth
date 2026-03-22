# type: ignore[attr-defined]
"""Shared utilities for binary export."""

import json
from pathlib import Path

import numpy as np

from climate_sim.core.grid import create_lat_lon_grid
from climate_sim.export.temperature_interpolation import interpolate_layer_map
from climate_sim.export.orographic_interpolation import recompute_fields_at_1deg
from climate_sim.data.elevation import compute_cell_elevation
from climate_sim.data.landmask import compute_land_mask


def _fill_island_vegetation(
    veg: np.ndarray,
    native_layers: dict[str, np.ndarray],
    native_land_mask: np.ndarray,
    fine_land_mask: np.ndarray,
) -> np.ndarray:
    """Recompute vegetation for coarse ocean cells that contain fine-grid land."""
    nlat, nlon = native_land_mask.shape
    fine_nlat, fine_nlon = fine_land_mask.shape
    lat_ratio = fine_nlat // nlat
    lon_ratio = fine_nlon // nlon

    precip = native_layers["precipitation"]
    surface = native_layers["surface"]

    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_seconds = np.array([d * 86400.0 for d in days_per_month])

    p_min, p_max = 50.0, 1000.0
    growing_thresh = 5.0
    full_months = 5.0

    for i in range(nlat):
        for j in range(nlon):
            if native_land_mask[i, j]:
                continue
            fi0 = i * lat_ratio
            fi1 = min((i + 1) * lat_ratio, fine_nlat)
            fj0 = j * lon_ratio
            fj1 = min((j + 1) * lon_ratio, fine_nlon)
            if not np.any(fine_land_mask[fi0:fi1, fj0:fj1]):
                continue
            annual_precip = float(np.sum(precip[:, i, j] * month_seconds))
            u = np.clip((annual_precip - p_min) / (p_max - p_min), 0.0, 1.0)
            veg_frac = u ** 0.6
            warm_months = float(np.sum(surface[:, i, j] > growing_thresh))
            gs_u = np.clip(warm_months / full_months, 0.0, 1.0)
            gs_cap = gs_u * gs_u * (3.0 - 2.0 * gs_u)
            veg_frac = min(veg_frac, gs_cap, 0.95)
            for m in range(12):
                veg[m, i, j] = veg_frac

    return veg


def write_binary_export(
    output_dir: Path,
    layers: dict[str, np.ndarray],
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    interpolate: bool,
    *,
    file_prefix: str = "main",
    fields_filter: list[str] | None = None,
    quantization_min: float = -60.0,
    quantization_max: float = 60.0,
) -> None:
    """Write {prefix}.bin + {prefix}.manifest.json to output_dir.

    Args:
        output_dir: Directory to write files to.
        layers: Dict of field name -> numpy array.
        lon2d, lat2d: Grid coordinates.
        interpolate: Whether to interpolate to higher resolution.
        file_prefix: Filename prefix (default "main").
        fields_filter: If set, only include these field names. None = all.
        quantization_min: Min value for uint8 temperature quantization.
        quantization_max: Max value for uint8 temperature quantization.
    """
    quant_range = quantization_max - quantization_min

    # Keep native-resolution copies before interpolation
    native_layers = dict(layers)

    # Interpolate temperature_2m if requested
    if interpolate:
        output_resolution = 0.25
        print(f"Interpolating temperature fields to {output_resolution}° resolution...")

        interp_layers: dict[str, np.ndarray] = {}
        if "surface" in layers:
            interp_layers["surface"] = layers["surface"]
        if "temperature_2m" in layers:
            interp_layers["temperature_2m"] = layers["temperature_2m"]

        _, _, interpolated = interpolate_layer_map(
            interp_layers,
            lon2d,
            lat2d,
            output_resolution_deg=output_resolution,
            apply_lapse_rate_to_2m=True,
            compute_snow_temperature=True,
        )
        print("Interpolation complete.")

        # Recompute precipitation, humidity, soil_moisture at 1° with orographic detail
        # Requires boundary_layer temperature (only available with atmosphere enabled)
        if "precipitation" in native_layers and "boundary_layer" in native_layers:
            print("Recomputing precipitation at 1° with orographic detail...")
            fine_fields = recompute_fields_at_1deg(native_layers, lon2d, lat2d)
        else:
            fine_fields = {}
    else:
        interpolated = {}
        fine_fields = {}

    # Compute land masks
    native_land_mask = compute_land_mask(lon2d, lat2d).astype(np.uint8)
    if interpolate:
        fine_lon2d, fine_lat2d = create_lat_lon_grid(output_resolution)
        land_mask = compute_land_mask(fine_lon2d, fine_lat2d).astype(np.uint8)
    else:
        land_mask = native_land_mask

    def should_include(name: str) -> bool:
        if fields_filter is None:
            return True
        return name in fields_filter

    # Assemble fields for export
    fields: list[tuple[str, np.ndarray, str]] = []

    # temperature_2m: quantized to uint8
    if should_include("temperature_2m"):
        t2m_src = None
        t2m_label = ""
        if "temperature_2m" in interpolated:
            t2m_src = interpolated["temperature_2m"]
            t2m_label = "interpolated"
        elif "temperature_2m" in native_layers:
            t2m_src = native_layers["temperature_2m"]
            t2m_label = "native"
        elif "surface" in native_layers:
            t2m_src = native_layers["surface"]
            t2m_label = "from surface"
        if t2m_src is not None:
            t2m_u8 = np.clip(
                (t2m_src - quantization_min) * (255.0 / quant_range), 0, 255
            ).astype(np.uint8)
            fields.append(("temperature_2m", t2m_u8, "uint8"))
            print(f"  temperature_2m: {t2m_u8.shape} ({t2m_label}, uint8, range [{quantization_min}, {quantization_max}])")

    # surface: always native resolution
    if should_include("surface") and "surface" in native_layers:
        fields.append(("surface", native_layers["surface"], "float16"))
        print(f"  surface: {native_layers['surface'].shape}")

    # land_mask
    if should_include("land_mask"):
        fields.append(("land_mask", land_mask, "uint8"))
        print(f"  land_mask: {land_mask.shape}")

    # land_mask_native
    if should_include("land_mask_native") and interpolate and native_land_mask is not land_mask:
        fields.append(("land_mask_native", native_land_mask, "uint8"))
        print(f"  land_mask_native: {native_land_mask.shape}")

    # land_mask_1deg
    if should_include("land_mask_1deg") and interpolate and fine_fields:
        lon1deg, lat1deg = create_lat_lon_grid(1.0)
        land_mask_1deg = compute_land_mask(lon1deg, lat1deg).astype(np.uint8)
        fields.append(("land_mask_1deg", land_mask_1deg, "uint8"))
        print(f"  land_mask_1deg: {land_mask_1deg.shape}")

    # vegetation_fraction
    if should_include("vegetation_fraction") and "vegetation_fraction" in native_layers:
        veg = native_layers["vegetation_fraction"].copy()
        if interpolate and "precipitation" in native_layers and "surface" in native_layers:
            veg = _fill_island_vegetation(
                veg, native_layers, native_land_mask, land_mask,
            )
        fields.append(("vegetation_fraction", veg, "float16"))
        print(f"  vegetation_fraction: {veg.shape}")

    # precipitation
    if should_include("precipitation"):
        if "precipitation" in fine_fields:
            fields.append(("precipitation", fine_fields["precipitation"], "float16"))
            print(f"  precipitation: {fine_fields['precipitation'].shape} (1° orographic)")
        elif "precipitation" in native_layers:
            fields.append(("precipitation", native_layers["precipitation"], "float16"))
            print(f"  precipitation: {native_layers['precipitation'].shape} (native)")

    # humidity
    if should_include("humidity"):
        if "humidity" in fine_fields:
            fields.append(("humidity", fine_fields["humidity"], "float16"))
            print(f"  humidity: {fine_fields['humidity'].shape} (1° interpolated)")
        elif "humidity" in native_layers:
            fields.append(("humidity", native_layers["humidity"], "float16"))
            print(f"  humidity: {native_layers['humidity'].shape} (native)")

    # soil_moisture
    if should_include("soil_moisture"):
        if "soil_moisture" in fine_fields:
            fields.append(("soil_moisture", fine_fields["soil_moisture"], "float16"))
            print(f"  soil_moisture: {fine_fields['soil_moisture'].shape} (1° orographic)")
        elif "soil_moisture" in native_layers:
            fields.append(("soil_moisture", native_layers["soil_moisture"], "float16"))
            print(f"  soil_moisture: {native_layers['soil_moisture'].shape} (native)")

    # cloud_fraction
    if should_include("cloud_fraction"):
        conv = native_layers.get("convective_cloud_frac")
        strat = native_layers.get("stratiform_cloud_frac")
        marine = native_layers.get("marine_sc_cloud_frac")
        high = native_layers.get("high_cloud_frac")
        if all(x is not None for x in [conv, strat, marine, high]):
            low = 1 - (1 - conv) * (1 - strat) * (1 - marine)
            total_cloud = np.clip(low + high * (1 - low), 0, 1)
            fields.append(("cloud_fraction", total_cloud, "float16"))
            print(f"  cloud_fraction: {total_cloud.shape}")

            if should_include("cloud_high"):
                fields.append(("cloud_high", high, "float16"))
                print(f"  cloud_high: {high.shape}")
            if should_include("cloud_low"):
                low_nonconv = np.clip(1 - (1 - strat) * (1 - marine), 0, 1)
                fields.append(("cloud_low", low_nonconv, "float16"))
                print(f"  cloud_low: {low_nonconv.shape}")
            if should_include("cloud_convective"):
                fields.append(("cloud_convective", conv, "float16"))
                print(f"  cloud_convective: {conv.shape}")

    # Elevation
    if should_include("elevation"):
        if interpolate:
            elevation = compute_cell_elevation(fine_lon2d, fine_lat2d, cache=False)
        else:
            elevation = compute_cell_elevation(lon2d, lat2d, cache=False)
        fields.append(("elevation", elevation, "float16"))
        print(f"  elevation: {elevation.shape}")

    # Snow temperature
    if should_include("snow_temperature") and "snow_temperature" in interpolated:
        snow_temp = interpolated["snow_temperature"]
        snow_temp_u8 = np.clip((snow_temp + 60.0) * (255.0 / 120.0), 0, 255).astype(np.uint8)
        fields.append(("snow_temperature", snow_temp_u8, "uint8"))
        print(f"  snow_temperature: {snow_temp_u8.shape}")

    # Wind fields
    for wind_key in ("wind_u_10m", "wind_v_10m", "wind_speed_10m"):
        if should_include(wind_key) and wind_key in native_layers:
            fields.append((wind_key, native_layers[wind_key], "float16"))
            print(f"  {wind_key}: {native_layers[wind_key].shape}")

    # Build binary blob and manifest
    manifest: dict = {"fields": []}
    # Add quantization range to manifest so frontend decoder knows the range
    manifest["quantization_min"] = quantization_min
    manifest["quantization_max"] = quantization_max

    blobs: list[bytes] = []
    offset = 0

    for name, arr, dtype_str in fields:
        if dtype_str == "float16":
            encoded = arr.astype(np.float16)
        elif dtype_str == "uint8":
            encoded = arr.astype(np.uint8)
        else:
            encoded = arr.astype(np.float32)

        raw = encoded.tobytes()
        manifest["fields"].append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": dtype_str,
            "offset": offset,
            "bytes": len(raw),
        })
        blobs.append(raw)
        offset += len(raw)

    # Write files
    bin_path = output_dir / f"{file_prefix}.bin"
    manifest_path = output_dir / f"{file_prefix}.manifest.json"

    with open(bin_path, "wb") as f:
        for blob in blobs:
            f.write(blob)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    bin_size = bin_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {bin_path} ({bin_size:.2f} MB)")
    print(f"Wrote {manifest_path}")


def write_landmask_1deg(output_dir: Path) -> None:
    """Write standalone 1° land mask for fast initial load (primordial globe)."""
    lon1d, lat1d = create_lat_lon_grid(1.0)
    lm1deg = compute_land_mask(lon1d, lat1d).astype(np.uint8)
    lm_path = output_dir / "landmask1deg.bin"
    with open(lm_path, "wb") as f:
        f.write(lm1deg.tobytes())
    print(f"Wrote {lm_path} ({lm_path.stat().st_size / 1024:.1f} KB, {lm1deg.shape})")
