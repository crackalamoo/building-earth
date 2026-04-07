# type: ignore[attr-defined]
"""Downsample exported binary climate data for mobile delivery.

Reads a high-resolution {prefix}.bin.gz + {prefix}.manifest.json pair
and writes a coarser {prefix}_mobile.bin.gz + manifest. The output
preserves the file format exactly so the frontend loader needs no
changes other than picking the URL.

Land/ocean separation is preserved: each output cell aggregates only
the input sub-cells of the same surface type, using the high-res
land mask as ground truth. This avoids smearing across coastlines
where land and ocean values have fundamentally different physics
(SST vs land air temperature, transpiration vs nothing, etc).

Usage:
    PYTHONPATH=backend uv run python -m export_frontend_data.downsample_for_mobile \
        --input-dir frontend/public --factor 4
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


# Fields that have meaningful land vs ocean distinction. For these we
# aggregate land sub-cells into land output cells and ocean sub-cells
# into ocean output cells. The output land mask is computed first.
SURFACE_DEPENDENT_FIELDS = {
    "temperature_2m",
    "surface",
    "precipitation",
    "humidity",
    "soil_moisture",
    "vegetation_fraction",
    "snow_temperature",
    "cloud_fraction",
    "cloud_high",
    "cloud_low",
    "cloud_convective",
    "elevation",
}

# Fields that exist everywhere with no land/ocean split (atmospheric).
SURFACE_INDEPENDENT_FIELDS = {
    "wind_u_10m",
    "wind_v_10m",
    "wind_speed_10m",
}

# Land mask fields. These are categorical: each output cell is land
# iff at least half of its input sub-cells are land.
LAND_MASK_FIELDS = {
    "land_mask",
    "land_mask_1deg",
    "land_mask_native",
}


def _read_export(bin_gz_path: Path, manifest_path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a binary export and decode each field into a numpy array."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    with gzip.open(bin_gz_path, "rb") as f:
        raw = f.read()

    arrays: dict[str, np.ndarray] = {}
    for field in manifest["fields"]:
        name = field["name"]
        shape = tuple(field["shape"])
        dtype_str = field["dtype"]
        offset = field["offset"]
        nbytes = field["bytes"]
        buf = raw[offset : offset + nbytes]
        if dtype_str == "float16":
            arr = np.frombuffer(buf, dtype=np.float16).reshape(shape).astype(np.float32)
        elif dtype_str == "float32":
            arr = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        elif dtype_str == "uint8":
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        else:
            raise ValueError(f"Unknown dtype {dtype_str} for field {name}")
        arrays[name] = arr
    return manifest, arrays


def _downsample_land_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a uint8 land mask. Output cell is land iff >= half of sub-cells are land."""
    nlat, nlon = mask.shape
    if nlat % factor != 0 or nlon % factor != 0:
        raise ValueError(
            f"Land mask shape {(nlat, nlon)} not divisible by factor {factor}"
        )
    out_nlat = nlat // factor
    out_nlon = nlon // factor
    blocks = mask.reshape(out_nlat, factor, out_nlon, factor)
    counts = blocks.sum(axis=(1, 3))  # how many land sub-cells per output cell
    threshold = (factor * factor) // 2  # >= half → land
    return (counts >= threshold).astype(np.uint8)


def _downsample_surface_dependent(
    arr: np.ndarray,
    high_mask: np.ndarray,
    low_mask: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Aggregate `arr` into a coarser grid using land/ocean separation.

    For each output cell, take the mean of only those input sub-cells
    whose surface type matches the output cell's surface type. If a
    cell has no matching sub-cells (rare edge case), fall back to the
    unconditional mean of the block.
    """
    if arr.shape[-2:] != high_mask.shape:
        raise ValueError(
            f"Field shape {arr.shape} does not match high-res mask {high_mask.shape}"
        )
    out_nlat, out_nlon = low_mask.shape

    # Reshape arr into blocks of (..., out_nlat, factor, out_nlon, factor)
    blocks_shape = arr.shape[:-2] + (out_nlat, factor, out_nlon, factor)
    blocks = arr.reshape(blocks_shape)

    mask_blocks = high_mask.reshape(out_nlat, factor, out_nlon, factor).astype(bool)
    # Reduce to (out_nlat, out_nlon) blockwise: how many sub-cells of each type
    land_count = mask_blocks.sum(axis=(1, 3))
    ocean_count = (factor * factor) - land_count

    # Sum of values restricted to land / ocean sub-cells
    # blocks shape (..., out_nlat, factor, out_nlon, factor)
    # mask_blocks broadcasts across leading dims
    land_sum = np.where(mask_blocks, blocks, 0.0).sum(axis=(-3, -1))
    ocean_sum = np.where(~mask_blocks, blocks, 0.0).sum(axis=(-3, -1))
    total_sum = blocks.sum(axis=(-3, -1))

    # Means (avoid division by zero)
    land_mean = np.where(land_count > 0, land_sum / np.maximum(land_count, 1), 0.0)
    ocean_mean = np.where(ocean_count > 0, ocean_sum / np.maximum(ocean_count, 1), 0.0)
    fallback_mean = total_sum / (factor * factor)

    # Pick land mean for land output cells, ocean mean for ocean cells
    is_land_low = low_mask.astype(bool)
    out = np.where(is_land_low, land_mean, ocean_mean)

    # Edge case: an output cell labeled "land" but with zero land sub-cells
    # (shouldn't happen given the threshold rule, but be defensive)
    needs_fallback = (
        (is_land_low & (land_count == 0))
        | (~is_land_low & (ocean_count == 0))
    )
    if np.any(needs_fallback):
        out = np.where(needs_fallback, fallback_mean, out)

    return out.astype(arr.dtype)


def _downsample_surface_independent(arr: np.ndarray, factor: int) -> np.ndarray:
    """Plain block-mean downsampling for fields with no surface-type split."""
    nlat, nlon = arr.shape[-2:]
    if nlat % factor != 0 or nlon % factor != 0:
        raise ValueError(
            f"Field shape {arr.shape} not divisible by factor {factor}"
        )
    out_nlat = nlat // factor
    out_nlon = nlon // factor
    blocks_shape = arr.shape[:-2] + (out_nlat, factor, out_nlon, factor)
    return arr.reshape(blocks_shape).mean(axis=(-3, -1)).astype(arr.dtype)


def _encode_field(arr: np.ndarray, dtype_str: str, quant_min: float, quant_max: float) -> bytes:
    """Encode an array back into the binary format used by the export."""
    if dtype_str == "float16":
        return arr.astype(np.float16).tobytes()
    elif dtype_str == "float32":
        return arr.astype(np.float32).tobytes()
    elif dtype_str == "uint8":
        # temperature_2m and snow_temperature are quantized; everything else
        # (land masks) is already uint8 categorical.
        if arr.dtype == np.uint8:
            return arr.tobytes()
        # Re-quantize to uint8 using the manifest's range
        qrange = quant_max - quant_min
        scaled = np.clip((arr - quant_min) * (255.0 / qrange), 0, 255)
        return scaled.astype(np.uint8).tobytes()
    raise ValueError(f"Unknown dtype {dtype_str}")


def _decode_field_for_aggregation(
    arr: np.ndarray, name: str, dtype_str: str, manifest: dict
) -> np.ndarray:
    """Decode a stored field into physical units for aggregation."""
    if dtype_str == "uint8" and name not in LAND_MASK_FIELDS:
        # Quantized field — recover physical value
        quant_min = manifest.get("quantization_min", -60.0)
        quant_max = manifest.get("quantization_max", 60.0)
        if name == "snow_temperature":
            # snow_temperature uses a fixed range in the export
            quant_min, quant_max = -60.0, 60.0
        qrange = quant_max - quant_min
        return arr.astype(np.float32) * (qrange / 255.0) + quant_min
    return arr.astype(np.float32) if dtype_str != "uint8" else arr


def downsample_export(
    input_bin_gz: Path,
    input_manifest: Path,
    output_bin_gz: Path,
    output_manifest: Path,
    factor: int,
) -> None:
    """Read a binary export, downsample it, and write the result.

    Only fields stored at the highest resolution (matching land_mask)
    are downsampled. Lower-resolution fields are passed through
    unchanged — they're already small enough.
    """
    print(f"Reading {input_bin_gz.name}")
    manifest, arrays = _read_export(input_bin_gz, input_manifest)

    if "land_mask" not in arrays:
        raise ValueError(
            "Input export must contain a 'land_mask' field for surface-aware downsampling"
        )
    high_mask = arrays["land_mask"]
    high_shape = high_mask.shape
    print(f"  high-res land mask shape: {high_shape}")

    low_mask = _downsample_land_mask(high_mask, factor)
    print(f"  low-res land mask shape:  {low_mask.shape}")

    out_arrays: dict[str, np.ndarray] = {"land_mask": low_mask}

    for field in manifest["fields"]:
        name = field["name"]
        if name == "land_mask":
            continue
        dtype_str = field["dtype"]
        arr = arrays[name]

        # Only downsample fields whose spatial dims match the high-res grid.
        # Everything else is already small enough and passes through.
        if arr.shape[-2:] != high_shape:
            out_arrays[name] = arr
            print(f"  {name}: {arr.shape} (passthrough)")
            continue

        try:
            if name in LAND_MASK_FIELDS:
                out = _downsample_land_mask(arr, factor)
            elif name in SURFACE_DEPENDENT_FIELDS:
                physical = _decode_field_for_aggregation(arr, name, dtype_str, manifest)
                out = _downsample_surface_dependent(physical, high_mask, low_mask, factor)
            elif name in SURFACE_INDEPENDENT_FIELDS:
                physical = _decode_field_for_aggregation(arr, name, dtype_str, manifest)
                out = _downsample_surface_independent(physical, factor)
            else:
                print(f"  WARNING: unknown high-res field {name}, passing through")
                out_arrays[name] = arr
                continue
        except ValueError as e:
            print(f"  skipping {name}: {e}")
            continue

        out_arrays[name] = out
        print(f"  {name}: {arr.shape} -> {out.shape}")

    # Re-emit binary
    new_manifest: dict = {
        "fields": [],
        "quantization_min": manifest.get("quantization_min"),
        "quantization_max": manifest.get("quantization_max"),
    }
    blobs: list[bytes] = []
    offset = 0

    quant_min = manifest.get("quantization_min", -60.0)
    quant_max = manifest.get("quantization_max", 60.0)

    # Preserve the original field order
    for field in manifest["fields"]:
        name = field["name"]
        if name not in out_arrays:
            continue
        arr = out_arrays[name]
        dtype_str = field["dtype"]
        raw = _encode_field(arr, dtype_str, quant_min, quant_max)
        new_manifest["fields"].append(
            {
                "name": name,
                "shape": list(arr.shape),
                "dtype": dtype_str,
                "offset": offset,
                "bytes": len(raw),
            }
        )
        blobs.append(raw)
        offset += len(raw)

    raw_data = b"".join(blobs)
    raw_size = len(raw_data) / (1024 * 1024)
    with gzip.open(output_bin_gz, "wb", compresslevel=9) as f:
        f.write(raw_data)
    with open(output_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)

    gz_size = output_bin_gz.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_bin_gz.name} ({gz_size:.2f} MB, {raw_size:.1f} MB uncompressed)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downsample binary climate data exports for mobile delivery"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="frontend/public",
        help="Directory containing the source .bin.gz files (default: frontend/public)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input-dir)",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=4,
        help="Downsample factor along each axis (default: 4 → 720x1440 becomes 180x360)",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default="main,stage1,stage2,stage3,stage4",
        help="Comma-separated file prefixes to process",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_mobile",
        help="Suffix to append to output file names (default: _mobile)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]
    for prefix in prefixes:
        bin_path = input_dir / f"{prefix}.bin.gz"
        manifest_path = input_dir / f"{prefix}.manifest.json"
        if not bin_path.exists() or not manifest_path.exists():
            print(f"Skipping {prefix}: not found in {input_dir}")
            continue
        out_bin = output_dir / f"{prefix}{args.suffix}.bin.gz"
        out_manifest = output_dir / f"{prefix}{args.suffix}.manifest.json"
        downsample_export(bin_path, manifest_path, out_bin, out_manifest, args.factor)

    print("\nDownsampling complete.")


if __name__ == "__main__":
    main()
