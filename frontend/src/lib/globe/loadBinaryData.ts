/**
 * Fetch and decode binary climate data from main.manifest.json + main.bin.
 */

interface ManifestField {
  name: string;
  shape: number[];
  dtype: 'float16' | 'uint8' | 'float32';
  offset: number;
  bytes: number;
}

interface Manifest {
  fields: ManifestField[];
}

export interface FieldData {
  data: Float32Array | Uint8Array;
  shape: number[];
}

export interface ClimateLayerData {
  /** Interpolated 0.25deg temperature [12, 720, 1440] */
  temperature_2m: FieldData;
  /** Native 5deg surface temp [12, 36, 72] */
  surface: FieldData;
  /** 0.25deg land mask [720, 1440] — matches temperature_2m resolution */
  land_mask: FieldData;
  /** Native 5deg land mask [36, 72] — for type-aware interpolation */
  land_mask_native?: FieldData;
  /** Native 5deg vegetation fraction [12, 36, 72] */
  vegetation_fraction?: FieldData;
  /** Native 5deg soil moisture fraction [12, 36, 72] */
  soil_moisture?: FieldData;
  /** Native 5deg wind fields [12, 36, 72] */
  wind_u_10m?: FieldData;
  wind_v_10m?: FieldData;
  wind_speed_10m?: FieldData;
  /** Native 5deg cloud fraction [12, 36, 72] */
  cloud_fraction?: FieldData;
  /** Native 5deg high cloud fraction [12, 36, 72] */
  cloud_high?: FieldData;
  /** Native 5deg low (stratiform+marine) cloud fraction [12, 36, 72] */
  cloud_low?: FieldData;
  /** Native 5deg convective cloud fraction [12, 36, 72] */
  cloud_convective?: FieldData;
  /** 0.25deg elevation in meters [720, 1440] */
  elevation?: FieldData;
  /** 0.25deg snow temperature uint8-quantized [12, 720, 1440] */
  snow_temperature?: FieldData;
}

/** Decode a Float16 buffer into Float32Array. */
function decodeFloat16(buffer: ArrayBuffer, byteOffset: number, count: number): Float32Array {
  const u16 = new Uint16Array(buffer, byteOffset, count);
  const f32 = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const h = u16[i];
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;

    let value: number;
    if (exponent === 0) {
      // Subnormal or zero
      value = (mantissa / 1024) * Math.pow(2, -14);
    } else if (exponent === 31) {
      // Inf or NaN
      value = mantissa === 0 ? Infinity : NaN;
    } else {
      // Normal
      value = (1 + mantissa / 1024) * Math.pow(2, exponent - 15);
    }

    f32[i] = sign ? -value : value;
  }

  return f32;
}

function decodeField(buffer: ArrayBuffer, field: ManifestField): Float32Array | Uint8Array {
  if (field.dtype === 'uint8') {
    return new Uint8Array(buffer, field.offset, field.bytes);
  }
  if (field.dtype === 'float16') {
    const count = field.bytes / 2;
    return decodeFloat16(buffer, field.offset, count);
  }
  // float32
  return new Float32Array(buffer, field.offset, field.bytes / 4);
}

export async function loadBinaryData(basePath: string = ''): Promise<ClimateLayerData> {
  const [manifestRes, binRes] = await Promise.all([
    fetch(`${basePath}/main.manifest.json`),
    fetch(`${basePath}/main.bin`),
  ]);

  if (!manifestRes.ok) {
    throw new Error(`Failed to load manifest: ${manifestRes.status}`);
  }
  if (!binRes.ok) {
    throw new Error(`Failed to load binary data: ${binRes.status}`);
  }

  const manifest: Manifest = await manifestRes.json();
  const buffer = await binRes.arrayBuffer();

  const result: Record<string, FieldData> = {};

  for (const field of manifest.fields) {
    result[field.name] = {
      data: decodeField(buffer, field),
      shape: field.shape,
    };
  }

  return result as unknown as ClimateLayerData;
}

/**
 * Read a single value from a field at [month, lat, lon] indices.
 * Uses flat indexing: index = month * (nlat * nlon) + lat * nlon + lon
 */
export function sampleField3D(field: FieldData, month: number, lat: number, lon: number): number {
  const [, nlat, nlon] = field.shape;
  return (field.data as Float32Array)[month * nlat * nlon + lat * nlon + lon];
}

/**
 * Read from a 2D field [lat, lon].
 */
export function sampleField2D(field: FieldData, lat: number, lon: number): number {
  const [, nlon] = field.shape;
  return field.data[lat * nlon + lon];
}

/**
 * Convert temperature_2m FieldData to the nested number[][][] format
 * expected by the existing Globe temperature renderer.
 */
export function fieldToNestedArray(field: FieldData): number[][][] {
  const [nmonths, nlat, nlon] = field.shape;
  const data = field.data as Float32Array;
  const result: number[][][] = [];

  for (let m = 0; m < nmonths; m++) {
    const month: number[][] = [];
    for (let i = 0; i < nlat; i++) {
      const row: number[] = [];
      const base = m * nlat * nlon + i * nlon;
      for (let j = 0; j < nlon; j++) {
        row.push(data[base + j]);
      }
      month.push(row);
    }
    result.push(month);
  }

  return result;
}
