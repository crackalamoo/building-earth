/**
 * Fetch and decode binary climate data from main.manifest.json + main.bin.
 */

/**
 * Base URL for binary climate data files. In production, these live on a
 * Cloudflare R2 bucket; in dev, they're served from frontend/public via Vite.
 * Set VITE_DATA_BASE in .env to point to the R2 public URL.
 */
const DATA_BASE: string = (import.meta.env.VITE_DATA_BASE || '').replace(/\/$/, '');

/**
 * Whether this device should load the lower-resolution mobile binaries.
 * The mobile files have the same field structure but the high-res grid
 * (720x1440) is downsampled to 180x360, dropping mesh and GPU memory by ~16x.
 */
const IS_MOBILE: boolean =
  typeof navigator !== 'undefined' &&
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

/** Pick the file basename (with _mobile suffix if appropriate). */
function variantName(base: string): string {
  return IS_MOBILE ? `${base}_mobile` : base;
}

/**
 * Fetch a gzip-compressed .bin.gz file and return the decompressed ArrayBuffer.
 * Some servers (Vite dev) transparently decompress via Content-Encoding: gzip,
 * while others (Cloudflare R2) serve raw gzip bytes. We detect which case
 * by checking for the gzip magic number (0x1f 0x8b) in the first two bytes.
 */
async function fetchBinary(path: string): Promise<ArrayBuffer> {
  const url = `${DATA_BASE}/${path}.gz`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const header = new Uint8Array(buf, 0, 2);
  if (header[0] === 0x1f && header[1] === 0x8b) {
    // Still gzip-compressed — decompress client-side
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    writer.write(new Uint8Array(buf));
    writer.close();
    return new Response(ds.readable).arrayBuffer();
  }
  // Already decompressed by the browser
  return buf;
}

interface ManifestField {
  name: string;
  shape: number[];
  dtype: 'float16' | 'uint8' | 'float32';
  offset: number;
  bytes: number;
}

interface Manifest {
  fields: ManifestField[];
  quantization_min?: number;
  quantization_max?: number;
}

export interface FieldData {
  data: Float32Array | Uint8Array;
  shape: number[];
}

export interface ClimateLayerData {
  /** Interpolated 0.25deg temperature [12, 720, 1440] */
  temperature_2m: FieldData;
  /** Native 5deg surface temp [12, 36, 72] */
  surface?: FieldData;
  /** 0.25deg land mask [720, 1440] — matches temperature_2m resolution */
  land_mask: FieldData;
  /** Native 5deg land mask [36, 72] — for type-aware interpolation */
  land_mask_native?: FieldData;
  /** 1deg land mask [180, 360] — matches soil_moisture/precipitation resolution */
  land_mask_1deg?: FieldData;
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
  /** 1deg precipitation in kg/m²/s [12, 180, 360] */
  precipitation?: FieldData;
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

export async function loadBinaryData(): Promise<ClimateLayerData> {
  const name = variantName('main');
  const [manifestRes, buffer] = await Promise.all([
    fetch(`${DATA_BASE}/${name}.manifest.json`),
    fetchBinary(`${name}.bin`),
  ]);

  if (!manifestRes.ok) {
    throw new Error(`Failed to load manifest: ${manifestRes.status}`);
  }

  const manifest: Manifest = await manifestRes.json();

  const result: Record<string, FieldData> = {};

  for (const field of manifest.fields) {
    let data = decodeField(buffer, field);
    // Decode uint8-quantized temperature_2m back to °C using manifest quantization range
    if (field.name === 'temperature_2m' && field.dtype === 'uint8') {
      const u8 = data as Uint8Array;
      const f32 = new Float32Array(u8.length);
      const qMin = manifest.quantization_min ?? -60;
      const qMax = manifest.quantization_max ?? 60;
      const qRange = qMax - qMin;
      for (let i = 0; i < u8.length; i++) f32[i] = u8[i] * (qRange / 255) + qMin;
      data = f32;
    }
    result[field.name] = { data, shape: field.shape };
  }

  return result as unknown as ClimateLayerData;
}

/**
 * Fetch the standalone 1° land mask (180×360 Uint8Array) for the primordial globe.
 */
export async function loadLandMask1deg(): Promise<{ data: Uint8Array; nlat: number; nlon: number }> {
  const url = `${DATA_BASE}/landmask1deg.bin`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load land mask: ${res.status}`);
  const buf = await res.arrayBuffer();
  const data = new Uint8Array(buf);
  // 1° grid: 180 lat × 360 lon
  return { data, nlat: 180, nlon: 360 };
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
 * Find the nearest land cell to a given lat/lon using the high-res land mask.
 * Spirals outward up to maxRadius cells. Returns snapped lat/lon or the
 * original coordinates if no land is found within range.
 */
export function snapToLand(
  landMask: FieldData,
  lat: number,
  lon: number,
  maxRadius = 40,
): { lat: number; lon: number } {
  const [nlat, nlon] = landMask.shape;
  const data = landMask.data as Uint8Array;
  const resolution = 180 / nlat;

  // Convert lat/lon to grid indices (grid starts at south pole)
  const latIdx = Math.round((lat + 90 - resolution / 2) / resolution);
  const lonIdx = Math.round(((lon + 180) % 360) / resolution);
  const clampLat = (i: number) => Math.max(0, Math.min(nlat - 1, i));
  const wrapLon = (j: number) => ((j % nlon) + nlon) % nlon;

  // Check original cell first
  if (data[clampLat(latIdx) * nlon + wrapLon(lonIdx)] > 0) {
    return { lat, lon };
  }

  // Spiral outward
  for (let r = 1; r <= maxRadius; r++) {
    let bestDist = Infinity;
    let bestLat = lat;
    let bestLon = lon;
    for (let di = -r; di <= r; di++) {
      for (let dj = -r; dj <= r; dj++) {
        if (Math.abs(di) !== r && Math.abs(dj) !== r) continue; // only ring
        const li = clampLat(latIdx + di);
        const lj = wrapLon(lonIdx + dj);
        if (data[li * nlon + lj] > 0) {
          const dist = di * di + dj * dj;
          if (dist < bestDist) {
            bestDist = dist;
            bestLat = li * resolution - 90 + resolution / 2;
            bestLon = lj * resolution - 180 + resolution / 2;
            if (bestLon > 180) bestLon -= 360;
          }
        }
      }
    }
    if (bestDist < Infinity) {
      return { lat: bestLat, lon: bestLon };
    }
  }

  // No land found within range — use original
  return { lat, lon };
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

/**
 * Cache for preloaded + decoded stage data.
 * preloadStageFile fetches AND decodes in background, so loadStageData
 * can return instantly when the user clicks advance.
 */
interface PreloadedStage {
  layerData: ClimateLayerData;
  temperatureData: number[][][];
}
const decodedCache = new Map<number, PreloadedStage>();
const preloadPromises = new Map<number, Promise<void>>();

/**
 * Fetch, decode, and cache a stage's data in the background.
 * Call this after loading stage N to pre-decode stage N+1.
 */
export function preloadStageFile(stage: number): Promise<void> {
  if (decodedCache.has(stage)) return Promise.resolve();
  if (preloadPromises.has(stage)) return preloadPromises.get(stage)!;

  const promise = (async () => {
    const prefix = variantName(stage === 5 ? 'main' : `stage${stage}`);
    try {
      const [manifestRes, buffer] = await Promise.all([
        fetch(`${DATA_BASE}/${prefix}.manifest.json`),
        fetchBinary(`${prefix}.bin`),
      ]);
      if (!manifestRes.ok) return;

      const manifestData: Manifest = await manifestRes.json();

      // Decode in worker
      const decoded = await new Promise<ClimateLayerData>((resolve, reject) => {
        const worker = new Worker(new URL('./decodeWorker.ts', import.meta.url), { type: 'module' });
        worker.onmessage = (e: MessageEvent) => {
          worker.terminate();
          resolve(e.data.fields as unknown as ClimateLayerData);
        };
        worker.onerror = (err) => {
          worker.terminate();
          reject(new Error(`Decode worker error: ${err.message}`));
        };
        worker.postMessage({ buffer, manifest: manifestData }, [buffer]);
      });

      // Build nested temperature array (runs in worker thread is done, this is fast)
      const t2m = decoded.temperature_2m;
      const temperatureData = t2m ? fieldToNestedArray(t2m) : [];

      decodedCache.set(stage, { layerData: decoded, temperatureData });
    } catch {
      // Preload failure is non-fatal
    } finally {
      preloadPromises.delete(stage);
    }
  })();

  preloadPromises.set(stage, promise);
  return promise;
}

/**
 * Load and decode a specific stage's data.
 * Returns instantly if preloaded, otherwise fetches + decodes.
 */
export async function loadStageData(
  stage: number,
  onLayerData: (ld: ClimateLayerData) => void,
  onTemperatureData: (td: number[][][]) => void,
  onError: (err: Error) => void,
): Promise<void> {
  // Wait for any in-flight preload to finish
  if (preloadPromises.has(stage)) {
    await preloadPromises.get(stage);
  }

  // Use decoded cache if available
  if (decodedCache.has(stage)) {
    const cached = decodedCache.get(stage)!;
    decodedCache.delete(stage);
    onLayerData(cached.layerData);
    onTemperatureData(cached.temperatureData);
    return;
  }

  // Fallback: fetch + decode now (shouldn't normally happen)
  try {
    await preloadStageFile(stage);
    if (decodedCache.has(stage)) {
      const cached = decodedCache.get(stage)!;
      decodedCache.delete(stage);
      onLayerData(cached.layerData);
      onTemperatureData(cached.temperatureData);
      return;
    }
    throw new Error('Failed to load stage data');
  } catch (e) {
    onError(e instanceof Error ? e : new Error(String(e)));
  }
}
