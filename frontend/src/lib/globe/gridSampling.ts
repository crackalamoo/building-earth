/**
 * Grid sampling utilities for interpolating low-resolution climate fields
 * onto high-resolution meshes.
 *
 * - Type-aware bilinear interpolation (land/ocean masking)
 * - Nearest same-type fallback for coastal cells
 * - Polar temperature smoothing
 */

/**
 * Manual overrides for regions where type-aware interpolation produces
 * geographic teleconnections (e.g., Sicily matching Tunisia).
 *
 * Each entry: fine cell lat/lon box + surface type → coarse source cell(s).
 * Coordinates in degrees: lat [-90,90], lon [-180,180).
 * Sources are (lat, lon) of coarse cell centers.
 */
interface InterpolationOverride {
  latMin: number; latMax: number;
  lonMin: number; lonMax: number;
  isLand: boolean;
  sources: [number, number][];  // (lat, lon) of coarse cells
}

const INTERPOLATION_OVERRIDES: InterpolationOverride[] = [
  { latMin: 36, latMax: 39, lonMin: 12, lonMax: 16, isLand: true,
    sources: [[42.5, 12.5]] },  // Sicily → Italy
  { latMin: 30, latMax: 35, lonMin: 15, lonMax: 25, isLand: true,
    sources: [[27.5, 17.5], [27.5, 22.5]] },  // Libya coast
  { latMin: 12, latMax: 30, lonMin: 32, lonMax: 44, isLand: false,
    sources: [[32.5, 27.5], [7.5, 52.5]] },  // Red Sea
  { latMin: 35, latMax: 41, lonMin: 23, lonMax: 28, isLand: false,
    sources: [[32.5, 22.5], [32.5, 27.5]] },  // Aegean Sea
];

/**
 * Check if a geographic point matches an override entry and return
 * the weighted value from override sources, or null if no override applies.
 */
function applyOverride(
  data: Float32Array, nlat: number, nlon: number, monthOffset: number,
  latDeg: number, lonDeg: number, isLand: boolean,
): number | null {
  for (const ov of INTERPOLATION_OVERRIDES) {
    if (isLand !== ov.isLand) continue;
    if (latDeg < ov.latMin || latDeg > ov.latMax) continue;
    if (lonDeg < ov.lonMin || lonDeg > ov.lonMax) continue;

    // Compute inverse-distance weights from fine point to each source
    let totalW = 0;
    let result = 0;
    const dLon = 360 / nlon;  // grid spacing in degrees
    const dLat = 180 / nlat;
    const lonOrigin = dLon / 2;  // first grid center (2.5° for 5° grid)
    const latOrigin = -90 + dLat / 2;  // first grid center (-87.5° for 5° grid)
    for (const [sLat, sLon] of ov.sources) {
      // Convert source (lat,lon) to grid index
      const si = Math.round((sLat - latOrigin) / dLat);
      const sj = Math.round((((sLon - lonOrigin) % 360 + 360) % 360) / dLon);
      const clampI = Math.max(0, Math.min(nlat - 1, si));
      const clampJ = ((sj % nlon) + nlon) % nlon;
      const val = data[monthOffset + clampI * nlon + clampJ];

      const dlat2 = latDeg - sLat;
      const dlon2 = lonDeg - sLon;
      const dist = Math.max(0.01, Math.sqrt(dlat2 * dlat2 + dlon2 * dlon2));
      const w = 1 / dist;
      totalW += w;
      result += w * val;
    }
    return result / totalW;
  }
  return null;
}

/**
 * Bilinear sample a low-res monthly field at a high-res (lat, lon) index.
 * When coarseLandMask is provided, only blends from coarse cells matching
 * the fine cell's surface type (land/ocean). Falls back to nearest same-type
 * neighbor if no bilinear neighbors match.
 */
export function sampleBilinear(
  data: Float32Array, lowNlat: number, lowNlon: number, monthOffset: number,
  hiLatIdx: number, hiLonIdx: number, hiNlat: number, hiNlon: number,
  isLand?: boolean, coarseLandMask?: Uint8Array,
): number {
  // Map high-res cell center to continuous low-res grid coordinates
  const latStep = hiNlat / lowNlat;
  const lonStep = hiNlon / lowNlon;
  const latF = (hiLatIdx + 0.5) / latStep - 0.5;
  const lonF = (hiLonIdx + 0.5) / lonStep - 0.5;

  const lat0 = Math.max(0, Math.min(lowNlat - 1, Math.floor(latF)));
  const lat1 = Math.min(lowNlat - 1, lat0 + 1);
  const lon0 = ((Math.floor(lonF) % lowNlon) + lowNlon) % lowNlon;
  const lon1 = (lon0 + 1) % lowNlon;

  const tLat = Math.max(0, Math.min(1, latF - Math.floor(latF)));
  const tLon = Math.max(0, Math.min(1, lonF - Math.floor(lonF)));

  const idx00 = lat0 * lowNlon + lon0;
  const idx10 = lat1 * lowNlon + lon0;
  const idx01 = lat0 * lowNlon + lon1;
  const idx11 = lat1 * lowNlon + lon1;

  // Check manual overrides for known problem regions
  if (isLand !== undefined && coarseLandMask !== undefined) {
    const latDeg = -90 + (hiLatIdx + 0.5) * (180 / hiNlat);
    let lonDeg = (hiLonIdx + 0.5) * (360 / hiNlon);
    if (lonDeg > 180) lonDeg -= 360;
    const ov = applyOverride(data, lowNlat, lowNlon, monthOffset, latDeg, lonDeg, isLand);
    if (ov !== null) return ov;
  }

  const v00 = data[monthOffset + idx00];
  const v10 = data[monthOffset + idx10];
  const v01 = data[monthOffset + idx01];
  const v11 = data[monthOffset + idx11];

  // Standard bilinear weights
  let w00 = (1 - tLat) * (1 - tLon);
  let w10 = tLat * (1 - tLon);
  let w01 = (1 - tLat) * tLon;
  let w11 = tLat * tLon;

  // Type-aware masking: zero out weights for coarse cells of wrong type
  if (coarseLandMask !== undefined && isLand !== undefined) {
    const target = isLand ? 1 : 0;
    if (coarseLandMask[idx00] !== target) w00 = 0;
    if (coarseLandMask[idx10] !== target) w10 = 0;
    if (coarseLandMask[idx01] !== target) w01 = 0;
    if (coarseLandMask[idx11] !== target) w11 = 0;

    const wSum = w00 + w10 + w01 + w11;
    if (wSum > 1e-10) {
      w00 /= wSum; w10 /= wSum; w01 /= wSum; w11 /= wSum;
    } else {
      return nearestSameType(
        data, lowNlat, lowNlon, monthOffset, coarseLandMask, target,
        latF, lonF,
      );
    }
  }

  return v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11;
}

/** Find nearest coarse cell of the given surface type and return its value. */
function nearestSameType(
  data: Float32Array, nlat: number, nlon: number, monthOffset: number,
  landMask: Uint8Array, target: number,
  latF: number, lonF: number,
): number {
  const centerLat = Math.round(Math.max(0, Math.min(nlat - 1, latF)));
  const centerLon = Math.round(lonF) % nlon;
  let bestDist = Infinity;
  let bestVal = data[monthOffset + centerLat * nlon + ((centerLon % nlon) + nlon) % nlon];

  const maxR = 5; // search up to 5 cells away (~25° at 5° resolution)
  for (let r = 0; r <= maxR; r++) {
    for (let di = -r; di <= r; di++) {
      for (let dj = -r; dj <= r; dj++) {
        if (Math.abs(di) !== r && Math.abs(dj) !== r) continue; // ring perimeter only
        const li = centerLat + di;
        if (li < 0 || li >= nlat) continue;
        const lj = ((centerLon + dj) % nlon + nlon) % nlon;
        if (landMask[li * nlon + lj] !== target) continue;
        const dist = (latF - li) * (latF - li) + (lonF - lj) * (lonF - lj);
        if (dist < bestDist) {
          bestDist = dist;
          bestVal = data[monthOffset + li * nlon + lj];
        }
      }
    }
    if (bestDist < Infinity) return bestVal;
  }
  return bestVal;
}

/**
 * Smooth temperature data near poles to reduce longitudinal artifacts.
 * Applies a moving-average window that grows above 75° latitude.
 */
export function averagePolarTemps(
  monthData: number[][], latCount: number, lonCount: number,
): number[][] {
  const latStep = 180 / latCount;
  const result = monthData.map(row => [...row]);

  for (let i = 0; i < latCount; i++) {
    const lat = -90 + (i + 0.5) * latStep;
    const absLat = Math.abs(lat);

    if (absLat > 75) {
      const t = (absLat - 75) / 15;
      const maxWindow = lonCount / 24;
      const windowSize = 1 + t * t * maxWindow;
      result[i] = smoothRow(result[i], windowSize);
    }
  }

  return result;
}

function smoothRow(row: number[], windowSize: number): number[] {
  if (windowSize <= 1) return row;
  windowSize = Math.floor(windowSize);
  if (windowSize % 2 === 0) windowSize++;
  const halfWindow = Math.floor(windowSize / 2);
  const newRow = new Array(row.length);

  for (let j = 0; j < row.length; j++) {
    let sum = 0;
    for (let k = -halfWindow; k <= halfWindow; k++) {
      const idx = (j + k + row.length) % row.length;
      sum += row[idx];
    }
    newRow[j] = sum / windowSize;
  }
  return newRow;
}