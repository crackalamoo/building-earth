/**
 * Elevation displacement and hillshade for the globe.
 *
 * Vertex positions are displaced radially by elevation * ELEVATION_SCALE.
 * Normals are blended between sphere and displaced-surface normals.
 * Hillshade is baked into vertex colors for visibility on the night side.
 */

export const ELEVATION_SCALE = 6e-6;

/** Blend factor between sphere normal (0) and displaced normal (1). */
export const NORMAL_BLEND = 0.50;

/** Look up elevation (meters) for a lat/lon from the elevation grid. */
export function sampleElevation(
  data: Float32Array, nlat: number, nlon: number,
  lat: number, lon: number,
): number {
  const renderI = Math.floor(((90 - lat) / 180) * nlat);
  const dataI = nlat - 1 - Math.min(renderI, nlat - 1);
  const j = Math.floor((lon / 360) * nlon) % nlon;
  return data[dataI * nlon + j];
}

/**
 * Compute hillshade factor for a point on the elevation grid.
 * Uses a fixed NW illumination (azimuth 315°, altitude 45°) baked into vertex colors
 * so terrain is visible on both day and night sides of the globe.
 * Returns a color multiplier: ~0.55 (deep shadow) to ~1.35 (bright face).
 */
export function hillshade(
  data: Float32Array, nlat: number, nlon: number,
  lat: number, lon: number,
): number {
  const dLat = 180 / nlat;
  const dLon = 360 / nlon;
  const degToRad = Math.PI / 180;

  const eN = sampleElevation(data, nlat, nlon, Math.min(lat + dLat, 89.9), lon);
  const eS = sampleElevation(data, nlat, nlon, Math.max(lat - dLat, -89.9), lon);
  const eE = sampleElevation(data, nlat, nlon, lat, (lon + dLon) % 360);
  const eW = sampleElevation(data, nlat, nlon, lat, (lon - dLon + 360) % 360);

  // Cell size in meters
  const cellLatM = dLat * 111320;
  const cosLat = Math.cos(lat * degToRad);
  const cellLonM = dLon * 111320 * (cosLat > 0.01 ? cosLat : 0.01);

  // Gradient: rise/run
  const dzdx = (eE - eW) / (2 * cellLonM);
  const dzdy = (eN - eS) / (2 * cellLatM);

  // Light from NW (azimuth 315°), altitude 45°
  const lx = -0.7071;
  const ly = 0.7071;

  // Heavy exaggeration needed at 0.25° resolution (real slopes max ~0.1)
  const zFactor = 40.0;
  const gx = dzdx * zFactor;
  const gy = dzdy * zFactor;

  const shade = (1 + lx * gx + ly * gy) / Math.sqrt(1 + gx * gx + gy * gy);

  return 0.55 + 0.8 * Math.max(0, Math.min(1, (shade + 1) * 0.5));
}

/**
 * Compute displaced-surface normal via finite differences on actual 3D positions.
 */
export function displacedNormal(
  data: Float32Array, nlat: number, nlon: number,
  lat: number, lon: number, radius: number,
): [number, number, number] {
  const dLat = 180 / nlat;
  const dLon = 360 / nlon;
  const degToRad = Math.PI / 180;

  function pos(la: number, lo: number): [number, number, number] {
    const elev = sampleElevation(data, nlat, nlon, la, lo);
    const r = radius + elev * ELEVATION_SCALE;
    const phi = (90 - la) * degToRad;
    const theta = lo * degToRad;
    return [
      -r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta),
    ];
  }

  const pN = pos(Math.min(lat + dLat, 89.9), lon);
  const pS = pos(Math.max(lat - dLat, -89.9), lon);
  const pE = pos(lat, (lon + dLon) % 360);
  const pW = pos(lat, (lon - dLon + 360) % 360);

  const tLatX = pN[0] - pS[0], tLatY = pN[1] - pS[1], tLatZ = pN[2] - pS[2];
  const tLonX = pE[0] - pW[0], tLonY = pE[1] - pW[1], tLonZ = pE[2] - pW[2];

  // Cross product: tLon × tLat
  let nx = tLonY * tLatZ - tLonZ * tLatY;
  let ny = tLonZ * tLatX - tLonX * tLatZ;
  let nz = tLonX * tLatY - tLonY * tLatX;

  // Ensure outward-pointing
  const phi = (90 - lat) * degToRad;
  const theta = lon * degToRad;
  const rx = -Math.sin(phi) * Math.cos(theta);
  const ry = Math.cos(phi);
  const rz = Math.sin(phi) * Math.sin(theta);
  if (nx * rx + ny * ry + nz * rz < 0) { nx = -nx; ny = -ny; nz = -nz; }

  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  return [nx / len, ny / len, nz / len];
}

/**
 * Pre-compute a per-cell hillshade grid for a lat/lon grid.
 * Returns Float32Array of size latCount * lonCount.
 */
export function computeHillshadeGrid(
  data: Float32Array, nlat: number, nlon: number,
  latCount: number, lonCount: number,
): Float32Array {
  const grid = new Float32Array(latCount * lonCount);
  const latStep = 180 / latCount;
  const lonStep = 360 / lonCount;
  for (let i = 0; i < latCount; i++) {
    const lat = 90 - (i + 0.5) * latStep;
    for (let j = 0; j < lonCount; j++) {
      const lon = (j + 0.5) * lonStep;
      grid[i * lonCount + j] = hillshade(data, nlat, nlon, lat, lon);
    }
  }
  return grid;
}
