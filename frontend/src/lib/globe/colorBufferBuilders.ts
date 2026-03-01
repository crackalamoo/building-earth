import { temperatureToColorNormalized } from './colormap';
import { precipitationToColorNormalized } from './precipitationColormap';
import { blueMarbleColor } from './blueMarbleColormap';
import { sampleBilinear, averagePolarTemps } from './gridSampling';
import { computeHillshadeGrid } from './elevation';
import type { ClimateLayerData } from './loadBinaryData';

// ---------------------------------------------------------------------------
// Lazy caches (module-level, shared across calls)
// ---------------------------------------------------------------------------

let hillshadeGridCache: Float32Array | null = null;
let annualMeanTempCache: Float32Array | null = null;
let annualMeanSoilCache: Float32Array | null = null;
let soilLandMaskCache: Uint8Array | null = null;

/** Invalidate all cached computations. Call when layerData changes. */
export function invalidateColorBuilderCaches(): void {
  hillshadeGridCache = null;
  annualMeanTempCache = null;
  annualMeanSoilCache = null;
  soilLandMaskCache = null;
}

// ---------------------------------------------------------------------------
// Hillshade grid (lazy, computed from elevation)
// ---------------------------------------------------------------------------

export function getHillshadeGrid(ld: ClimateLayerData): Float32Array | null {
  if (hillshadeGridCache) return hillshadeGridCache;
  const elevData = ld.elevation?.data as Float32Array | undefined;
  const elevNlat = ld.elevation?.shape[0] ?? 0;
  const elevNlon = ld.elevation?.shape[1] ?? 0;
  const hiNlat = ld.land_mask.shape[0];
  const hiNlon = ld.land_mask.shape[1];
  if (elevData && elevNlat && elevNlon) {
    hillshadeGridCache = computeHillshadeGrid(elevData, elevNlat, elevNlon, hiNlat, hiNlon);
  }
  return hillshadeGridCache;
}

// ---------------------------------------------------------------------------
// Annual means (lazy, computed from full 12-month data)
// ---------------------------------------------------------------------------

export function getAnnualMeans(ld: ClimateLayerData): { temp: Float32Array; soil: Float32Array; soilLandMask: Uint8Array } {
  if (annualMeanTempCache) return { temp: annualMeanTempCache, soil: annualMeanSoilCache!, soilLandMask: soilLandMaskCache! };

  const surfaceData = ld.surface.data as Float32Array;
  const soilData = ld.soil_moisture?.data as Float32Array | undefined;
  const lowNlat = ld.surface.shape[1];
  const lowNlon = ld.surface.shape[2];
  const sNlat = ld.soil_moisture?.shape[1] ?? lowNlat;
  const sNlon = ld.soil_moisture?.shape[2] ?? lowNlon;

  const annualMeanTemp = new Float32Array(lowNlat * lowNlon);
  for (let ci = 0; ci < lowNlat; ci++) {
    for (let cj = 0; cj < lowNlon; cj++) {
      let tsum = 0;
      for (let m = 0; m < 12; m++) {
        tsum += surfaceData[m * lowNlat * lowNlon + ci * lowNlon + cj];
      }
      annualMeanTemp[ci * lowNlon + cj] = tsum / 12;
    }
  }

  const annualMeanSoil = new Float32Array(sNlat * sNlon);
  if (soilData) {
    for (let ci = 0; ci < sNlat; ci++) {
      for (let cj = 0; cj < sNlon; cj++) {
        let ssum = 0;
        for (let m = 0; m < 12; m++) {
          ssum += soilData[m * sNlat * sNlon + ci * sNlon + cj];
        }
        annualMeanSoil[ci * sNlon + cj] = ssum / 12;
      }
    }
  } else {
    annualMeanSoil.fill(0.5);
  }

  // Use exported 1deg land mask if available, otherwise downsample from 0.25deg
  let soilLandMask: Uint8Array;
  if (ld.land_mask_1deg) {
    soilLandMask = ld.land_mask_1deg.data as Uint8Array;
  } else {
    const hiLandMask = ld.land_mask.data as Uint8Array;
    const hiNlatM = ld.land_mask.shape[0];
    const hiNlonM = ld.land_mask.shape[1];
    soilLandMask = new Uint8Array(sNlat * sNlon);
    const latRatio = hiNlatM / sNlat;
    const lonRatio = hiNlonM / sNlon;
    for (let ci = 0; ci < sNlat; ci++) {
      const hi0 = Math.floor(ci * latRatio);
      const hi1 = Math.min(Math.floor((ci + 1) * latRatio), hiNlatM);
      for (let cj = 0; cj < sNlon; cj++) {
        const hj0 = Math.floor(cj * lonRatio);
        const hj1 = Math.min(Math.floor((cj + 1) * lonRatio), hiNlonM);
        let landCount = 0;
        for (let ii = hi0; ii < hi1; ii++) {
          for (let jj = hj0; jj < hj1; jj++) {
            if (hiLandMask[ii * hiNlonM + jj] === 1) landCount++;
          }
        }
        soilLandMask[ci * sNlon + cj] = landCount > 0 ? 1 : 0;
      }
    }
  }

  annualMeanTempCache = annualMeanTemp;
  annualMeanSoilCache = annualMeanSoil;
  soilLandMaskCache = soilLandMask;
  return { temp: annualMeanTemp, soil: annualMeanSoil, soilLandMask };
}

// ---------------------------------------------------------------------------
// Temperature color buffer
// ---------------------------------------------------------------------------

export function buildTemperatureColorBuffer(climateData: number[][][], monthIdx: number): Float32Array {
  const nlat = climateData[0].length;
  const nlon = climateData[0][0].length;
  const monthData = averagePolarTemps(climateData[monthIdx], nlat, nlon);
  const nVerts = nlat * nlon * 6;
  const buf = new Float32Array(nVerts * 3);

  let idx = 0;
  for (let i = 0; i < nlat; i++) {
    const dataLatIdx = nlat - 1 - i;
    for (let j = 0; j < nlon; j++) {
      const temp = monthData[dataLatIdx][j];
      const [r, g, b] = temperatureToColorNormalized(temp);
      for (let v = 0; v < 6; v++) {
        buf[idx++] = r; buf[idx++] = g; buf[idx++] = b;
      }
    }
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Precipitation color buffer
// ---------------------------------------------------------------------------

export function buildPrecipitationColorBuffer(ld: ClimateLayerData, monthIdx: number): Float32Array {
  const precip = ld.precipitation;
  if (!precip) return new Float32Array(0);
  const pNlat = precip.shape[1];
  const pNlon = precip.shape[2];
  const precipData = precip.data as Float32Array;
  const nVerts = pNlat * pNlon * 6;
  const buf = new Float32Array(nVerts * 3);
  const monthOffset = monthIdx * pNlat * pNlon;

  let idx = 0;
  for (let i = 0; i < pNlat; i++) {
    const dataLatIdx = pNlat - 1 - i;
    for (let j = 0; j < pNlon; j++) {
      const val = precipData[monthOffset + dataLatIdx * pNlon + j];
      const [r, g, b] = precipitationToColorNormalized(val);
      for (let v = 0; v < 6; v++) {
        buf[idx++] = r; buf[idx++] = g; buf[idx++] = b;
      }
    }
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Blue marble buffers (RGB + specular)
// ---------------------------------------------------------------------------

export function buildBlueMarbleBuffers(ld: ClimateLayerData, monthIdx: number, snowMonthIdx?: number): { rgb: Float32Array; spec: Float32Array } {
  const surfaceData = ld.surface.data as Float32Array;
  const landMaskData = ld.land_mask.data as Uint8Array;
  const soilData = ld.soil_moisture?.data as Float32Array | undefined;
  const vegData = ld.vegetation_fraction?.data as Float32Array | undefined;
  const coarseLandMask = ld.land_mask_native?.data as Uint8Array | undefined;
  const elevData = ld.elevation?.data as Float32Array | undefined;
  const elevNlat = ld.elevation?.shape[0] ?? 0;
  const elevNlon = ld.elevation?.shape[1] ?? 0;
  const snowTempRaw = ld.snow_temperature?.data as Uint8Array | Float32Array | undefined;
  const snowNlat = ld.snow_temperature?.shape[1] ?? 0;
  const snowNlon = ld.snow_temperature?.shape[2] ?? 0;
  const snowMonthOff = (snowMonthIdx ?? monthIdx) * snowNlat * snowNlon;

  const hiNlat = ld.land_mask.shape[0];
  const hiNlon = ld.land_mask.shape[1];

  const lowNlat = ld.surface.shape[1];
  const lowNlon = ld.surface.shape[2];
  const monthOffset = monthIdx * lowNlat * lowNlon;

  const soilNlat = ld.soil_moisture?.shape[1] ?? lowNlat;
  const soilNlon = ld.soil_moisture?.shape[2] ?? lowNlon;
  const soilMonthOffset = monthIdx * soilNlat * soilNlon;

  const annuals = getAnnualMeans(ld);
  const annualMeanTemp = annuals.temp;
  const annualMeanSoil = annuals.soil;
  const soilLandMask = annuals.soilLandMask;

  // First pass: compute per-cell RGB into a flat buffer
  const rgbBuf = new Float32Array(hiNlat * hiNlon * 3);
  for (let i = 0; i < hiNlat; i++) {
    const dataLatIdx = hiNlat - 1 - i;
    for (let j = 0; j < hiNlon; j++) {
      const isLand = landMaskData[dataLatIdx * hiNlon + j] === 1;

      const surfaceTemp = sampleBilinear(
        surfaceData, lowNlat, lowNlon, monthOffset,
        dataLatIdx, j, hiNlat, hiNlon,
        isLand, coarseLandMask,
      );
      const soilMoisture = soilData
        ? sampleBilinear(
            soilData, soilNlat, soilNlon, soilMonthOffset,
            dataLatIdx, j, hiNlat, hiNlon,
            isLand, soilLandMask,
          )
        : 0;

      let elev = 0;
      if (elevData && elevNlat > 0) {
        const ei = Math.floor(dataLatIdx * elevNlat / hiNlat);
        const ej = Math.floor(j * elevNlon / hiNlon);
        elev = elevData[ei * elevNlon + ej];
      }

      const vegFrac = vegData
        ? sampleBilinear(
            vegData, lowNlat, lowNlon, monthOffset,
            dataLatIdx, j, hiNlat, hiNlon,
            isLand, coarseLandMask,
          )
        : 0;

      const annMeanT = sampleBilinear(
        annualMeanTemp, lowNlat, lowNlon, 0,
        dataLatIdx, j, hiNlat, hiNlon,
        isLand, coarseLandMask,
      );
      const annMeanSoil = sampleBilinear(
        annualMeanSoil, soilNlat, soilNlon, 0,
        dataLatIdx, j, hiNlat, hiNlon,
        isLand, soilLandMask,
      );

      let snowTempC = surfaceTemp;
      if (snowTempRaw && isLand) {
        const si = Math.floor(dataLatIdx * snowNlat / hiNlat);
        const sj = Math.floor(j * snowNlon / hiNlon);
        const raw = snowTempRaw[snowMonthOff + si * snowNlon + sj];
        snowTempC = raw * (120.0 / 255.0) - 60.0;
      }

      const [r, g, b] = blueMarbleColor(isLand, surfaceTemp, soilMoisture, elev, vegFrac, annMeanT, annMeanSoil, snowTempC);
      const base = (i * hiNlon + j) * 3;
      rgbBuf[base] = r;
      rgbBuf[base + 1] = g;
      rgbBuf[base + 2] = b;
    }
  }

  // Second pass: polar smoothing on the high-res RGB rows
  const latStep = 180 / hiNlat;
  for (let i = 0; i < hiNlat; i++) {
    const lat = 90 - (i + 0.5) * latStep;
    const absLat = Math.abs(lat);
    if (absLat <= 75) continue;

    const t = (absLat - 75) / 15;
    const maxWindow = hiNlon / 24;
    let windowSize = 1 + t * t * maxWindow;
    windowSize = Math.floor(windowSize);
    if (windowSize < 2) continue;
    if (windowSize % 2 === 0) windowSize++;
    const halfWindow = Math.floor(windowSize / 2);
    const rowBase = i * hiNlon * 3;

    for (let ch = 0; ch < 3; ch++) {
      const tmp = new Float32Array(hiNlon);
      for (let j = 0; j < hiNlon; j++) {
        let sum = 0;
        for (let k = -halfWindow; k <= halfWindow; k++) {
          const idx = ((j + k) % hiNlon + hiNlon) % hiNlon;
          sum += rgbBuf[rowBase + idx * 3 + ch];
        }
        tmp[j] = sum / windowSize;
      }
      for (let j = 0; j < hiNlon; j++) {
        rgbBuf[rowBase + j * 3 + ch] = tmp[j];
      }
    }
  }

  // Apply hillshade to RGB buffer and build specular + vertex-expanded buffers
  const hsGrid = getHillshadeGrid(ld);
  const nVerts = hiNlat * hiNlon * 6;
  const outRgb = new Float32Array(nVerts * 3);
  const outSpec = new Float32Array(nVerts);
  let idx = 0;
  for (let i = 0; i < hiNlat; i++) {
    const dataLatIdx = hiNlat - 1 - i;
    for (let j = 0; j < hiNlon; j++) {
      const base = (i * hiNlon + j) * 3;
      let r = rgbBuf[base];
      let g = rgbBuf[base + 1];
      let b = rgbBuf[base + 2];
      const isLand = landMaskData[dataLatIdx * hiNlon + j] === 1;
      if (hsGrid && isLand) {
        const hs = hsGrid[i * hiNlon + j];
        r = Math.min(1, r * hs);
        g = Math.min(1, g * hs);
        b = Math.min(1, b * hs);
      }

      let spec = 0.0;
      if (!isLand) {
        const temp = sampleBilinear(
          surfaceData, lowNlat, lowNlon, monthOffset,
          dataLatIdx, j, hiNlat, hiNlon,
          false, coarseLandMask,
        );
        const ICE_FREEZE = -1.8, ICE_FULL = -8.0, ICE_MAX = 0.70;
        const u = Math.max(0, Math.min(1, (ICE_FREEZE - temp) / (ICE_FREEZE - ICE_FULL)));
        const iceFrac = u * u * (3 - 2 * u) * ICE_MAX;
        spec = 1.0 - iceFrac;
      }

      for (let v = 0; v < 6; v++) {
        outRgb[idx * 3] = r; outRgb[idx * 3 + 1] = g; outRgb[idx * 3 + 2] = b;
        outSpec[idx] = spec;
        idx++;
      }
    }
  }
  return { rgb: outRgb, spec: outSpec };
}
