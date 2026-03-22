/**
 * Web Worker for building color cache buffers off the main thread.
 * Receives climate typed arrays + grid dimensions, builds all 12 base months
 * of temperature color buffers and blue marble RGB/spec buffers.
 * Returns results via Transferable (zero-copy).
 */

import { temperatureToColorNormalized } from './colormap';
import { blueMarbleColor } from './blueMarbleColormap';
import { sampleBilinear, averagePolarTemps } from './gridSampling';
import { computeHillshadeGrid } from './elevation';

interface LayerInfo {
  // All typed arrays transferred in
  surfaceData?: Float32Array;
  surfaceShape?: number[];      // [12, lowNlat, lowNlon]
  landMaskData: Uint8Array;
  landMaskShape: number[];      // [hiNlat, hiNlon]
  coarseLandMask?: Uint8Array;  // native 5deg land mask
  soilData?: Float32Array;
  soilShape?: number[];         // [12, sNlat, sNlon]
  landMask1deg?: Uint8Array;
  landMask1degShape?: number[];
  vegData?: Float32Array;
  vegShape?: number[];
  elevData?: Float32Array;
  elevShape?: number[];         // [elevNlat, elevNlon]
  snowTempData?: Uint8Array | Float32Array;
  snowTempShape?: number[];     // [12, snowNlat, snowNlon]
}

interface TempInfo {
  // Flat Float32Array of temperature_2m [12, nlat, nlon]
  data: Float32Array;
  nlat: number;
  nlon: number;
}

interface WorkerInput {
  layers: LayerInfo;
  temp: TempInfo;
}

function computeAnnualMeans(layers: LayerInfo): { temp: Float32Array; soil: Float32Array; soilLandMask: Uint8Array } {
  const { surfaceData, surfaceShape, soilData, soilShape, landMaskData, landMaskShape, landMask1deg, landMask1degShape } = layers;
  const lowNlat = surfaceShape[1];
  const lowNlon = surfaceShape[2];
  const sNlat = soilShape ? soilShape[1] : lowNlat;
  const sNlon = soilShape ? soilShape[2] : lowNlon;

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

  let soilLandMask: Uint8Array;
  if (landMask1deg) {
    soilLandMask = landMask1deg;
  } else {
    const hiNlat = landMaskShape[0];
    const hiNlon = landMaskShape[1];
    soilLandMask = new Uint8Array(sNlat * sNlon);
    const latRatio = hiNlat / sNlat;
    const lonRatio = hiNlon / sNlon;
    for (let ci = 0; ci < sNlat; ci++) {
      const hi0 = Math.floor(ci * latRatio);
      const hi1 = Math.min(Math.floor((ci + 1) * latRatio), hiNlat);
      for (let cj = 0; cj < sNlon; cj++) {
        const hj0 = Math.floor(cj * lonRatio);
        const hj1 = Math.min(Math.floor((cj + 1) * lonRatio), hiNlon);
        let landCount = 0;
        for (let ii = hi0; ii < hi1; ii++) {
          for (let jj = hj0; jj < hj1; jj++) {
            if (landMaskData[ii * hiNlon + jj] === 1) landCount++;
          }
        }
        soilLandMask[ci * sNlon + cj] = landCount > 0 ? 1 : 0;
      }
    }
  }

  return { temp: annualMeanTemp, soil: annualMeanSoil, soilLandMask };
}

function buildTemperatureColorBuffer(
  tempData: Float32Array, nlat: number, nlon: number, monthIdx: number,
): Float32Array {
  // Extract month slice as number[][]
  const monthBase = monthIdx * nlat * nlon;
  const monthArr: number[][] = [];
  for (let i = 0; i < nlat; i++) {
    const row: number[] = [];
    for (let j = 0; j < nlon; j++) {
      row.push(tempData[monthBase + i * nlon + j]);
    }
    monthArr.push(row);
  }
  const monthData = averagePolarTemps(monthArr, nlat, nlon);
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

function buildBlueMarbleBuffers(
  layers: LayerInfo,
  monthIdx: number,
  annualMeanTemp: Float32Array,
  annualMeanSoil: Float32Array,
  soilLandMask: Uint8Array,
  hillshadeGrid: Float32Array | null,
): { rgb: Float32Array; spec: Float32Array } {
  const { surfaceData, surfaceShape, landMaskData, landMaskShape,
    soilData, soilShape, vegData, coarseLandMask,
    elevData, elevShape, snowTempData, snowTempShape } = layers;

  const hiNlat = landMaskShape[0];
  const hiNlon = landMaskShape[1];
  const lowNlat = surfaceShape[1];
  const lowNlon = surfaceShape[2];
  const monthOffset = monthIdx * lowNlat * lowNlon;
  const soilNlat = soilShape ? soilShape[1] : lowNlat;
  const soilNlon = soilShape ? soilShape[2] : lowNlon;
  const soilMonthOffset = monthIdx * soilNlat * soilNlon;
  const elevNlat = elevShape ? elevShape[0] : 0;
  const elevNlon = elevShape ? elevShape[1] : 0;
  const snowNlat = snowTempShape ? snowTempShape[1] : 0;
  const snowNlon = snowTempShape ? snowTempShape[2] : 0;
  const snowMonthOff = monthIdx * snowNlat * snowNlon;

  // First pass: per-cell RGB
  const rgbBuf = new Float32Array(hiNlat * hiNlon * 3);
  for (let i = 0; i < hiNlat; i++) {
    const dataLatIdx = hiNlat - 1 - i;
    for (let j = 0; j < hiNlon; j++) {
      const isLand = landMaskData[dataLatIdx * hiNlon + j] === 1;

      const surfaceTemp = sampleBilinear(
        surfaceData, lowNlat, lowNlon, monthOffset,
        dataLatIdx, j, hiNlat, hiNlon, isLand, coarseLandMask,
      );
      const soilMoisture = soilData
        ? sampleBilinear(soilData, soilNlat, soilNlon, soilMonthOffset,
            dataLatIdx, j, hiNlat, hiNlon, isLand, soilLandMask)
        : 0;

      let elev = 0;
      if (elevData && elevNlat > 0) {
        const ei = Math.floor(dataLatIdx * elevNlat / hiNlat);
        const ej = Math.floor(j * elevNlon / hiNlon);
        elev = elevData[ei * elevNlon + ej];
      }

      const vegFrac = vegData
        ? sampleBilinear(vegData, lowNlat, lowNlon, monthOffset,
            dataLatIdx, j, hiNlat, hiNlon, isLand, coarseLandMask)
        : 0;

      const annMeanT = sampleBilinear(
        annualMeanTemp, lowNlat, lowNlon, 0,
        dataLatIdx, j, hiNlat, hiNlon, isLand, coarseLandMask,
      );
      const annMeanSoil = sampleBilinear(
        annualMeanSoil, soilNlat, soilNlon, 0,
        dataLatIdx, j, hiNlat, hiNlon, isLand, soilLandMask,
      );

      let snowTempC = surfaceTemp;
      if (snowTempData && isLand) {
        const si = Math.floor(dataLatIdx * snowNlat / hiNlat);
        const sj = Math.floor(j * snowNlon / hiNlon);
        const raw = snowTempData[snowMonthOff + si * snowNlon + sj];
        snowTempC = raw * (120.0 / 255.0) - 60.0;
      }

      const [r, g, b] = blueMarbleColor(isLand, surfaceTemp, soilMoisture, elev, vegFrac, annMeanT, annMeanSoil, snowTempC);
      const base = (i * hiNlon + j) * 3;
      rgbBuf[base] = r;
      rgbBuf[base + 1] = g;
      rgbBuf[base + 2] = b;
    }
  }

  // Second pass: polar smoothing
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

  // Third pass: hillshade + vertex expansion
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
      if (hillshadeGrid && isLand) {
        const hs = hillshadeGrid[i * hiNlon + j];
        r = Math.min(1, r * hs);
        g = Math.min(1, g * hs);
        b = Math.min(1, b * hs);
      }

      let spec = 0.0;
      if (!isLand) {
        const temp = sampleBilinear(
          surfaceData, lowNlat, lowNlon, monthOffset,
          dataLatIdx, j, hiNlat, hiNlon, false, coarseLandMask,
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

self.onmessage = (e: MessageEvent<WorkerInput>) => {
  const { layers, temp } = e.data;
  const transferables: ArrayBuffer[] = [];

  // Pre-compute shared data (only if surface data exists for blue marble)
  const hasSurface = !!layers.surfaceData && !!layers.surfaceShape;
  const annuals = hasSurface ? computeAnnualMeans(layers) : null;

  // Hillshade grid (computed once, shared across months)
  let hillshadeGrid: Float32Array | null = null;
  if (hasSurface && layers.elevData && layers.elevShape) {
    const hiNlat = layers.landMaskShape[0];
    const hiNlon = layers.landMaskShape[1];
    hillshadeGrid = computeHillshadeGrid(
      layers.elevData, layers.elevShape[0], layers.elevShape[1], hiNlat, hiNlon,
    );
  }

  // Build all 12 base months
  const tempBuffers: Float32Array[] = [];
  const bmRgbBuffers: (Float32Array | null)[] = [];
  const bmSpecBuffers: (Float32Array | null)[] = [];

  for (let m = 0; m < 12; m++) {
    // Temperature color buffer
    const tb = buildTemperatureColorBuffer(temp.data, temp.nlat, temp.nlon, m);
    tempBuffers.push(tb);
    transferables.push(tb.buffer);

    // Blue marble buffers (only when surface data is available)
    if (hasSurface && annuals) {
      const bm = buildBlueMarbleBuffers(
        layers, m, annuals.temp, annuals.soil, annuals.soilLandMask, hillshadeGrid,
      );
      bmRgbBuffers.push(bm.rgb);
      bmSpecBuffers.push(bm.spec);
      transferables.push(bm.rgb.buffer);
      transferables.push(bm.spec.buffer);
    } else {
      bmRgbBuffers.push(null);
      bmSpecBuffers.push(null);
    }

    // Post progress after each month
    self.postMessage({ type: 'progress', month: m }, []);
  }

  self.postMessage({
    type: 'done',
    tempBuffers,
    bmRgbBuffers,
    bmSpecBuffers,
  }, transferables as any);
};
