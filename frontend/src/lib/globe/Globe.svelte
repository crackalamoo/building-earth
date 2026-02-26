<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { temperatureToColorNormalized } from './colormap';
  import { blueMarbleColor } from './blueMarbleColormap';
  import { sampleBilinear, averagePolarTemps } from './gridSampling';
  import { loadBorders } from './borders';
  import { WindParticles } from './WindParticles';
  import { TreeInstances } from './TreeInstances';
  import { CloudInstances } from './CloudInstances';
  import { createAtmosphere, updateAtmosphereSunDirection } from './Atmosphere';
  import { createStarField, updateStarRotation, type StarField } from './Stars';
  import { CityLights } from './CityLights';
  import type { ClimateLayerData } from './loadBinaryData';
  import { ELEVATION_SCALE, NORMAL_BLEND, sampleElevation, displacedNormal, computeHillshadeGrid } from './elevation';
  import { getSunDeclination, createSunOrb, createSunBloom, updateSunBloom, disposeSun } from './Sun';

  export let data: number[][][] | null = null; // [month][lat][lon] temperature in Celsius
  export let monthProgress: number = 0; // Continuous 0-12 (wraps), controls sun position
  export let showBorders: boolean = true;
  export let activeLayer: 'temperature' | 'blue-marble' = 'temperature';
  export let layerData: ClimateLayerData | null = null;
  export let uniformLighting: boolean = false;
  export let primordialLandMask: { data: Uint8Array; nlat: number; nlon: number } | null = null;
  export let revealed: boolean = false;

  const dispatch = createEventDispatcher();

  let container: HTMLDivElement;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let globe: THREE.Mesh | null = null;
  let blueMarbleGlobe: THREE.Mesh | null = null;
  let bordersGroup: THREE.Group | null = null;
  let animationId: number;
  let sunLight: THREE.DirectionalLight;
  let displayMonthProgress = 0; // Smoothly interpolates toward target
  let sunOrbitAngle = 0; // Rotates sun when camera is not auto-rotating
  let lastAnimateTime: number | null = null;
  let windParticles: WindParticles | null = null;
  let treeInstances: TreeInstances | null = null;
  let cloudInstances: CloudInstances | null = null;
  let atmosphereMesh: THREE.Mesh | null = null;
  let sunOrb: THREE.Mesh | null = null;
  let sunGlow: THREE.Object3D | null = null;
  let starField: StarField | null = null;
  let cityLights: CityLights | null = null;
  let markerMesh: THREE.Mesh | null = null;
  let markerWorldPos: THREE.Vector3 | null = null;
  let raycaster = new THREE.Raycaster();
  let mouseDownPos = { x: 0, y: 0 };
  let primordialGlobe: THREE.Mesh | null = null;
  let flashQuad: THREE.Mesh | null = null;
  let flashStartTime: number | null = null;  // wall-clock ms when bloom begins
  let flashFadeStart: number | null = null;  // wall-clock ms when fade-out begins
  const FLASH_BLOOM_DURATION = 0.6;  // seconds to bloom in
  const FLASH_FADE_DURATION = 1.5;   // seconds to fade out

  // Cached color buffers: 12 base months + sub-step interpolations
  const SUB_STEPS = 3;
  const TOTAL_STEPS = 12 * SUB_STEPS;
  let tempBaseCache: (Float32Array | null)[] = new Array(12).fill(null);
  let bmBaseRgbCache: (Float32Array | null)[] = new Array(12).fill(null);
  let bmBaseSpecCache: (Float32Array | null)[] = new Array(12).fill(null);
  let tempStepCache: (Float32Array | null)[] = new Array(TOTAL_STEPS).fill(null);
  let bmStepRgbCache: (Float32Array | null)[] = new Array(TOTAL_STEPS).fill(null);
  let bmStepSpecCache: (Float32Array | null)[] = new Array(TOTAL_STEPS).fill(null);
  let lastAppliedStep = -1;
  let warmupGeneration = 0;

  // Derive discrete month for temperature display (nearest month)
  $: displayMonth = Math.round(displayMonthProgress) % 12;

  // Expose renderer for GIF capture
  export function getCanvas(): HTMLCanvasElement | null {
    return renderer?.domElement ?? null;
  }

  export function renderFrame(): void {
    if (renderer && scene && camera) {
      renderer.render(scene, camera);
    }
  }

  export function rotateGlobe(radians: number): void {
    if (globe) {
      globe.rotation.y += radians;
    }
    if (blueMarbleGlobe) {
      blueMarbleGlobe.rotation.y += radians;
    }
    if (bordersGroup) {
      bordersGroup.rotation.y += radians;
    }
    if (windParticles) {
      windParticles.getObject().rotation.y += radians;
    }
    if (treeInstances) {
      treeInstances.getObject().rotation.y += radians;
    }
    if (cloudInstances) {
      cloudInstances.getObject().rotation.y += radians;
    }
    if (cityLights) {
      cityLights.getObject().rotation.y += radians;
    }
  }

  export function resetView(): void {
    if (controls) {
      controls.reset();
      controls.autoRotate = true;
    }
    if (globe) {
      globe.rotation.y = 0;
    }
    if (blueMarbleGlobe) {
      blueMarbleGlobe.rotation.y = 0;
    }
    if (bordersGroup) {
      bordersGroup.rotation.y = 0;
    }
    if (windParticles) {
      windParticles.getObject().rotation.y = 0;
    }
    if (treeInstances) {
      treeInstances.getObject().rotation.y = 0;
    }
    if (cloudInstances) {
      cloudInstances.getObject().rotation.y = 0;
    }
    if (cityLights) {
      cityLights.getObject().rotation.y = 0;
    }
    sunOrbitAngle = 0;
  }

  export function setAutoRotate(enabled: boolean): void {
    if (controls) {
      controls.autoRotate = enabled;
    }
  }

  export function isAutoRotating(): boolean {
    return controls?.autoRotate ?? false;
  }

  // Derived grid dimensions from data
  $: nlat = data ? data[0].length : 0;
  $: nlon = data ? data[0][0].length : 0;

  // Background cache worker — builds all 12 base months off-thread
  let cacheWorker: Worker | null = null;
  let cacheWorkerReady = false; // true when worker has finished

  function launchCacheWorker() {
    if (cacheWorker) { cacheWorker.terminate(); cacheWorker = null; }
    cacheWorkerReady = false;
    if (!data || !layerData) return;

    // Gather typed arrays from layerData (read-only, no transfer — worker gets copies via structured clone)
    const ld = layerData;
    const t2m = ld.temperature_2m;
    const layers = {
      surfaceData: ld.surface.data as Float32Array,
      surfaceShape: ld.surface.shape,
      landMaskData: ld.land_mask.data as Uint8Array,
      landMaskShape: ld.land_mask.shape,
      coarseLandMask: ld.land_mask_native?.data as Uint8Array | undefined,
      soilData: ld.soil_moisture?.data as Float32Array | undefined,
      soilShape: ld.soil_moisture?.shape,
      landMask1deg: ld.land_mask_1deg?.data as Uint8Array | undefined,
      landMask1degShape: ld.land_mask_1deg?.shape,
      vegData: ld.vegetation_fraction?.data as Float32Array | undefined,
      vegShape: ld.vegetation_fraction?.shape,
      elevData: ld.elevation?.data as Float32Array | undefined,
      elevShape: ld.elevation?.shape,
      snowTempData: ld.snow_temperature?.data as Uint8Array | Float32Array | undefined,
      snowTempShape: ld.snow_temperature?.shape,
    };
    const temp = {
      data: t2m.data as Float32Array,
      nlat: t2m.shape[1],
      nlon: t2m.shape[2],
    };

    const worker = new Worker(new URL('./cacheWorker.ts', import.meta.url), { type: 'module' });
    cacheWorker = worker;
    const gen = ++warmupGeneration;

    worker.onmessage = (e: MessageEvent) => {
      if (gen !== warmupGeneration) { worker.terminate(); return; }
      if (e.data.type === 'progress') return; // ignore progress for now
      if (e.data.type === 'done') {
        const { tempBuffers, bmRgbBuffers, bmSpecBuffers } = e.data;
        for (let m = 0; m < 12; m++) {
          tempBaseCache[m] = tempBuffers[m];
          bmBaseRgbCache[m] = bmRgbBuffers[m];
          bmBaseSpecCache[m] = bmSpecBuffers[m];
        }
        // Build sub-steps (cheap, synchronous)
        for (let step = 0; step < TOTAL_STEPS; step++) {
          if (step % SUB_STEPS !== 0) {
            ensureTempStep(step);
            ensureBmStep(step);
          }
        }
        cacheWorkerReady = true;
        worker.terminate();
        cacheWorker = null;
      }
    };
    worker.onerror = () => { worker.terminate(); cacheWorker = null; };
    worker.postMessage({ layers, temp });
  }

  // Invalidate caches and launch worker as soon as both data sources are available
  $: if (data) {
    tempBaseCache = new Array(12).fill(null);
    tempStepCache = new Array(TOTAL_STEPS).fill(null);
    lastAppliedStep = -1;
  }
  $: if (layerData) {
    hillshadeGridCache = null;
    annualMeanTempCache = null;
    annualMeanSoilCache = null;
    soilLandMaskCache = null;
    bmBaseRgbCache = new Array(12).fill(null);
    bmBaseSpecCache = new Array(12).fill(null);
    bmStepRgbCache = new Array(TOTAL_STEPS).fill(null);
    bmStepSpecCache = new Array(TOTAL_STEPS).fill(null);
    lastAppliedStep = -1;
  }
  // Launch cache worker as soon as both data sources exist (even before reveal)
  $: if (data && layerData) {
    launchCacheWorker();
  }

  let ambientLight: THREE.AmbientLight;

  // Toggle visibility when activeLayer changes
  // Reference activeLayer before the if so Svelte tracks it as a dependency
  $: {
    const layer = activeLayer;
    if (scene) {
      if (globe) globe.visible = layer === 'temperature';
      if (blueMarbleGlobe) blueMarbleGlobe.visible = layer === 'blue-marble';
      if (windParticles) windParticles.getObject().visible = layer === 'blue-marble';
      if (treeInstances) treeInstances.getObject().visible = layer === 'blue-marble';
      if (cloudInstances) cloudInstances.getObject().visible = layer === 'blue-marble';
      if (cityLights) cityLights.getObject().visible = layer === 'blue-marble';
      if (atmosphereMesh) atmosphereMesh.visible = layer === 'blue-marble';
    }
  }

  function buildTemperatureColorBuffer(climateData: number[][][], monthIdx: number): Float32Array {
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

  /** Cached hillshade grid (computed from elevation, independent of globe mesh). */
  let hillshadeGridCache: Float32Array | null = null;

  function getHillshadeGrid(ld: ClimateLayerData): Float32Array | null {
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

  /** Compute annual means from full 12-month data (cached to avoid recomputation). */
  let annualMeanTempCache: Float32Array | null = null;
  let annualMeanSoilCache: Float32Array | null = null;
  let soilLandMaskCache: Uint8Array | null = null;

  function getAnnualMeans(ld: ClimateLayerData): { temp: Float32Array; soil: Float32Array; soilLandMask: Uint8Array } {
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

    // Use exported 1° land mask if available, otherwise downsample from 0.25°
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

  function buildBlueMarbleBuffers(ld: ClimateLayerData, monthIdx: number, snowMonthIdx?: number): { rgb: Float32Array; spec: Float32Array } {
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

    // High-res grid dimensions (from land_mask, 0.25deg)
    const hiNlat = ld.land_mask.shape[0];
    const hiNlon = ld.land_mask.shape[1];

    // Low-res grid dimensions (from surface, 5deg)
    const lowNlat = ld.surface.shape[1];
    const lowNlon = ld.surface.shape[2];
    const monthOffset = monthIdx * lowNlat * lowNlon;

    // Soil moisture may be at a different resolution (e.g. 1deg = 180x360)
    const soilNlat = ld.soil_moisture?.shape[1] ?? lowNlat;
    const soilNlon = ld.soil_moisture?.shape[2] ?? lowNlon;
    const soilMonthOffset = monthIdx * soilNlat * soilNlon;

    // Use cached annual means (computed from full 12-month original data)
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

        // Sample elevation for bathymetry shading
        let elev = 0;
        if (elevData && elevNlat > 0) {
          // Map hi-res grid index to elevation grid index
          const ei = Math.floor(dataLatIdx * elevNlat / hiNlat);
          const ej = Math.floor(j * elevNlon / hiNlon);
          elev = elevData[ei * elevNlon + ej];
        }

        // Sample vegetation fraction (same grid as soil moisture)
        const vegFrac = vegData
          ? sampleBilinear(
              vegData, lowNlat, lowNlon, monthOffset,
              dataLatIdx, j, hiNlat, hiNlon,
              isLand, coarseLandMask,
            )
          : 0;

        // Sample annual means bilinearly (offset=0, single 2D fields)
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

        // Snow temperature from backend (interpolated surface + lapse correction)
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

    // Second pass: polar smoothing on the high-res RGB rows (same logic as averagePolarTemps)
    const latStep = 180 / hiNlat;
    for (let i = 0; i < hiNlat; i++) {
      const lat = 90 - (i + 0.5) * latStep; // i=0 is north pole in render order
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

      // Smooth each channel independently using a temp buffer for the row
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
    const hsGrid = getHillshadeGrid(layerData!);
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

        // Compute specular mask: 0 for land/ice, 1 for open ocean
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

  function ensureTempBase(m: number) {
    if (!tempBaseCache[m] && data) tempBaseCache[m] = buildTemperatureColorBuffer(data, m);
  }

  function ensureBmBase(m: number) {
    if (!bmBaseRgbCache[m] && layerData) {
      const r = buildBlueMarbleBuffers(layerData, m);
      bmBaseRgbCache[m] = r.rgb; bmBaseSpecCache[m] = r.spec;
    }
  }

  function ensureTempStep(step: number) {
    if (tempStepCache[step]) return;
    // Snap to nearest base month (no sub-step interpolation for hi-res T2m)
    const m0 = Math.floor(step / SUB_STEPS) % 12;
    const t = (step % SUB_STEPS) / SUB_STEPS;
    const nearest = t < 0.5 ? m0 : (m0 + 1) % 12;
    ensureTempBase(nearest);
    tempStepCache[step] = tempBaseCache[nearest]!;
  }

  function ensureBmStep(step: number) {
    if (bmStepRgbCache[step]) return;
    // Snap to nearest base month (no sub-step interpolation to save memory)
    const m0 = Math.floor(step / SUB_STEPS) % 12;
    const t = (step % SUB_STEPS) / SUB_STEPS;
    const nearest = t < 0.5 ? m0 : (m0 + 1) % 12;
    ensureBmBase(nearest);
    bmStepRgbCache[step] = bmBaseRgbCache[nearest]!;
    bmStepSpecCache[step] = bmBaseSpecCache[nearest]!;
  }

  function progressToStep(progress: number): number {
    return Math.round(progress * SUB_STEPS) % TOTAL_STEPS;
  }

  /** Apply cached sub-step colors to temperature globe. */
  function applyTemperatureColors(step: number) {
    if (!globe || !data) return;
    ensureTempStep(step);
    const colors = globe.geometry.attributes.color;
    ((colors as any).array as Float32Array).set(tempStepCache[step]!);
    colors.needsUpdate = true;
  }

  /** Apply cached sub-step colors to blue marble globe. */
  function applyBlueMarbleColors(step: number) {
    if (!blueMarbleGlobe || !layerData) return;
    ensureBmStep(step);
    const geometry = blueMarbleGlobe.geometry;
    const colors = geometry.attributes.color;
    ((colors as any).array as Float32Array).set(bmStepRgbCache[step]!);
    colors.needsUpdate = true;
    const specAttr = geometry.attributes.isOcean as THREE.BufferAttribute | undefined;
    if (specAttr) {
      ((specAttr as any).array as Float32Array).set(bmStepSpecCache[step]!);
      specAttr.needsUpdate = true;
    }
  }

  function createGlobeMesh(
    latCount: number, lonCount: number, radius: number,
    elevationData?: Float32Array, elevNlat?: number, elevNlon?: number,
    landMaskData?: Uint8Array, maskNlat?: number, maskNlon?: number,
  ): THREE.Mesh {
    // Build geometry manually with per-face colors (no interpolation)
    const positions: number[] = [];
    const colors: number[] = [];

    function getVertex(lat: number, lon: number): [number, number, number] {
      let r = radius;
      if (elevationData && elevNlat && elevNlon) {
        const elev = sampleElevation(elevationData, elevNlat, elevNlon, lat, lon);
        r += Math.max(0, elev) * ELEVATION_SCALE;
      }
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = lon * (Math.PI / 180);
      return [
        -r * Math.sin(phi) * Math.cos(theta),
        r * Math.cos(phi),
        r * Math.sin(phi) * Math.sin(theta)
      ];
    }

    const blendedNormals: number[] = [];

    function addVertex(lat: number, lon: number, r: number, g: number, b: number, isLand: boolean) {
      const [x, y, z] = getVertex(lat, lon);
      positions.push(x, y, z);
      colors.push(r, g, b);

      const phi = (90 - lat) * (Math.PI / 180);
      const theta = lon * (Math.PI / 180);
      const snx = -Math.sin(phi) * Math.cos(theta);
      const sny = Math.cos(phi);
      const snz = Math.sin(phi) * Math.sin(theta);

      // Ocean: use pure sphere normals (bathymetry shouldn't affect lighting)
      if (elevationData && elevNlat && elevNlon && isLand) {
        const [dnx, dny, dnz] = displacedNormal(elevationData, elevNlat, elevNlon, lat, lon, radius);
        let nx = snx + NORMAL_BLEND * (dnx - snx);
        let ny = sny + NORMAL_BLEND * (dny - sny);
        let nz = snz + NORMAL_BLEND * (dnz - snz);
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        blendedNormals.push(nx / len, ny / len, nz / len);
      } else {
        blendedNormals.push(snx, sny, snz);
      }
    }

    const latStep = 180 / latCount;
    const lonStep = 360 / lonCount;

    // Pre-compute per-cell hillshade if elevation data provided
    const hillshadeGrid = (elevationData && elevNlat && elevNlon)
      ? computeHillshadeGrid(elevationData, elevNlat, elevNlon, latCount, lonCount)
      : null;

    for (let i = 0; i < latCount; i++) {
      const lat0 = 90 - i * latStep;
      const lat1 = 90 - (i + 1) * latStep;
      const dataLatIdx = latCount - 1 - i;

      for (let j = 0; j < lonCount; j++) {
        const lon0 = j * lonStep;
        const lon1 = (j + 1) * lonStep;

        // Check if this cell is land for normal computation
        const cellIsLand = (landMaskData && maskNlat && maskNlon)
          ? landMaskData[dataLatIdx * maskNlon + j] === 1
          : false;

        // Default color (will be updated immediately)
        const r = 0.1, g = 0.1, b = 0.1;

        addVertex(lat0, lon0, r, g, b, cellIsLand);
        addVertex(lat1, lon0, r, g, b, cellIsLand);
        addVertex(lat1, lon1, r, g, b, cellIsLand);

        addVertex(lat0, lon0, r, g, b, cellIsLand);
        addVertex(lat1, lon1, r, g, b, cellIsLand);
        addVertex(lat0, lon1, r, g, b, cellIsLand);
      }
    }

    // Build per-vertex isOcean attribute if land mask provided
    const isOceanArr: number[] = [];
    if (landMaskData && maskNlat && maskNlon) {
      const latStep2 = 180 / latCount;
      const lonStep2 = 360 / lonCount;
      for (let i = 0; i < latCount; i++) {
        const lat0 = 90 - i * latStep2;
        const lat1 = 90 - (i + 1) * latStep2;
        const cellLat = (lat0 + lat1) / 2;
        // Convert to data index (data is south-to-north)
        const dataLatIdx = latCount - 1 - i;
        for (let j = 0; j < lonCount; j++) {
          const lon0 = j * (360 / lonCount);
          const lon1 = (j + 1) * (360 / lonCount);
          const isLand = landMaskData[dataLatIdx * maskNlon + j] === 1;
          const val = isLand ? 0.0 : 1.0;
          for (let v = 0; v < 6; v++) isOceanArr.push(val);
        }
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('normal', new THREE.Float32BufferAttribute(blendedNormals, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    if (isOceanArr.length > 0) {
      geometry.setAttribute('isOcean', new THREE.Float32BufferAttribute(isOceanArr, 1));
    }

    const material = new THREE.MeshLambertMaterial({
      vertexColors: true,
      side: THREE.FrontSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    // Attach hillshade grid for use in color updates
    if (hillshadeGrid) {
      (mesh as any)._hillshadeGrid = hillshadeGrid;
    }
    return mesh;
  }

  function createTemperatureGlobe(climateData: number[][][]) {
    const latCount = climateData[0].length;
    const lonCount = climateData[0][0].length;
    const mesh = createGlobeMesh(latCount, lonCount, 1);
    return mesh;
  }

  let bmShaderMaterial: THREE.ShaderMaterial | null = null;

  function createOceanSpecularMaterial(): THREE.ShaderMaterial {
    bmShaderMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float isOcean;
        varying vec3 vColor;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying float vIsOcean;
        void main() {
          vColor = color;
          // Transform normal to world space (not view space)
          vNormal = mat3(modelMatrix) * normal;
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vWorldPos = worldPos.xyz;
          vIsOcean = isOcean;
          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: `
        uniform vec3 sunDirection;
        uniform float ambientIntensity;
        varying vec3 vColor;
        varying vec3 vNormal;
        varying vec3 vWorldPos;
        varying float vIsOcean;

        void main() {
          vec3 normal = normalize(vNormal);
          float rawNdotL = dot(normal, sunDirection);
          // Soften terminator for ocean — wrap lighting slightly into shadow
          float NdotL = vIsOcean > 0.01
            ? smoothstep(-0.15, 0.3, rawNdotL)
            : max(rawNdotL, 0.0);
          vec3 diffuse = vColor * (ambientIntensity + NdotL * (1.0 - ambientIntensity));

          vec3 viewDir = normalize(cameraPosition - vWorldPos);

          // Fresnel: view-angle effect, applies day and night
          float fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
          fresnel = fresnel * fresnel; // quadratic
          float fresnelStrength = fresnel * 0.4 * vIsOcean;
          vec3 fresnelColor = vec3(0.5, 0.65, 0.85) * fresnelStrength;

          // Specular: fade smoothly across terminator
          vec3 specColor = vec3(0.0);
          if (vIsOcean > 0.01) {
            vec3 halfVec = normalize(sunDirection + viewDir);
            float specSharp = pow(max(dot(normal, halfVec), 0.0), 60.0) * 0.6;
            float specBroad = pow(max(dot(normal, halfVec), 0.0), 8.0) * 0.15;
            float specFade = smoothstep(-0.05, 0.15, rawNdotL);
            specColor = vec3(1.0, 0.97, 0.90) * (specSharp + specBroad) * vIsOcean * specFade;
          }

          gl_FragColor = vec4(diffuse + specColor + fresnelColor, 1.0);
        }
      `,
      uniforms: {
        sunDirection: { value: new THREE.Vector3(1, 0, 0) },
        ambientIntensity: { value: 0.15 },
      },
      vertexColors: true,
      side: THREE.FrontSide,
    });
    return bmShaderMaterial;
  }

  function createBlueMarbleGlobe(ld: ClimateLayerData) {
    // Use land_mask resolution (0.25deg, same as temperature) for crisp coastlines
    const bmNlat = ld.land_mask.shape[0];
    const bmNlon = ld.land_mask.shape[1];
    const elevData = ld.elevation?.data as Float32Array | undefined;
    const elevNlat = ld.elevation?.shape[0];
    const elevNlon = ld.elevation?.shape[1];
    const landMaskData = ld.land_mask.data as Uint8Array;
    const mesh = createGlobeMesh(bmNlat, bmNlon, 1, elevData, elevNlat, elevNlon, landMaskData, bmNlat, bmNlon);
    // Replace default Lambert material with ocean specular shader
    mesh.material = createOceanSpecularMaterial();
    return mesh;
  }

  // Update sun position - either relative to camera (when auto-rotating) or orbiting the globe (when static)
  function updateSunPosition(isAutoRotating: boolean) {
    if (!sunLight || !camera) return;

    const distance = 50;

    if (uniformLighting) {
      // "Always Day": place sun directly behind camera so visible side is always lit
      const cameraDir = new THREE.Vector3();
      camera.getWorldDirection(cameraDir);
      const dir = cameraDir.negate();
      const normalizedDir = dir.clone().normalize();
      if (treeInstances) treeInstances.setSunDirection(normalizedDir);
      if (cloudInstances) cloudInstances.setSunDirection(normalizedDir);
      if (cityLights) cityLights.setSunDirection(normalizedDir);
      if (atmosphereMesh) updateAtmosphereSunDirection(atmosphereMesh, normalizedDir);
      if (bmShaderMaterial) bmShaderMaterial.uniforms.sunDirection.value.copy(normalizedDir);
      sunLight.position.copy(normalizedDir).multiplyScalar(distance);
      if (sunOrb) sunOrb.position.copy(normalizedDir).multiplyScalar(360);
      return;
    }

    const declination = getSunDeclination(displayMonthProgress);

    let horizontalDir: THREE.Vector3;

    if (isAutoRotating) {
      // Sun stays fixed relative to camera view (illuminates from viewer's left-front)
      const cameraDir = new THREE.Vector3();
      camera.getWorldDirection(cameraDir);

      // Get "left" direction relative to camera view (cross with world up)
      const worldUp = new THREE.Vector3(0, 1, 0);
      const leftDir = new THREE.Vector3().crossVectors(worldUp, cameraDir).normalize();

      // Blend left direction away from camera direction to push sun more behind camera
      // This makes ~60% of visible globe lit instead of 50%
      const backwardBias = 0.4;
      horizontalDir = new THREE.Vector3(
        leftDir.x - cameraDir.x * backwardBias,
        0,
        leftDir.z - cameraDir.z * backwardBias
      ).normalize();
    } else {
      // Sun orbits around the globe in world space
      horizontalDir = new THREE.Vector3(
        Math.cos(sunOrbitAngle),
        0,
        Math.sin(sunOrbitAngle)
      );
    }

    // Apply declination tilt
    const sunDir = new THREE.Vector3(
      horizontalDir.x * Math.cos(declination),
      Math.sin(declination),
      horizontalDir.z * Math.cos(declination)
    ).normalize();

    if (treeInstances) treeInstances.setSunDirection(sunDir);
    if (cloudInstances) cloudInstances.setSunDirection(sunDir.clone());
    if (cityLights) cityLights.setSunDirection(sunDir);
    if (atmosphereMesh) updateAtmosphereSunDirection(atmosphereMesh, sunDir);

    // Update ocean specular shader sun direction (in world space, pre-scaling)
    if (bmShaderMaterial) bmShaderMaterial.uniforms.sunDirection.value.copy(sunDir);

    const sunDirNorm = sunDir.clone().normalize();
    sunLight.position.copy(sunDirNorm).multiplyScalar(distance);
    if (sunOrb) sunOrb.position.copy(sunDirNorm).multiplyScalar(360);
  }

  function initWindParticles() {
    if (!layerData?.wind_u_10m || !layerData?.wind_v_10m || !layerData?.wind_speed_10m) return;
    if (windParticles) {
      scene.remove(windParticles.getObject());
      windParticles.dispose();
    }
    windParticles = new WindParticles({
      wind_u_10m: layerData.wind_u_10m,
      wind_v_10m: layerData.wind_v_10m,
      wind_speed_10m: layerData.wind_speed_10m,
    });
    const obj = windParticles.getObject();
    obj.visible = activeLayer === 'blue-marble';
    // Sync rotation with globe
    if (globe) obj.rotation.y = globe.rotation.y;
    else if (blueMarbleGlobe) obj.rotation.y = blueMarbleGlobe.rotation.y;
    scene.add(obj);
  }

  function initCloudInstances() {
    if (!layerData?.cloud_convective && !layerData?.cloud_low && !layerData?.cloud_high) return;
    if (cloudInstances) {
      scene.remove(cloudInstances.getObject());
      cloudInstances.dispose();
    }
    cloudInstances = new CloudInstances(layerData!);
    if (container) cloudInstances.setViewportHeight(container.clientHeight * Math.min(window.devicePixelRatio, 2), camera.fov);
    const obj = cloudInstances.getObject();
    obj.visible = activeLayer === 'blue-marble';
    if (globe) obj.rotation.y = globe.rotation.y;
    else if (blueMarbleGlobe) obj.rotation.y = blueMarbleGlobe.rotation.y;
    scene.add(obj);
  }

  function initTreeInstances() {
    if (!layerData?.vegetation_fraction || !layerData?.soil_moisture || !layerData?.land_mask_native) return;
    if (treeInstances) {
      scene.remove(treeInstances.getObject());
      treeInstances.dispose();
    }
    treeInstances = new TreeInstances(layerData);
    const obj = treeInstances.getObject();
    obj.visible = activeLayer === 'blue-marble';
    if (globe) obj.rotation.y = globe.rotation.y;
    else if (blueMarbleGlobe) obj.rotation.y = blueMarbleGlobe.rotation.y;
    scene.add(obj);
  }

  function onMouseDown(e: MouseEvent) {
    mouseDownPos = { x: e.clientX, y: e.clientY };
  }

  function onMouseUp(e: MouseEvent) {
    const dx = e.clientX - mouseDownPos.x;
    const dy = e.clientY - mouseDownPos.y;
    if (dx * dx + dy * dy > 25) return; // drag, not click

    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(mouse, camera);

    // Intersect whichever globe is visible
    const target = activeLayer === 'blue-marble' ? blueMarbleGlobe : globe;
    if (!target) return;
    const hits = raycaster.intersectObject(target);
    if (hits.length === 0) {
      dismissMarker();
      return;
    }

    const hit = hits[0].point;
    // Transform to local mesh coords (undo globe rotation)
    const local = target.worldToLocal(hit.clone());
    const r = local.length();
    const lat = Math.asin(local.y / r) * (180 / Math.PI);
    const lon = Math.atan2(local.z, -local.x) * (180 / Math.PI);

    placeMarker(lat, lon, target);
    dispatch('pick', { lat, lon, screenX: e.clientX, screenY: e.clientY });
  }

  function placeMarker(lat: number, lon: number, parentMesh: THREE.Mesh) {
    if (!markerMesh) {
      const ring = new THREE.RingGeometry(0.012, 0.018, 32);
      const mat = new THREE.MeshBasicMaterial({ color: 0x2aabab, side: THREE.DoubleSide, depthTest: false, transparent: true, opacity: 0.9 });
      markerMesh = new THREE.Mesh(ring, mat);
      markerMesh.renderOrder = 999;
    }
    // Position on sphere surface
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = lon * (Math.PI / 180);
    const r = 1.008; // offset above terrain
    const x = -r * Math.sin(phi) * Math.cos(theta);
    const y = r * Math.cos(phi);
    const z = r * Math.sin(phi) * Math.sin(theta);
    markerMesh.position.set(x, y, z);
    // Orient along surface normal
    markerMesh.lookAt(0, 0, 0);

    if (!markerMesh.parent) {
      scene.add(markerMesh);
    }
    // Store unrotated local position for per-frame updates
    markerWorldPos = new THREE.Vector3(x, y, z);
  }

  export function dismissMarker() {
    if (markerMesh && markerMesh.parent) {
      markerMesh.parent.remove(markerMesh);
    }
    markerWorldPos = null;
    dispatch('pick', null);
  }

  function createPrimordialGlobe(landMask: { data: Uint8Array; nlat: number; nlon: number }): THREE.Mesh {
    const { data: mask, nlat: latCount, nlon: lonCount } = landMask;
    const positions: number[] = [];
    const colors: number[] = [];
    const radius = 1;
    const latStep = 180 / latCount;
    const lonStep = 360 / lonCount;

    for (let i = 0; i < latCount; i++) {
      const lat0 = 90 - i * latStep;
      const lat1 = 90 - (i + 1) * latStep;
      const dataLatIdx = latCount - 1 - i;

      for (let j = 0; j < lonCount; j++) {
        const lon0 = j * lonStep;
        const lon1 = (j + 1) * lonStep;
        const isLand = mask[dataLatIdx * lonCount + j] === 1;
        const [r, g, b] = isLand ? [0.003, 0.003, 0.0025] : [0.001, 0.001, 0.003];

        // Two triangles per cell
        for (const [la, lo] of [[lat0, lon0], [lat1, lon0], [lat1, lon1], [lat0, lon0], [lat1, lon1], [lat0, lon1]]) {
          const phi = (90 - la) * (Math.PI / 180);
          const theta = lo * (Math.PI / 180);
          positions.push(
            -radius * Math.sin(phi) * Math.cos(theta),
            radius * Math.cos(phi),
            radius * Math.sin(phi) * Math.sin(theta)
          );
          colors.push(r, g, b);
        }
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshBasicMaterial({ vertexColors: true });
    return new THREE.Mesh(geometry, material);
  }


  function createFlashQuad(aspect: number, sunScreenPos: THREE.Vector2): THREE.Mesh {
    const geo = new THREE.PlaneGeometry(2, 2);
    const mat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = position.xy * 0.5 + 0.5;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uIntensity;
        uniform float uRadius;
        uniform float uAspect;
        uniform vec2 uCenter;
        varying vec2 vUv;
        void main() {
          vec2 d = vUv - uCenter;
          d.x *= uAspect;
          float dist = length(d);
          // Solid core up to 60% of radius, then smooth falloff to edge
          float coreEnd = uRadius * 0.6;
          float bloom = dist < coreEnd ? 1.0 : smoothstep(uRadius, coreEnd, dist);
          float alpha = bloom * uIntensity;
          gl_FragColor = vec4(1.0, 0.98, 0.95, alpha);
        }
      `,
      uniforms: {
        uIntensity: { value: 0.0 },
        uRadius: { value: 0.1 },
        uAspect: { value: aspect },
        uCenter: { value: sunScreenPos },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = 9999;
    return mesh;
  }

  /** Compute where the sun would be on screen (UV 0-1) for the current view. */
  function getSunScreenPos(): THREE.Vector2 {
    if (!camera) return new THREE.Vector2(0.5, 0.5);

    // Compute sun direction the same way updateSunPosition does
    const declination = getSunDeclination(monthProgress);
    const isAutoRotating = controls?.autoRotate ?? true;

    let horizontalDir: THREE.Vector3;
    if (isAutoRotating) {
      const cameraDir = new THREE.Vector3();
      camera.getWorldDirection(cameraDir);
      const worldUp = new THREE.Vector3(0, 1, 0);
      const leftDir = new THREE.Vector3().crossVectors(worldUp, cameraDir).normalize();
      const backwardBias = 0.4;
      horizontalDir = new THREE.Vector3(
        leftDir.x - cameraDir.x * backwardBias, 0, leftDir.z - cameraDir.z * backwardBias
      ).normalize();
    } else {
      horizontalDir = new THREE.Vector3(Math.cos(sunOrbitAngle), 0, Math.sin(sunOrbitAngle));
    }

    const sunDir = new THREE.Vector3(
      horizontalDir.x * Math.cos(declination),
      Math.sin(declination),
      horizontalDir.z * Math.cos(declination)
    ).normalize();

    // Project a point along sun direction (use a moderate distance, not 360)
    const sunWorldPos = sunDir.clone().multiplyScalar(10);
    const projected = sunWorldPos.clone().project(camera);
    // Clamp to screen bounds — sun may be behind/off screen
    return new THREE.Vector2(
      Math.max(0, Math.min(1, projected.x * 0.5 + 0.5)),
      Math.max(0, Math.min(1, projected.y * 0.5 + 0.5))
    );
  }

  /** Start the bloom-in effect centered on the sun. Stays at peak until startFlashFade(). */
  export function triggerFlash() {
    if (!scene) return;
    const aspect = camera ? camera.aspect : 1;
    const sunPos = getSunScreenPos();
    flashQuad = createFlashQuad(aspect, sunPos);
    scene.add(flashQuad);
    flashStartTime = performance.now();
    flashFadeStart = null;
  }

  /** Begin fading the flash from current intensity to zero. */
  export function startFlashFade() {
    flashFadeStart = performance.now();
  }

  // When revealed and data ready, swap primordial → full scene
  // App.svelte handles sequencing: triggerFlash → rAF → revealed=true → rAF → startFlashFade
  let revealScheduled = false;
  $: if (revealed && scene && data && layerData && !revealScheduled) {
    revealScheduled = true;

    // Remove primordial globe
    if (primordialGlobe) {
      scene.remove(primordialGlobe);
      primordialGlobe.geometry.dispose();
      (primordialGlobe.material as THREE.Material).dispose();
      primordialGlobe = null;
    }

    // Create full globes
    if (!globe && data) {
      globe = createTemperatureGlobe(data);
      globe.visible = activeLayer === 'temperature';
      scene.add(globe);
      applyTemperatureColors(progressToStep(displayMonthProgress));
    }
    if (!blueMarbleGlobe && layerData) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      applyBlueMarbleColors(progressToStep(displayMonthProgress));
      initWindParticles();
      initTreeInstances();
      initCloudInstances();
    }

    // Show sun
    if (sunOrb) sunOrb.visible = true;
    if (sunGlow) sunGlow.visible = true;
    sunLight.intensity = 2.0;
    ambientLight.intensity = 0.15;

    // Create atmosphere and city lights
    if (!atmosphereMesh) {
      atmosphereMesh = createAtmosphere();
      atmosphereMesh.visible = activeLayer === 'blue-marble';
      scene.add(atmosphereMesh);
    } else {
      atmosphereMesh.visible = activeLayer === 'blue-marble';
    }
    if (!cityLights) {
      cityLights = new CityLights();
      const cityObj = cityLights.getObject();
      cityObj.visible = activeLayer === 'blue-marble';
      scene.add(cityObj);
    } else {
      cityLights.getObject().visible = activeLayer === 'blue-marble';
    }
  }

  // Show primordial globe when land mask arrives (before reveal)
  $: if (primordialLandMask && scene && !revealed && !primordialGlobe) {
    primordialGlobe = createPrimordialGlobe(primordialLandMask);
    scene.add(primordialGlobe);
  }

  function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Dim ambient light so night side is slightly visible
    ambientLight = new THREE.AmbientLight(0xffffff, revealed ? 0.15 : 0.0);
    scene.add(ambientLight);

    // Directional light as sun - position updated each frame relative to camera
    sunLight = new THREE.DirectionalLight(0xffffff, revealed ? 2.0 : 0.0);
    scene.add(sunLight);

    // Visible sun orb with limb darkening
    sunOrb = createSunOrb();
    sunOrb.visible = revealed;
    scene.add(sunOrb);

    // Screen-space sun bloom
    sunGlow = createSunBloom();
    (sunGlow as THREE.Mesh).visible = revealed;
    scene.add(sunGlow);

    // Initialize display month progress
    displayMonthProgress = monthProgress;

    // Camera - positioned slightly above equator for natural globe view
    camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    // Position camera at ~25° above equator
    const cameraDistance = 3;
    const tiltAngle = 20 * (Math.PI / 180); // 20 degrees
    camera.position.set(0, cameraDistance * Math.sin(tiltAngle), cameraDistance * Math.cos(tiltAngle));
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1.1;
    controls.maxDistance = 20;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;

    // Stop auto-rotation on user interaction
    controls.addEventListener('start', () => {
      controls.autoRotate = false;
      dispatch('interact');
    });

    // Create primordial globe if land mask is available and not yet revealed
    if (primordialLandMask && !revealed) {
      primordialGlobe = createPrimordialGlobe(primordialLandMask);
      scene.add(primordialGlobe);
    }

    // Create full globes only if already revealed
    if (revealed && data) {
      globe = createTemperatureGlobe(data);
      globe.visible = activeLayer === 'temperature';
      scene.add(globe);
      applyTemperatureColors(progressToStep(displayMonthProgress));
    }

    if (revealed && layerData) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      applyBlueMarbleColors(progressToStep(displayMonthProgress));
      initWindParticles();
      initTreeInstances();
      initCloudInstances();
    }

    // City lights and atmosphere — only create after reveal
    if (revealed) {
      cityLights = new CityLights();
      const cityObj = cityLights.getObject();
      cityObj.visible = activeLayer === 'blue-marble';
      scene.add(cityObj);

      atmosphereMesh = createAtmosphere();
      atmosphereMesh.visible = activeLayer === 'blue-marble';
      scene.add(atmosphereMesh);
    }

    // Load star field
    createStarField().then(sf => {
      starField = sf;
      scene.add(sf.group);     // star points (rotated by group)
      scene.add(sf.milkyWay);  // milky way (rotated via shader uniform)
    }).catch(e => console.error('Failed to load stars:', e));

    // Load borders
    if (showBorders) {
      const elev = layerData?.elevation;
      loadBorders(
        elev?.data as Float32Array | undefined,
        elev?.shape[0],
        elev?.shape[1],
      ).then(group => {
        bordersGroup = group;
        scene.add(bordersGroup);
      }).catch(e => console.error('Failed to load borders:', e));
    }

    // Handle resize
    window.addEventListener('resize', onResize);

    // Click-to-inspect: raycasting
    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mouseup', onMouseUp);

    // Animation loop
    animate();
  }

  function animate(time?: number) {
    animationId = requestAnimationFrame(animate);
    controls.update();

    const isAutoRotating = controls.autoRotate;

    // Snap to target month progress (no easing)
    displayMonthProgress = monthProgress;

    // Update colors only when nearest sub-step changes
    const step = progressToStep(displayMonthProgress);
    if (step !== lastAppliedStep) {
      if (activeLayer === 'temperature') {
        applyTemperatureColors(step);
      } else if (activeLayer === 'blue-marble') {
        applyBlueMarbleColors(step);
      }
      lastAppliedStep = step;
    }

    // When not auto-rotating, orbit the sun around the globe at 4x auto-rotate speed
    // This gives 1 day per ~60 seconds
    if (!isAutoRotating && lastAnimateTime !== null && time !== undefined) {
      const dt = (time - lastAnimateTime) / 1000;
      sunOrbitAngle += dt * controls.autoRotateSpeed * (2 * Math.PI / 60) * 2;
    }

    // Update wind particles
    if (windParticles && time !== undefined && lastAnimateTime !== null) {
      const dt = (time - lastAnimateTime) / 1000;
      windParticles.setMonth(displayMonthProgress);
      if (activeLayer === 'blue-marble') {
        windParticles.update(dt);
      }
    }

    // Update seasonal tree foliage
    if (treeInstances && layerData && activeLayer === 'blue-marble') {
      treeInstances.setMonth(displayMonthProgress, layerData);
    }

    // Update clouds: setMonth for target opacity, update for drift + fade animation
    if (cloudInstances && layerData && activeLayer === 'blue-marble') {
      cloudInstances.setMonth(displayMonthProgress, layerData);
      if (time !== undefined && lastAnimateTime !== null) {
        const cdt = (time - lastAnimateTime) / 1000;
        cloudInstances.update(cdt);
      }
    }

    // Update sun position (only after reveal)
    if (revealed) updateSunPosition(isAutoRotating);

    // Rotate stars to match sun's RA with its world-space orbit angle.
    // Use sunOrbitAngle (not the camera-relative sun position) so stars
    // stay fixed in world space during auto-rotate — they drift past
    // naturally as the camera orbits, just like the real night sky.
    if (starField) {
      updateStarRotation(starField, sunOrbitAngle, displayMonthProgress);
    }

    // Update screen-space sun bloom (only after reveal)
    if (revealed && sunGlow && camera && sunOrb) {
      updateSunBloom(sunGlow as THREE.Mesh, sunOrb, camera);
    }

    // Sync marker with globe rotation and project to screen
    if (markerMesh && markerWorldPos) {
      const refMesh = activeLayer === 'blue-marble' ? blueMarbleGlobe : globe;
      const rotY = refMesh?.rotation.y ?? 0;

      // Apply globe rotation to get world position
      const worldPos = markerWorldPos.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), rotY);
      markerMesh.position.copy(worldPos);
      markerMesh.lookAt(0, 0, 0);

      // Visibility: is marker facing camera?
      const toCamera = new THREE.Vector3().subVectors(camera.position, worldPos).normalize();
      const surfaceNormal = worldPos.clone().normalize();
      const visible = surfaceNormal.dot(toCamera) > 0;
      markerMesh.visible = visible;

      // Project to screen
      const projPos = worldPos.clone().project(camera);
      const rect = renderer.domElement.getBoundingClientRect();
      const sx = (projPos.x * 0.5 + 0.5) * rect.width + rect.left;
      const sy = (-projPos.y * 0.5 + 0.5) * rect.height + rect.top;
      dispatch('markerScreen', { x: sx, y: sy, visible });
    }

    // Flash bloom: bloom-in → hold → fade-out
    if (flashQuad && flashStartTime !== null) {
      const uniforms = (flashQuad.material as THREE.ShaderMaterial).uniforms;
      const now = performance.now();

      if (flashFadeStart !== null) {
        // Fade-out phase
        const elapsed = (now - flashFadeStart) / 1000;
        const t = Math.min(1, elapsed / FLASH_FADE_DURATION);
        // Ease out
        const fadeT = t * t;
        uniforms.uIntensity.value = 1 - fadeT;
        // Hold radius at full coverage during fade
        const aspect = uniforms.uAspect.value;
        const cx = uniforms.uCenter.value.x;
        const cy = uniforms.uCenter.value.y;
        let maxDist = 0;
        for (const [px, py] of [[0,0],[1,0],[0,1],[1,1]]) {
          const dx = (px - cx) * aspect;
          const dy = py - cy;
          maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy));
        }
        uniforms.uRadius.value = maxDist * 2.0 + 0.05;
        if (t >= 1) {
          scene.remove(flashQuad);
          flashQuad.geometry.dispose();
          (flashQuad.material as THREE.Material).dispose();
          flashQuad = null;
          flashStartTime = null;
          flashFadeStart = null;
        }
      } else {
        // Bloom-in phase: intensity and radius grow from sun position outward
        const elapsed = (now - flashStartTime) / 1000;
        const t = Math.min(1, elapsed / FLASH_BLOOM_DURATION);
        // Ease in-out for smooth bloom
        const bloomT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        uniforms.uIntensity.value = bloomT;
        // Radius must cover the farthest corner from the bloom center (in aspect-corrected space)
        const aspect = uniforms.uAspect.value;
        const cx = uniforms.uCenter.value.x;
        const cy = uniforms.uCenter.value.y;
        const corners = [
          [0, 0], [1, 0], [0, 1], [1, 1]
        ];
        let maxDist = 0;
        for (const [px, py] of corners) {
          const dx = (px - cx) * aspect;
          const dy = py - cy;
          maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy));
        }
        // Overshoot by 2x so smoothstep falloff is well beyond screen edges
        const maxRadius = maxDist * 2.0;
        uniforms.uRadius.value = 0.05 + bloomT * maxRadius;
      }
    }

    lastAnimateTime = time ?? null;

    renderer.render(scene, camera);
  }

  function onResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
    if (cloudInstances) cloudInstances.setViewportHeight(container.clientHeight * Math.min(window.devicePixelRatio, 2), camera.fov);
  }

  // Recreate temperature globe when data changes (only after reveal)
  $: if (scene && data && !globe && revealed) {
    globe = createTemperatureGlobe(data);
    globe.visible = activeLayer === 'temperature';
    scene.add(globe);
    applyTemperatureColors(progressToStep(displayMonthProgress));
  }

  // Create blue marble globe when layerData arrives (only after reveal)
  $: if (scene && layerData && !blueMarbleGlobe && revealed) {
    blueMarbleGlobe = createBlueMarbleGlobe(layerData);
    blueMarbleGlobe.visible = activeLayer === 'blue-marble';
    scene.add(blueMarbleGlobe);
    applyBlueMarbleColors(progressToStep(displayMonthProgress));
    initWindParticles();
    initTreeInstances();
  }

  onMount(() => {
    init();
  });

  onDestroy(() => {
    cancelAnimationFrame(animationId);
    if (cacheWorker) { cacheWorker.terminate(); cacheWorker = null; }
    window.removeEventListener('resize', onResize);
    renderer?.domElement.removeEventListener('mousedown', onMouseDown);
    renderer?.domElement.removeEventListener('mouseup', onMouseUp);
    renderer?.dispose();
    controls?.dispose();
    disposeSun(sunOrb, sunGlow);
    if (primordialGlobe) {
      primordialGlobe.geometry.dispose();
      (primordialGlobe.material as THREE.Material).dispose();
    }
    if (flashQuad) {
      flashQuad.geometry.dispose();
      (flashQuad.material as THREE.Material).dispose();
    }
    if (windParticles) {
      windParticles.dispose();
    }
    if (treeInstances) {
      treeInstances.dispose();
    }
    if (cloudInstances) {
      cloudInstances.dispose();
    }
    if (cityLights) {
      cityLights.dispose();
    }
    if (starField) {
      starField.dispose();
    }
    if (bordersGroup) {
      bordersGroup.traverse((obj) => {
        if (obj instanceof THREE.Line) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
    }
    if (globe) {
      globe.geometry.dispose();
      (globe.material as THREE.Material).dispose();
    }
    if (blueMarbleGlobe) {
      blueMarbleGlobe.geometry.dispose();
      (blueMarbleGlobe.material as THREE.Material).dispose();
    }
    if (atmosphereMesh) {
      atmosphereMesh.geometry.dispose();
      (atmosphereMesh.material as THREE.Material).dispose();
    }
    if (markerMesh) {
      markerMesh.geometry.dispose();
      (markerMesh.material as THREE.Material).dispose();
    }
  });
</script>

<div bind:this={container} class="globe-container"></div>

<style>
  .globe-container {
    width: 100%;
    height: 100%;
  }
</style>
