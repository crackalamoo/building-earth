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

  function smoothstep(edge0: number, edge1: number, x: number): number {
    const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
    return t * t * (3 - 2 * t);
  }

  export let data: number[][][] | null = null; // [month][lat][lon] temperature in Celsius
  export let monthProgress: number = 0; // Continuous 0-12 (wraps), controls sun position
  export let showBorders: boolean = true;
  export let activeLayer: 'temperature' | 'blue-marble' = 'temperature';
  export let layerData: ClimateLayerData | null = null;
  export let uniformLighting: boolean = false;

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

  // Precompute all cached steps in background (one step per frame to avoid blocking)

  // Build one cache item per idle callback, yielding to the browser between each
  function warmCache(gen: number) {
    if (gen !== warmupGeneration) return;
    const ric = typeof requestIdleCallback === 'function' ? requestIdleCallback : (cb: () => void) => setTimeout(cb, 50);
    // Phase 1: build 12 base months, Phase 2: build 24 sub-step lerps
    let baseMonth = 0;
    let step = 0;
    function next() {
      if (gen !== warmupGeneration) return;
      // Build base months first (expensive)
      if (baseMonth < 12) {
        if (data && !tempBaseCache[baseMonth]) tempBaseCache[baseMonth] = buildTemperatureColorBuffer(data, baseMonth);
        if (layerData && !bmBaseRgbCache[baseMonth]) {
          const r = buildBlueMarbleBuffers(layerData, baseMonth);
          bmBaseRgbCache[baseMonth] = r.rgb; bmBaseSpecCache[baseMonth] = r.spec;
        }
        baseMonth++;
        ric(next);
      } else if (step < TOTAL_STEPS) {
        // Sub-steps: skip whole-month steps (t=0), only build intermediate lerps
        if (step % SUB_STEPS !== 0) {
          if (data) ensureTempStep(step);
          if (layerData) ensureBmStep(step);
        }
        step++;
        ric(next);
      }
    }
    ric(next);
  }

  // Invalidate caches when data changes and start idle-time warmup
  $: if (data) {
    tempBaseCache = new Array(12).fill(null);
    tempStepCache = new Array(TOTAL_STEPS).fill(null);
    lastAppliedStep = -1;
    warmCache(++warmupGeneration);
  }
  $: if (layerData) {
    bmBaseRgbCache = new Array(12).fill(null);
    bmBaseSpecCache = new Array(12).fill(null);
    bmStepRgbCache = new Array(TOTAL_STEPS).fill(null);
    bmStepSpecCache = new Array(TOTAL_STEPS).fill(null);
    lastAppliedStep = -1;
    warmCache(++warmupGeneration);
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

  function buildBlueMarbleBuffers(ld: ClimateLayerData, monthIdx: number): { rgb: Float32Array; spec: Float32Array } {
    const surfaceData = ld.surface.data as Float32Array;
    const landMaskData = ld.land_mask.data as Uint8Array;
    const soilData = ld.soil_moisture?.data as Float32Array | undefined;
    const vegData = ld.vegetation_fraction?.data as Float32Array | undefined;
    const coarseLandMask = ld.land_mask_native?.data as Uint8Array | undefined;
    const elevData = ld.elevation?.data as Float32Array | undefined;
    const elevNlat = ld.elevation?.shape[0] ?? 0;
    const elevNlon = ld.elevation?.shape[1] ?? 0;

    // High-res grid dimensions (from land_mask, 0.25deg)
    const hiNlat = ld.land_mask.shape[0];
    const hiNlon = ld.land_mask.shape[1];

    // Low-res grid dimensions (from surface, 5deg)
    const lowNlat = ld.surface.shape[1];
    const lowNlon = ld.surface.shape[2];
    const monthOffset = monthIdx * lowNlat * lowNlon;

    // Precompute annual means per coarse cell
    const annualMeanTemp = new Float32Array(lowNlat * lowNlon);
    const annualMeanSoil = new Float32Array(lowNlat * lowNlon);
    for (let ci = 0; ci < lowNlat; ci++) {
      for (let cj = 0; cj < lowNlon; cj++) {
        let tsum = 0, ssum = 0;
        for (let m = 0; m < 12; m++) {
          const idx = m * lowNlat * lowNlon + ci * lowNlon + cj;
          tsum += surfaceData[idx];
          if (soilData) ssum += soilData[idx];
        }
        annualMeanTemp[ci * lowNlon + cj] = tsum / 12;
        annualMeanSoil[ci * lowNlon + cj] = soilData ? ssum / 12 : 0.5;
      }
    }

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
              soilData, lowNlat, lowNlon, monthOffset,
              dataLatIdx, j, hiNlat, hiNlon,
              isLand, coarseLandMask,
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
          annualMeanSoil, lowNlat, lowNlon, 0,
          dataLatIdx, j, hiNlat, hiNlon,
          isLand, coarseLandMask,
        );

        const [r, g, b] = blueMarbleColor(isLand, surfaceTemp, soilMoisture, elev, vegFrac, annMeanT, annMeanSoil);
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
    const hsGrid = (blueMarbleGlobe as any)?._hillshadeGrid as Float32Array | undefined;
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
    const m0 = Math.floor(step / SUB_STEPS) % 12;
    const m1 = (m0 + 1) % 12;
    ensureTempBase(m0); ensureTempBase(m1);
    const t = (step % SUB_STEPS) / SUB_STEPS;
    if (t < 0.001) { tempStepCache[step] = tempBaseCache[m0]!; return; }
    const b0 = tempBaseCache[m0]!, b1 = tempBaseCache[m1]!;
    const out = new Float32Array(b0.length);
    const s = 1 - t;
    for (let i = 0; i < out.length; i++) out[i] = b0[i] * s + b1[i] * t;
    tempStepCache[step] = out;
  }

  function ensureBmStep(step: number) {
    if (bmStepRgbCache[step]) return;
    const m0 = Math.floor(step / SUB_STEPS) % 12;
    const m1 = (m0 + 1) % 12;
    ensureBmBase(m0); ensureBmBase(m1);
    const t = (step % SUB_STEPS) / SUB_STEPS;
    if (t < 0.001) {
      bmStepRgbCache[step] = bmBaseRgbCache[m0]!;
      bmStepSpecCache[step] = bmBaseSpecCache[m0]!;
      return;
    }
    const r0 = bmBaseRgbCache[m0]!, r1 = bmBaseRgbCache[m1]!;
    const s0 = bmBaseSpecCache[m0]!, s1 = bmBaseSpecCache[m1]!;
    const s = 1 - t;
    const outR = new Float32Array(r0.length);
    const outS = new Float32Array(s0.length);
    for (let i = 0; i < outR.length; i++) outR[i] = r0[i] * s + r1[i] * t;
    for (let i = 0; i < outS.length; i++) outS[i] = s0[i] * s + s1[i] * t;
    bmStepRgbCache[step] = outR;
    bmStepSpecCache[step] = outS;
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

  // Calculate sun declination based on continuous month progress
  function getSunDeclination(monthValue: number): number {
    // Declination: +23.5° in June (month 5), -23.5° in December (month 11)
    // Using sin wave: peak at month 5 (June), trough at month 11 (December)
    const declinationDeg = 23.5 * Math.sin((monthValue - 2) / 12 * 2 * Math.PI);
    return declinationDeg * (Math.PI / 180);
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

  function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Dim ambient light so night side is slightly visible
    ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
    scene.add(ambientLight);

    // Directional light as sun - position updated each frame relative to camera
    sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
    scene.add(sunLight);

    // Visible sun orb with limb darkening — placed far from globe to minimize parallax
    const sunOrbGeo = new THREE.SphereGeometry(7.2, 32, 32);
    const sunOrbMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vViewDir = normalize(-mvPos.xyz);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vViewDir;
        void main() {
          float mu = dot(vNormal, vViewDir);
          float limb = pow(max(mu, 0.0), 0.4);
          vec3 core = vec3(1.0, 0.95, 0.85);
          vec3 edge = vec3(1.0, 0.5, 0.1);
          vec3 col = mix(edge, core, limb);
          gl_FragColor = vec4(col, 1.0);
        }
      `,
      side: THREE.FrontSide,
      transparent: true,
      depthWrite: false,
    });
    sunOrb = new THREE.Mesh(sunOrbGeo, sunOrbMat);
    scene.add(sunOrb);

    // Screen-space sun bloom — fullscreen overlay that washes out when looking sunward
    const screenBloomMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec2 sunScreenPos;  // sun position in screen UV space (0-1)
        uniform float sunVisible;   // 0 = behind camera, 1 = in front
        uniform float sunIntensity; // how directly we're facing the sun
        uniform float aspectRatio;  // width / height
        uniform vec2 globeScreenPos;  // globe center in screen UV space
        uniform float globeScreenRadius; // globe's apparent radius in UV units (corrected for aspect)
        varying vec2 vUv;

        void main() {
          if (sunVisible < 0.01) discard;

          // Fade bloom behind globe disc
          vec2 toGlobe = vUv - globeScreenPos;
          toGlobe.x *= aspectRatio;
          float globeDist = length(toGlobe);
          float globeMask = smoothstep(globeScreenRadius - 0.003, globeScreenRadius + 0.003, globeDist);
          if (globeMask < 0.001) discard;

          // Distance from sun's screen position, aspect-corrected
          vec2 delta = vUv - sunScreenPos;
          delta.x *= aspectRatio;
          float d = length(delta);

          // Core bloom — intense near sun position
          float core = exp(-d * d * 120.0) * 2.5;

          // Mid glow — warm spread
          float mid = exp(-d * d * 15.0) * 0.6;

          // Wide wash — entire screen tint when facing sun
          float wash = exp(-d * d * 2.5) * 0.2;

          // Subtle anamorphic horizontal streak (lens artifact)
          float streak = exp(-abs(delta.y) * 40.0) * exp(-delta.x * delta.x * 3.0) * 0.12;

          float intensity = (core + mid + wash + streak) * sunIntensity * sunVisible * globeMask;

          // White-hot center → warm amber edge
          vec3 white = vec3(1.0, 1.0, 0.95);
          vec3 amber = vec3(1.0, 0.75, 0.35);
          vec3 col = mix(white, amber, smoothstep(0.0, 0.3, d));

          gl_FragColor = vec4(col * intensity, intensity);
        }
      `,
      uniforms: {
        sunScreenPos: { value: new THREE.Vector2(0.5, 0.5) },
        sunVisible: { value: 0.0 },
        sunIntensity: { value: 0.0 },
        aspectRatio: { value: 1.0 },
        globeScreenPos: { value: new THREE.Vector2(0.5, 0.5) },
        globeScreenRadius: { value: 0.3 },
      },
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    });
    const screenBloomGeo = new THREE.PlaneGeometry(2, 2);
    const screenBloomQuad = new THREE.Mesh(screenBloomGeo, screenBloomMat);
    screenBloomQuad.frustumCulled = false;
    screenBloomQuad.renderOrder = 999;
    scene.add(screenBloomQuad);
    sunGlow = screenBloomQuad;

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
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
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

    // Create temperature globe if data is available
    if (data) {
      globe = createTemperatureGlobe(data);
      globe.visible = activeLayer === 'temperature';
      scene.add(globe);
      applyTemperatureColors(progressToStep(displayMonthProgress));
    }

    // Create blue marble globe if layerData is available
    if (layerData) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      applyBlueMarbleColors(progressToStep(displayMonthProgress));
      initWindParticles();
      initTreeInstances();
      initCloudInstances();
    }

    // City lights (nightside glow)
    cityLights = new CityLights();
    const cityObj = cityLights.getObject();
    cityObj.visible = activeLayer === 'blue-marble';
    scene.add(cityObj);

    // Atmosphere glow
    atmosphereMesh = createAtmosphere();
    atmosphereMesh.visible = activeLayer === 'blue-marble';
    scene.add(atmosphereMesh);

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
        cloudInstances.update(cdt, camera);
      }
    }

    lastAnimateTime = time ?? null;

    // Update sun position
    updateSunPosition(isAutoRotating);

    // Rotate stars to match sun's RA with its world-space orbit angle.
    // Use sunOrbitAngle (not the camera-relative sun position) so stars
    // stay fixed in world space during auto-rotate — they drift past
    // naturally as the camera orbits, just like the real night sky.
    if (starField) {
      updateStarRotation(starField, sunOrbitAngle, displayMonthProgress);
    }

    // Update screen-space sun bloom
    if (sunGlow && camera && sunOrb) {
      const mat = (sunGlow as THREE.Mesh).material as THREE.ShaderMaterial;
      const sunWorldPos = sunOrb.position.clone();
      const sunNDC = sunWorldPos.project(camera);

      // Check if sun is in front of camera
      const camDir = new THREE.Vector3();
      camera.getWorldDirection(camDir);
      const toSun = sunOrb.position.clone().normalize();
      const facing = camDir.dot(toSun);

      // Ray-sphere occlusion: check if sun disc is blocked by globe
      // Sun orb radius 7.2 at distance 360 → angular radius in radians
      const sunAngularRadius = 7.2 / 360; // ~0.02 rad
      const globeRadius = 1.0;
      const camPos = camera.position;
      const camDist = camPos.length();
      // Globe's angular radius as seen from camera
      const globeAngularRadius = Math.asin(Math.min(1, globeRadius / camDist));
      // Angle between camera-to-sun direction and camera-to-globe-center
      const toSunDir = sunOrb.position.clone().sub(camPos).normalize();
      const toGlobeDir = camPos.clone().negate().normalize();
      const angleBetween = Math.acos(Math.min(1, Math.max(-1, toSunDir.dot(toGlobeDir))));
      // Sun edge clears globe when angle > globeAngularRadius + sunAngularRadius
      const clearAngle = globeAngularRadius + sunAngularRadius;
      const visibility = smoothstep(clearAngle - 0.04, clearAngle + 0.02, angleBetween);

      if (facing > -0.2 && visibility > 0.001) {
        mat.uniforms.sunVisible.value = smoothstep(-0.2, 0.1, facing) * visibility;
        mat.uniforms.sunScreenPos.value.set(
          sunNDC.x * 0.5 + 0.5,
          sunNDC.y * 0.5 + 0.5,
        );
        const intensity = Math.pow(Math.max(0, facing), 2.0);
        mat.uniforms.sunIntensity.value = intensity;
        mat.uniforms.aspectRatio.value = camera.aspect;

        // Globe screen-space disc for masking
        // Project globe center
        const globeCenter = new THREE.Vector3(0, 0, 0).clone().project(camera);
        const gcx = globeCenter.x * 0.5 + 0.5;
        const gcy = globeCenter.y * 0.5 + 0.5;
        mat.uniforms.globeScreenPos.value.set(gcx, gcy);
        // Visible limb angular radius (larger than asin(R/d) due to perspective)
        const cDist = camera.position.length();
        const R = 1.0;
        const limbAngularRadius = Math.asin(Math.min(1, R / cDist));
        // Convert to screen UV units: angular size → fraction of vertical FOV
        const vFov = camera.fov * Math.PI / 180;
        const screenRadiusY = Math.tan(limbAngularRadius) / Math.tan(vFov / 2);
        // In aspect-corrected UV space (where x is already multiplied by aspect)
        // screenRadiusY is in NDC half-extent; UV space spans 0–1, so halve it
        mat.uniforms.globeScreenRadius.value = screenRadiusY * 0.5;
      } else {
        mat.uniforms.sunVisible.value = 0.0;
      }
    }

    renderer.render(scene, camera);
  }

  function onResize() {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }

  // Recreate temperature globe when data changes
  $: if (scene && data && !globe) {
    globe = createTemperatureGlobe(data);
    globe.visible = activeLayer === 'temperature';
    scene.add(globe);
    applyTemperatureColors(progressToStep(displayMonthProgress));
  }

  // Create blue marble globe when layerData arrives
  $: if (scene && layerData && !blueMarbleGlobe) {
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
    window.removeEventListener('resize', onResize);
    renderer?.dispose();
    controls?.dispose();
    if (sunOrb) {
      sunOrb.geometry.dispose();
      (sunOrb.material as THREE.Material).dispose();
      if (sunGlow) {
        ((sunGlow as THREE.Mesh).geometry as THREE.BufferGeometry)?.dispose();
        ((sunGlow as THREE.Mesh).material as THREE.Material)?.dispose();
      }
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
  });
</script>

<div bind:this={container} class="globe-container"></div>

<style>
  .globe-container {
    width: 100%;
    height: 100%;
  }
</style>
