<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import { loadBorders } from './borders';
  import { WindParticles } from './WindParticles';
  import { TreeInstances } from './TreeInstances';
  import { CloudInstances } from './CloudInstances';
  import { createAtmosphere, updateAtmosphereSunDirection } from './Atmosphere';
  import { createStarField, updateStarRotation, type StarField } from './Stars';
  import { CityLights } from './CityLights';
  import { rendering } from './deviceScaling';
  import type { ClimateLayerData } from './loadBinaryData';
  import { getSunDeclination, createSunOrb, createSunBloom, updateSunBloom, disposeSun } from './Sun';
  import { createGlobeMesh, createTemperatureGlobe, createBlueMarbleGlobe, createPrimordialGlobe } from './globeFactory';
  import { buildTemperatureColorBuffer, buildBlueMarbleBuffers, buildPrecipitationColorBuffer, invalidateColorBuilderCaches } from './colorBufferBuilders';

  export let data: number[][][] | null = null; // [month][lat][lon] temperature in Celsius
  export let monthProgress: number = 0; // Continuous 0-12 (wraps), controls sun position
  export let showBorders: boolean = true;
  export let activeLayer: 'temperature' | 'precipitation' | 'blue-marble' = 'temperature';
  export let layerData: ClimateLayerData | null = null;
  export let uniformLighting: boolean = false;
  export let primordialLandMask: { data: Uint8Array; nlat: number; nlon: number } | null = null;
  export let revealed: boolean = false;
  export let stage: number = 5;

  const dispatch = createEventDispatcher();

  let container: HTMLDivElement;
  let containerResizeObserver: ResizeObserver | null = null;
  let renderer: THREE.WebGLRenderer;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let controls: OrbitControls;
  let globe: THREE.Mesh | null = null;
  let blueMarbleGlobe: THREE.Mesh | null = null;
  let precipGlobe: THREE.Mesh | null = null;
  let bordersGroup: THREE.Group | null = null; // parent of both variants
  let bordersFlat: THREE.Group | null = null;
  let bordersTerrain: THREE.Group | null = null;
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
  let precipBaseCache: (Float32Array | null)[] = new Array(12).fill(null);
  let precipStepCache: (Float32Array | null)[] = new Array(TOTAL_STEPS).fill(null);
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
    if (precipGlobe) {
      precipGlobe.rotation.y += radians;
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
    if (precipGlobe) {
      precipGlobe.rotation.y = 0;
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

  /** Rotate the globe so a lat/lon faces the camera, then place a marker. */
  export function flyTo(lat: number, lon: number, duration = 1500): void {
    if (!camera || !controls) return;

    controls.autoRotate = false;

    const startDist = camera.position.length();
    const mobile = container.clientWidth <= 640 || container.clientHeight <= 500;
    const targetDist = mobile ? 2.8 : 2.2;

    // --- Rotation: bring target longitude to face camera ---
    const camAngle = Math.atan2(camera.position.x, camera.position.z);
    const currentRotY = globe?.rotation.y ?? 0;
    const theta = lon * (Math.PI / 180);
    const pointAzimuth = Math.atan2(-Math.cos(theta), Math.sin(theta));
    // Offset so target lands at horizontal center of space left of inspect panel.
    // Panel: min(700, 0.9*W) on right. Desired position: (W-P)/2 from left.
    // NDC shift = P/W. Convert to rotation angle on unit sphere at distance d:
    //   α ≈ ndcShift * (d - 1) * tan(hFov/2)
    const W = container.clientWidth;
    const panelW = mobile ? 0 : Math.min(700, W * 0.9);
    const ndcShift = panelW / W;
    const aspect = W / container.clientHeight;
    const hFovHalf = Math.atan(Math.tan((45 / 2) * Math.PI / 180) * aspect);
    const offsetRad = -(ndcShift * (targetDist - 1) * Math.tan(hFovHalf));
    const targetRotY = camAngle - pointAzimuth + offsetRad;
    let deltaRot = targetRotY - currentRotY;
    while (deltaRot > Math.PI) deltaRot -= 2 * Math.PI;
    while (deltaRot < -Math.PI) deltaRot += 2 * Math.PI;
    const startRotY = currentRotY;
    const startPolar = Math.acos(camera.position.y / startDist);
    // Tilt camera to target latitude, with vertical offset on mobile
    // to keep the point above the bottom sheet (65vh panel).
    let targetPolar = (90 - lat) * (Math.PI / 180);
    if (mobile) {
      const vFovHalf = (45 / 2) * Math.PI / 180;
      // Panel covers 65vh → available 35vh → center at 17.5vh from top
      // Shift up from screen center: 0.325 of viewport height → NDC = 0.65
      const vertNdcShift = 0.65;
      targetPolar += vertNdcShift * (targetDist - 1) * Math.tan(vFovHalf);
    }
    const fixedAzimuth = Math.atan2(camera.position.x, camera.position.z);
    const startTime = performance.now();

    function animate() {
      const t = Math.min((performance.now() - startTime) / duration, 1);
      const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

      // Rotate globe
      const rot = startRotY + deltaRot * ease;
      if (globe) globe.rotation.y = rot;
      if (blueMarbleGlobe) blueMarbleGlobe.rotation.y = rot;
      if (precipGlobe) precipGlobe.rotation.y = rot;
      if (bordersGroup) bordersGroup.rotation.y = rot;
      if (windParticles) windParticles.getObject().rotation.y = rot;
      if (treeInstances) treeInstances.getObject().rotation.y = rot;
      if (cloudInstances) cloudInstances.getObject().rotation.y = rot;
      if (cityLights) cityLights.getObject().rotation.y = rot;

      // Zoom + tilt camera (fixed azimuth)
      const dist = startDist + (targetDist - startDist) * ease;
      const polar = startPolar + (targetPolar - startPolar) * ease;
      camera.position.set(
        dist * Math.sin(polar) * Math.sin(fixedAzimuth),
        dist * Math.cos(polar),
        dist * Math.sin(polar) * Math.cos(fixedAzimuth),
      );
      controls.update();

      if (t < 1) requestAnimationFrame(animate);
    }
    animate();

    // Place marker
    const target = activeLayer === 'blue-marble' ? blueMarbleGlobe : activeLayer === 'precipitation' ? precipGlobe : globe;
    if (target) {
      placeMarker(lat, lon, target);
      dispatch('pick', { lat, lon, screenX: 0, screenY: 0 });
    }
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
      surfaceData: ld.surface?.data as Float32Array | undefined,
      surfaceShape: ld.surface?.shape,
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
        const hasBm = bmBaseRgbCache.some(b => b !== null);
        for (let step = 0; step < TOTAL_STEPS; step++) {
          if (step % SUB_STEPS !== 0) {
            ensureTempStep(step);
            if (hasBm) ensureBmStep(step);
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
    invalidateColorBuilderCaches();
    bmBaseRgbCache = new Array(12).fill(null);
    bmBaseSpecCache = new Array(12).fill(null);
    bmStepRgbCache = new Array(TOTAL_STEPS).fill(null);
    bmStepSpecCache = new Array(TOTAL_STEPS).fill(null);
    precipBaseCache = new Array(12).fill(null);
    precipStepCache = new Array(TOTAL_STEPS).fill(null);
    lastAppliedStep = -1;
  }
  // Launch cache worker as soon as both data sources exist (even before reveal)
  $: if (data && layerData) {
    launchCacheWorker();
  }

  let ambientLight: THREE.AmbientLight;

  // Toggle visibility when activeLayer changes
  // Reference activeLayer before the if so Svelte tracks it as a dependency.
  // bordersGroup is referenced so the block re-runs after async border load.
  $: {
    const layer = activeLayer;
    void bordersGroup;
    if (scene) {
      if (globe) globe.visible = layer === 'temperature';
      if (precipGlobe) precipGlobe.visible = layer === 'precipitation';
      if (blueMarbleGlobe) blueMarbleGlobe.visible = layer === 'blue-marble';
      if (windParticles) windParticles.getObject().visible = layer === 'blue-marble';
      if (treeInstances) treeInstances.getObject().visible = layer === 'blue-marble';
      if (cloudInstances) cloudInstances.getObject().visible = layer === 'blue-marble';
      if (cityLights) cityLights.getObject().visible = layer === 'blue-marble';
      if (atmosphereMesh) atmosphereMesh.visible = layer === 'blue-marble';
      if (bordersFlat) bordersFlat.visible = layer !== 'blue-marble';
      if (bordersTerrain) bordersTerrain.visible = layer === 'blue-marble';
    }
  }

  // Build the two borders variants and add them to the scene. Called from
  // init() and re-called whenever layerData changes (which happens after
  // each stage transition once disposeStageMeshes() has cleared the old
  // bordersGroup).
  let bordersLoading = false;
  function loadBordersIntoScene(): void {
    if (!showBorders || !scene || bordersGroup || bordersLoading) return;
    bordersLoading = true;
    const elev = layerData?.elevation;
    loadBorders(
      elev?.data as Float32Array | undefined,
      elev?.shape[0],
      elev?.shape[1],
    ).then(({ flat, terrain }) => {
      bordersLoading = false;
      if (!scene) return;
      const group = new THREE.Group();
      group.add(flat);
      group.add(terrain);
      flat.visible = activeLayer !== 'blue-marble';
      terrain.visible = activeLayer === 'blue-marble';
      scene.add(group);
      bordersFlat = flat;
      bordersTerrain = terrain;
      bordersGroup = group;
    }).catch(e => {
      bordersLoading = false;
      console.error('Failed to load borders:', e);
    });
  }

  // Reactive trigger: whenever layerData changes (i.e. after a stage
  // transition) and borders aren't currently loaded, build them.
  $: if (layerData && !bordersGroup && !bordersLoading && scene) {
    loadBordersIntoScene();
  }

  function ensurePrecipBase(month: number) {
    if (precipBaseCache[month] || !layerData?.precipitation) return;
    precipBaseCache[month] = buildPrecipitationColorBuffer(layerData, month);
  }

  function ensurePrecipStep(step: number) {
    if (precipStepCache[step]) return;
    const m0 = Math.floor(step / SUB_STEPS) % 12;
    const t = (step % SUB_STEPS) / SUB_STEPS;
    const nearest = t < 0.5 ? m0 : (m0 + 1) % 12;
    ensurePrecipBase(nearest);
    precipStepCache[step] = precipBaseCache[nearest]!;
  }

  function applyPrecipitationColors(step: number) {
    if (!precipGlobe || !layerData?.precipitation) return;
    ensurePrecipStep(step);
    const colors = precipGlobe.geometry.attributes.color;
    ((colors as any).array as Float32Array).set(precipStepCache[step]!);
    colors.needsUpdate = true;
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

  let bmShaderMaterial: THREE.ShaderMaterial | null = null;

  // Update sun position - either relative to camera (when auto-rotating) or orbiting the globe (when static)
  function updateSunPosition(isAutoRotating: boolean) {
    if (!sunLight || !camera) return;

    const distance = 50;

    if (uniformLighting) {
      // "Always Day": place sun directly behind camera so visible side is always lit
      camera.getWorldDirection(_sunCameraDir);
      _sunDir.copy(_sunCameraDir).negate().normalize();
      if (treeInstances) treeInstances.setSunDirection(_sunDir);
      if (cloudInstances) cloudInstances.setSunDirection(_sunDir);
      if (cityLights) cityLights.setSunDirection(_sunDir);
      if (atmosphereMesh) updateAtmosphereSunDirection(atmosphereMesh, _sunDir);
      if (bmShaderMaterial) bmShaderMaterial.uniforms.sunDirection.value.copy(_sunDir);
      sunLight.position.copy(_sunDir).multiplyScalar(distance);
      if (sunOrb) sunOrb.position.copy(_sunDir).multiplyScalar(360);
      return;
    }

    const declination = getSunDeclination(displayMonthProgress);

    if (isAutoRotating) {
      // Sun stays fixed relative to camera view (illuminates from viewer's left-front)
      camera.getWorldDirection(_sunCameraDir);

      // Get "left" direction relative to camera view (cross with world up)
      _sunLeftDir.crossVectors(_worldUp, _sunCameraDir).normalize();

      // Blend left direction away from camera direction to push sun more behind camera
      // This makes ~60% of visible globe lit instead of 50%
      const backwardBias = 0.4;
      _sunHorizontalDir.set(
        _sunLeftDir.x - _sunCameraDir.x * backwardBias,
        0,
        _sunLeftDir.z - _sunCameraDir.z * backwardBias,
      ).normalize();
    } else {
      // Sun orbits around the globe in world space
      _sunHorizontalDir.set(
        Math.cos(sunOrbitAngle),
        0,
        Math.sin(sunOrbitAngle),
      );
    }

    // Apply declination tilt
    const cosDecl = Math.cos(declination);
    _sunDir.set(
      _sunHorizontalDir.x * cosDecl,
      Math.sin(declination),
      _sunHorizontalDir.z * cosDecl,
    ).normalize();

    if (treeInstances) treeInstances.setSunDirection(_sunDir);
    if (cloudInstances) cloudInstances.setSunDirection(_sunDir);
    if (cityLights) cityLights.setSunDirection(_sunDir);
    if (atmosphereMesh) updateAtmosphereSunDirection(atmosphereMesh, _sunDir);

    // Update ocean specular shader sun direction (in world space, pre-scaling)
    if (bmShaderMaterial) bmShaderMaterial.uniforms.sunDirection.value.copy(_sunDir);

    sunLight.position.copy(_sunDir).multiplyScalar(distance);
    if (sunOrb) sunOrb.position.copy(_sunDir).multiplyScalar(360);
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

  // Reused per-click and per-frame to avoid GC churn
  const _pickSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1);
  const _pickHit = new THREE.Vector3();
  const _axisY = new THREE.Vector3(0, 1, 0);
  const _worldUp = new THREE.Vector3(0, 1, 0);
  const _markerWorldPos = new THREE.Vector3();
  const _markerToCam = new THREE.Vector3();
  const _markerNormal = new THREE.Vector3();
  const _markerProj = new THREE.Vector3();
  const _sunCameraDir = new THREE.Vector3();
  const _sunLeftDir = new THREE.Vector3();
  const _sunHorizontalDir = new THREE.Vector3();
  const _sunDir = new THREE.Vector3();

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

    const target = activeLayer === 'blue-marble' ? blueMarbleGlobe : activeLayer === 'precipitation' ? precipGlobe : globe;
    if (!target) return;

    // Analytic sphere intersection — O(1) instead of raycasting against the
    // multi-million-vertex mesh, which can stall Safari for hundreds of ms.
    // The globe is a unit sphere centered at origin (radius 1).
    const hit = raycaster.ray.intersectSphere(_pickSphere, _pickHit);
    if (!hit) {
      dismissMarker();
      return;
    }

    // Transform world hit into the rotated mesh's local frame
    const local = target.worldToLocal(_pickHit.clone());
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

  /** Dispose all globe meshes and visual layers while keeping scene/camera/controls/stars alive. */
  export function disposeStageMeshes() {
    if (globe) {
      scene.remove(globe);
      globe.geometry.dispose();
      (globe.material as THREE.Material).dispose();
      globe = null;
    }
    if (blueMarbleGlobe) {
      scene.remove(blueMarbleGlobe);
      blueMarbleGlobe.geometry.dispose();
      (blueMarbleGlobe.material as THREE.Material).dispose();
      blueMarbleGlobe = null;
      bmShaderMaterial = null;
    }
    if (precipGlobe) {
      scene.remove(precipGlobe);
      precipGlobe.geometry.dispose();
      (precipGlobe.material as THREE.Material).dispose();
      precipGlobe = null;
    }
    if (windParticles) {
      scene.remove(windParticles.getObject());
      windParticles.dispose();
      windParticles = null;
    }
    if (treeInstances) {
      scene.remove(treeInstances.getObject());
      treeInstances.dispose();
      treeInstances = null;
    }
    if (cloudInstances) {
      scene.remove(cloudInstances.getObject());
      cloudInstances.dispose();
      cloudInstances = null;
    }
    if (cityLights) {
      scene.remove(cityLights.getObject());
      cityLights.dispose();
      cityLights = null;
    }
    if (atmosphereMesh) {
      scene.remove(atmosphereMesh);
      atmosphereMesh.geometry.dispose();
      (atmosphereMesh.material as THREE.Material).dispose();
      atmosphereMesh = null;
    }
    if (bordersGroup) {
      scene.remove(bordersGroup);
      bordersGroup.traverse((obj) => {
        if (obj instanceof THREE.Line) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
      bordersGroup = null;
      bordersFlat = null;
      bordersTerrain = null;
    }
    // Reset caches
    tempBaseCache = new Array(12).fill(null);
    tempStepCache = new Array(TOTAL_STEPS).fill(null);
    bmBaseRgbCache = new Array(12).fill(null);
    bmBaseSpecCache = new Array(12).fill(null);
    bmStepRgbCache = new Array(TOTAL_STEPS).fill(null);
    bmStepSpecCache = new Array(TOTAL_STEPS).fill(null);
    precipBaseCache = new Array(12).fill(null);
    precipStepCache = new Array(TOTAL_STEPS).fill(null);
    invalidateColorBuilderCaches();
    lastAppliedStep = -1;
    revealScheduled = false;
    if (cacheWorker) { cacheWorker.terminate(); cacheWorker = null; }
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
  $: if (revealed && scene && data && !revealScheduled) {
    revealScheduled = true;

    // Remove primordial globe
    if (primordialGlobe) {
      scene.remove(primordialGlobe);
      primordialGlobe.geometry.dispose();
      (primordialGlobe.material as THREE.Material).dispose();
      primordialGlobe = null;
    }

    // Create temperature globe (always needed at stages 1-4)
    if (!globe && data) {
      globe = createTemperatureGlobe(data);
      globe.visible = activeLayer === 'temperature';
      scene.add(globe);
      applyTemperatureColors(progressToStep(displayMonthProgress));
    }

    // Create blue marble globe when surface data is available
    if (!blueMarbleGlobe && layerData?.surface) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      bmShaderMaterial = blueMarbleGlobe.material as THREE.ShaderMaterial;
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      applyBlueMarbleColors(progressToStep(displayMonthProgress));
    }

    // Wind particles when wind data exists
    if (layerData?.wind_u_10m) {
      initWindParticles();
    }

    // Trees and clouds when vegetation/cloud data exists
    if (layerData?.vegetation_fraction) {
      initTreeInstances();
    }
    if (layerData?.cloud_fraction) {
      initCloudInstances();
    }

    if (!precipGlobe && layerData?.precipitation) {
      const p = layerData.precipitation;
      precipGlobe = createGlobeMesh(p.shape[1], p.shape[2], 1);
      precipGlobe.visible = activeLayer === 'precipitation';
      scene.add(precipGlobe);
      applyPrecipitationColors(progressToStep(displayMonthProgress));
    }

    // Show sun
    if (sunOrb) sunOrb.visible = true;
    if (sunGlow) sunGlow.visible = true;
    sunLight.intensity = 2.0;
    ambientLight.intensity = 0.15;

    // Atmosphere glow from stage 2 (once there's air)
    if (stage >= 2) {
      if (!atmosphereMesh) {
        atmosphereMesh = createAtmosphere();
        atmosphereMesh.visible = activeLayer === 'blue-marble';
        scene.add(atmosphereMesh);
      } else {
        atmosphereMesh.visible = activeLayer === 'blue-marble';
      }
    }
    // City lights only at stage 5
    if (stage >= 5) {
      if (!cityLights) {
        const elev = layerData?.elevation;
        cityLights = new CityLights(
          elev?.data as Float32Array | undefined,
          elev?.shape[0],
          elev?.shape[1],
        );
        const cityObj = cityLights.getObject();
        cityObj.visible = activeLayer === 'blue-marble';
        scene.add(cityObj);
      } else {
        cityLights.getObject().visible = activeLayer === 'blue-marble';
      }
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
    const cameraDistance = container.clientWidth <= 480 ? 5.2 : container.clientWidth <= 600 ? 4.5 : container.clientWidth <= 640 ? 3.8 : 3;
    const tiltAngle = 20 * (Math.PI / 180); // 20 degrees
    camera.position.set(0, cameraDistance * Math.sin(tiltAngle), cameraDistance * Math.cos(tiltAngle));
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, rendering.pixelRatioCap));
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

    if (revealed && layerData?.surface) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      bmShaderMaterial = blueMarbleGlobe.material as THREE.ShaderMaterial;
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      applyBlueMarbleColors(progressToStep(displayMonthProgress));
      if (layerData.wind_u_10m) initWindParticles();
      if (layerData.vegetation_fraction) initTreeInstances();
      if (layerData.cloud_fraction) initCloudInstances();

      if (layerData.precipitation) {
        const p = layerData.precipitation;
        precipGlobe = createGlobeMesh(p.shape[1], p.shape[2], 1);
        precipGlobe.visible = activeLayer === 'precipitation';
        scene.add(precipGlobe);
        applyPrecipitationColors(progressToStep(displayMonthProgress));
      }
    }

    // Atmosphere glow from stage 2
    if (revealed && stage >= 2) {
      atmosphereMesh = createAtmosphere();
      atmosphereMesh.visible = activeLayer === 'blue-marble';
      scene.add(atmosphereMesh);
    }

    // City lights at stage 5
    if (revealed && stage >= 5) {
      const elev = layerData?.elevation;
      cityLights = new CityLights(
        elev?.data as Float32Array | undefined,
        elev?.shape[0],
        elev?.shape[1],
      );
      const cityObj = cityLights.getObject();
      cityObj.visible = activeLayer === 'blue-marble';
      scene.add(cityObj);
    }

    // Load star field
    createStarField().then(sf => {
      starField = sf;
      scene.add(sf.group);     // star points (rotated by group)
      scene.add(sf.milkyWay);  // milky way (rotated via shader uniform)
    }).catch(e => console.error('Failed to load stars:', e));

    // Load borders for the first time (also re-runs reactively after stage
    // transitions when bordersGroup is reset to null).
    loadBordersIntoScene();

    // Handle resize. window resize covers viewport changes, but layout shifts
    // (e.g. the control bar appearing) don't fire a window resize — so we
    // also observe the container itself.
    window.addEventListener('resize', onResize);
    if (typeof ResizeObserver !== 'undefined') {
      containerResizeObserver = new ResizeObserver(() => onResize());
      containerResizeObserver.observe(container);
    }

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
      } else if (activeLayer === 'precipitation') {
        applyPrecipitationColors(step);
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

    // Update seasonal tree foliage + wind sway
    if (treeInstances && layerData && activeLayer === 'blue-marble') {
      treeInstances.setMonth(displayMonthProgress, layerData);
      if (time !== undefined && lastAnimateTime !== null) {
        const tdt = (time - lastAnimateTime) / 1000;
        treeInstances.update(tdt);
      }
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
      if (uniformLighting && camera) {
        // Always-day: position stars as if it's noon wherever the camera looks.
        // Camera azimuth = the longitude the user is viewing = where the sun would be at noon.
        const camAzimuth = Math.atan2(camera.position.x, camera.position.z);
        updateStarRotation(starField, camAzimuth, displayMonthProgress);
      } else {
        updateStarRotation(starField, sunOrbitAngle, displayMonthProgress);
      }
    }

    // Update screen-space sun bloom (only after reveal)
    if (revealed && sunGlow && camera && sunOrb) {
      updateSunBloom(sunGlow as THREE.Mesh, sunOrb, camera);
    }

    // Sync marker with globe rotation and project to screen
    if (markerMesh && markerWorldPos) {
      const refMesh = activeLayer === 'blue-marble' ? blueMarbleGlobe : activeLayer === 'precipitation' ? precipGlobe : globe;
      const rotY = refMesh?.rotation.y ?? 0;

      // Apply globe rotation to get world position (in-place; no allocation)
      _markerWorldPos.copy(markerWorldPos).applyAxisAngle(_axisY, rotY);
      markerMesh.position.copy(_markerWorldPos);
      markerMesh.lookAt(0, 0, 0);

      // Visibility: is marker facing camera?
      _markerToCam.copy(camera.position).sub(_markerWorldPos).normalize();
      _markerNormal.copy(_markerWorldPos).normalize();
      const visible = _markerNormal.dot(_markerToCam) > 0;
      markerMesh.visible = visible;

      // Project to screen
      _markerProj.copy(_markerWorldPos).project(camera);
      const rect = renderer.domElement.getBoundingClientRect();
      const sx = (_markerProj.x * 0.5 + 0.5) * rect.width + rect.left;
      const sy = (-_markerProj.y * 0.5 + 0.5) * rect.height + rect.top;
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

  // Create blue marble globe when surface data arrives (only after reveal)
  $: if (scene && layerData?.surface && !blueMarbleGlobe && revealed) {
    blueMarbleGlobe = createBlueMarbleGlobe(layerData);
    bmShaderMaterial = blueMarbleGlobe.material as THREE.ShaderMaterial;
    blueMarbleGlobe.visible = activeLayer === 'blue-marble';
    scene.add(blueMarbleGlobe);
    applyBlueMarbleColors(progressToStep(displayMonthProgress));
    if (layerData.wind_u_10m) initWindParticles();
    if (layerData.vegetation_fraction) initTreeInstances();
    if (layerData.cloud_fraction) initCloudInstances();
  }

  // Create precipitation globe when precipitation data arrives (only after reveal)
  $: if (scene && layerData?.precipitation && !precipGlobe && revealed) {
    const p = layerData.precipitation!;
    precipGlobe = createGlobeMesh(p.shape[1], p.shape[2], 1);
    precipGlobe.visible = activeLayer === 'precipitation';
    scene.add(precipGlobe);
    applyPrecipitationColors(progressToStep(displayMonthProgress));
  }

  onMount(() => {
    init();
  });

  onDestroy(() => {
    cancelAnimationFrame(animationId);
    if (cacheWorker) { cacheWorker.terminate(); cacheWorker = null; }
    window.removeEventListener('resize', onResize);
    containerResizeObserver?.disconnect();
    containerResizeObserver = null;
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
    if (precipGlobe) {
      precipGlobe.geometry.dispose();
      (precipGlobe.material as THREE.Material).dispose();
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
