<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  import * as topojson from 'topojson-client';
  import type { Topology, GeometryCollection } from 'topojson-specification';
  import { temperatureToColorNormalized } from './colormap';
  import { blueMarbleColor } from './blueMarbleColormap';
  import { WindParticles } from './WindParticles';
  import type { ClimateLayerData } from './loadBinaryData';

  import { createEventDispatcher } from 'svelte';

  export let data: number[][][] | null = null; // [month][lat][lon] temperature in Celsius
  export let monthProgress: number = 0; // Continuous 0-12 (wraps), controls sun position
  export let showBorders: boolean = true;
  export let activeLayer: 'temperature' | 'blue-marble' = 'temperature';
  export let layerData: ClimateLayerData | null = null;

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

  // Update colors when displayMonth or data changes (temperature layer)
  $: if (globe && data && activeLayer === 'temperature') {
    updateTemperatureColors(data, displayMonth);
  }

  // Update blue marble colors when month or layerData changes
  $: if (blueMarbleGlobe && layerData && activeLayer === 'blue-marble') {
    updateBlueMarbleColors(layerData, displayMonth);
  }

  // Toggle visibility when activeLayer changes
  // Reference activeLayer before the if so Svelte tracks it as a dependency
  $: {
    const layer = activeLayer;
    if (scene) {
      if (globe) globe.visible = layer === 'temperature';
      if (blueMarbleGlobe) blueMarbleGlobe.visible = layer === 'blue-marble';
      if (windParticles) windParticles.getObject().visible = layer === 'blue-marble';
    }
  }

  function updateTemperatureColors(climateData: number[][][], monthIdx: number) {
    if (!globe) return;

    const geometry = globe.geometry;
    const colors = geometry.attributes.color;
    const monthData = averagePolarTemps(climateData[monthIdx], nlat, nlon);

    // Each cell has 6 vertices (2 triangles), all same color
    let idx = 0;
    for (let i = 0; i < nlat; i++) {
      const dataLatIdx = nlat - 1 - i;
      for (let j = 0; j < nlon; j++) {
        const dataLonIdx = j;
        const temp = monthData[dataLatIdx][dataLonIdx];
        const [r, g, b] = temperatureToColorNormalized(temp);

        // 6 vertices per cell
        for (let v = 0; v < 6; v++) {
          colors.setXYZ(idx, r, g, b);
          idx++;
        }
      }
    }
    colors.needsUpdate = true;
  }

  function updateBlueMarbleColors(ld: ClimateLayerData, monthIdx: number) {
    if (!blueMarbleGlobe) return;

    const geometry = blueMarbleGlobe.geometry;
    const colors = geometry.attributes.color;
    const surfaceData = ld.surface.data as Float32Array;
    const landMaskData = ld.land_mask.data as Uint8Array;
    const vegData = ld.vegetation_fraction?.data as Float32Array | undefined;

    const bmNlat = ld.surface.shape[1];
    const bmNlon = ld.surface.shape[2];
    const monthOffset = monthIdx * bmNlat * bmNlon;
    // vegetation_fraction may be monthly or absent
    const vegMonthOffset = vegData ? monthIdx * bmNlat * bmNlon : 0;

    let idx = 0;
    for (let i = 0; i < bmNlat; i++) {
      const dataLatIdx = bmNlat - 1 - i;
      for (let j = 0; j < bmNlon; j++) {
        const flatIdx = dataLatIdx * bmNlon + j;
        const surfaceTemp = surfaceData[monthOffset + flatIdx];
        const isLand = landMaskData[flatIdx] === 1;
        const vegFrac = vegData ? vegData[vegMonthOffset + flatIdx] : 0;

        const [r, g, b] = blueMarbleColor(isLand, surfaceTemp, vegFrac);

        for (let v = 0; v < 6; v++) {
          colors.setXYZ(idx, r, g, b);
          idx++;
        }
      }
    }
    colors.needsUpdate = true;
  }

  function averagePolarTemps(monthData: number[][], latCount: number, lonCount: number): number[][] {
    // Reduce longitude resolution near poles to avoid artifacts.
    // Use a moving average window that grows as we approach the poles.
    const latStep = 180 / latCount;
    const result = monthData.map(row => [...row]);

    function smoothRow(row: number[], windowSize: number): number[] {
      if (windowSize <= 1) return row;
      windowSize = Math.floor(windowSize);
      if (windowSize % 2 === 0) windowSize++; // make odd for symmetric window
      const halfWindow = Math.floor(windowSize / 2);
      const newRow = new Array(row.length);

      for (let j = 0; j < row.length; j++) {
        let sum = 0;
        for (let k = -halfWindow; k <= halfWindow; k++) {
          // Wrap around for longitude
          const idx = (j + k + row.length) % row.length;
          sum += row[idx];
        }
        newRow[j] = sum / windowSize;
      }
      return newRow;
    }

    for (let i = 0; i < latCount; i++) {
      // Calculate latitude in degrees (-90 to 90)
      const lat = -90 + (i + 0.5) * latStep;
      const absLat = Math.abs(lat);

      // Above 75° latitude, apply progressively stronger smoothing
      if (absLat > 75) {
        // Window size grows from 1 at 75° to cover ~15° of longitude at the pole
        const t = (absLat - 75) / 15; // 0 at 75°, 1 at 90°
        const maxWindow = lonCount / 24; // ~15° of longitude
        const windowSize = 1 + t * t * maxWindow; // quadratic growth
        result[i] = smoothRow(result[i], windowSize);
      }
    }

    return result;
  }

  function createGlobeMesh(latCount: number, lonCount: number, radius: number): THREE.Mesh {
    // Build geometry manually with per-face colors (no interpolation)
    const positions: number[] = [];
    const colors: number[] = [];

    function getVertex(lat: number, lon: number): [number, number, number] {
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = lon * (Math.PI / 180);
      return [
        -radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta)
      ];
    }

    function addVertex(lat: number, lon: number, r: number, g: number, b: number) {
      const [x, y, z] = getVertex(lat, lon);
      positions.push(x, y, z);
      colors.push(r, g, b);
    }

    const latStep = 180 / latCount;
    const lonStep = 360 / lonCount;

    for (let i = 0; i < latCount; i++) {
      const lat0 = 90 - i * latStep;
      const lat1 = 90 - (i + 1) * latStep;

      for (let j = 0; j < lonCount; j++) {
        const lon0 = j * lonStep;
        const lon1 = (j + 1) * lonStep;

        // Default color (will be updated immediately)
        const r = 0.1, g = 0.1, b = 0.1;

        addVertex(lat0, lon0, r, g, b);
        addVertex(lat1, lon0, r, g, b);
        addVertex(lat1, lon1, r, g, b);

        addVertex(lat0, lon0, r, g, b);
        addVertex(lat1, lon1, r, g, b);
        addVertex(lat0, lon1, r, g, b);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshLambertMaterial({
      vertexColors: true,
      side: THREE.FrontSide,
    });

    return new THREE.Mesh(geometry, material);
  }

  function createTemperatureGlobe(climateData: number[][][]) {
    const latCount = climateData[0].length;
    const lonCount = climateData[0][0].length;
    const mesh = createGlobeMesh(latCount, lonCount, 1);
    return mesh;
  }

  function createBlueMarbleGlobe(ld: ClimateLayerData) {
    const bmNlat = ld.surface.shape[1];
    const bmNlon = ld.surface.shape[2];
    const mesh = createGlobeMesh(bmNlat, bmNlon, 1);
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

    const declination = getSunDeclination(displayMonthProgress);
    const distance = 50;

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

    sunLight.position.copy(sunDir.multiplyScalar(distance));
  }

  function latLonToVector3(lat: number, lon: number, r: number): THREE.Vector3 {
    // Must match the coordinate system used in createGlobeMesh
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = lon * (Math.PI / 180);
    return new THREE.Vector3(
      -r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    );
  }

  function createLineFromCoords(coords: number[][], r: number, darkColor: number, lightColor: number): THREE.Line[] {
    const points: THREE.Vector3[] = [];
    for (const [lon, lat] of coords) {
      points.push(latLonToVector3(lat, lon, r));
    }
    // Two passes: dark outline visible on bright/day side, light outline visible on dark/night side
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const dark = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: darkColor,
      transparent: true,
      opacity: 0.8,
    }));
    const light = new THREE.Line(geom.clone(), new THREE.LineBasicMaterial({
      color: lightColor,
      transparent: true,
      opacity: 0.3,
    }));
    return [dark, light];
  }

  function processMultiLineString(coords: number[][][], r: number, darkColor: number, lightColor: number): THREE.Line[] {
    return coords.flatMap(lineCoords => createLineFromCoords(lineCoords, r, darkColor, lightColor));
  }

  function processPolygon(coords: number[][][], r: number, darkColor: number, lightColor: number): THREE.Line[] {
    return coords.flatMap(ring => createLineFromCoords(ring, r, darkColor, lightColor));
  }

  function processMultiPolygon(coords: number[][][][], r: number, darkColor: number, lightColor: number): THREE.Line[] {
    const lines: THREE.Line[] = [];
    for (const polygon of coords) {
      lines.push(...processPolygon(polygon, r, darkColor, lightColor));
    }
    return lines;
  }

  async function loadBorders() {
    bordersGroup = new THREE.Group();
    const radius = 1.002; // Slightly above globe surface

    try {
      // Load countries for borders
      const response = await fetch('/countries-110m.json');
      const topology = await response.json() as Topology;
      const countries = topology.objects.countries as GeometryCollection;
      const mesh = topojson.mesh(topology, countries);

      if (mesh.type === 'MultiLineString') {
        const lines = processMultiLineString(mesh.coordinates, radius, 0x333333, 0xcccccc);
        lines.forEach(line => bordersGroup!.add(line));
      }

      // Load land for coastlines
      const landResponse = await fetch('/land-110m.json');
      const landTopology = await landResponse.json() as Topology;
      const land = landTopology.objects.land as GeometryCollection;
      const landFeature = topojson.feature(landTopology, land);

      const features = 'features' in landFeature
        ? landFeature.features
        : [landFeature as GeoJSON.Feature];
      for (const feature of features) {
        const geom = feature.geometry;
        if (geom.type === 'Polygon') {
          const lines = processPolygon(geom.coordinates, radius, 0x000000, 0xffffff);
          lines.forEach(line => bordersGroup!.add(line));
        } else if (geom.type === 'MultiPolygon') {
          const lines = processMultiPolygon(geom.coordinates, radius, 0x000000, 0xffffff);
          lines.forEach(line => bordersGroup!.add(line));
        }
      }

      scene.add(bordersGroup);
    } catch (e) {
      console.error('Failed to load borders:', e);
    }
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

  function init() {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    // Dim ambient light so night side is slightly visible
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
    scene.add(ambientLight);

    // Directional light as sun - position updated each frame relative to camera
    sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
    scene.add(sunLight);

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
    controls.minDistance = 1.5;
    controls.maxDistance = 10;
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
      updateTemperatureColors(data, displayMonth);
    }

    // Create blue marble globe if layerData is available
    if (layerData) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      updateBlueMarbleColors(layerData, displayMonth);
      initWindParticles();
    }

    // Load borders
    if (showBorders) {
      loadBorders();
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

    // Smooth interpolation toward target month progress
    // Handle wraparound (e.g., 11.5 -> 0.5 should go forward, not backward)
    let delta = monthProgress - displayMonthProgress;
    if (delta > 6) delta -= 12;
    if (delta < -6) delta += 12;
    displayMonthProgress += delta * 0.05;
    // Keep in 0-12 range
    if (displayMonthProgress < 0) displayMonthProgress += 12;
    if (displayMonthProgress >= 12) displayMonthProgress -= 12;

    // When not auto-rotating, orbit the sun around the globe at 8x auto-rotate speed
    // This gives 1 day per 15 seconds
    if (!isAutoRotating && lastAnimateTime !== null && time !== undefined) {
      const dt = (time - lastAnimateTime) / 1000;
      sunOrbitAngle += dt * controls.autoRotateSpeed * (2 * Math.PI / 60) * 8;
    }

    // Update wind particles
    if (windParticles && time !== undefined && lastAnimateTime !== null) {
      const dt = (time - lastAnimateTime) / 1000;
      windParticles.setMonth(Math.floor(displayMonthProgress) % 12);
      if (activeLayer === 'blue-marble') {
        windParticles.update(dt);
      }
    }

    lastAnimateTime = time ?? null;

    // Update sun position
    updateSunPosition(isAutoRotating);

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
    updateTemperatureColors(data, displayMonth);
  }

  // Create blue marble globe when layerData arrives
  $: if (scene && layerData && !blueMarbleGlobe) {
    blueMarbleGlobe = createBlueMarbleGlobe(layerData);
    blueMarbleGlobe.visible = activeLayer === 'blue-marble';
    scene.add(blueMarbleGlobe);
    updateBlueMarbleColors(layerData, displayMonth);
    initWindParticles();
  }

  onMount(() => {
    init();
  });

  onDestroy(() => {
    cancelAnimationFrame(animationId);
    window.removeEventListener('resize', onResize);
    renderer?.dispose();
    controls?.dispose();
    if (windParticles) {
      windParticles.dispose();
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
