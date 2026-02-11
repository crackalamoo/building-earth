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
  import type { ClimateLayerData } from './loadBinaryData';
  import { ELEVATION_SCALE, NORMAL_BLEND, sampleElevation, displacedNormal, computeHillshadeGrid } from './elevation';

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
  let sunGlow: THREE.Sprite | null = null;
  let starField: StarField | null = null;

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
      if (atmosphereMesh) atmosphereMesh.visible = layer === 'blue-marble';
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
    const soilData = ld.soil_moisture?.data as Float32Array | undefined;
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

        const [r, g, b] = blueMarbleColor(isLand, surfaceTemp, soilMoisture, elev);
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

    // Write smoothed RGB to vertex colors, applying hillshade
    // Also update specular mask: open ocean=1, sea ice/land=0, smooth transition
    const hsGrid = (blueMarbleGlobe as any)?._hillshadeGrid as Float32Array | undefined;
    const specAttr = geometry.attributes.isOcean as THREE.BufferAttribute | undefined;
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
        if (specAttr) {
          const isLand = landMaskData[dataLatIdx * hiNlon + j] === 1;
          if (!isLand) {
            const temp = sampleBilinear(
              surfaceData, lowNlat, lowNlon, monthOffset,
              dataLatIdx, j, hiNlat, hiNlon,
              false, coarseLandMask,
            );
            // Mirror seaIceFraction from blueMarbleColormap.ts:
            // Hermite smoothstep from -1.8°C to -8°C, max 0.70
            const ICE_FREEZE = -1.8, ICE_FULL = -8.0, ICE_MAX = 0.70;
            const u = Math.max(0, Math.min(1, (ICE_FREEZE - temp) / (ICE_FREEZE - ICE_FULL)));
            const iceFrac = u * u * (3 - 2 * u) * ICE_MAX;
            spec = 1.0 - iceFrac;
          }
        }

        for (let v = 0; v < 6; v++) {
          colors.setXYZ(idx, r, g, b);
          if (specAttr) specAttr.setX(idx, spec);
          idx++;
        }
      }
    }
    colors.needsUpdate = true;
    if (specAttr) specAttr.needsUpdate = true;
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

    function addVertex(lat: number, lon: number, r: number, g: number, b: number) {
      const [x, y, z] = getVertex(lat, lon);
      positions.push(x, y, z);
      colors.push(r, g, b);

      const phi = (90 - lat) * (Math.PI / 180);
      const theta = lon * (Math.PI / 180);
      const snx = -Math.sin(phi) * Math.cos(theta);
      const sny = Math.cos(phi);
      const snz = Math.sin(phi) * Math.sin(theta);

      if (elevationData && elevNlat && elevNlon) {
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
          float NdotL = max(dot(normal, sunDirection), 0.0);
          vec3 diffuse = vColor * (ambientIntensity + NdotL * (1.0 - ambientIntensity));

          // Early out for land/ice/night — no specular
          if (vIsOcean < 0.01 || NdotL <= 0.0) {
            gl_FragColor = vec4(diffuse, 1.0);
            return;
          }

          // Blinn-Phong specular with wave perturbation (ocean only)
          vec3 viewDir = normalize(cameraPosition - vWorldPos);
          vec3 halfVec = normalize(sunDirection + viewDir);

          float spec = pow(max(dot(normal, halfVec), 0.0), 200.0) * 0.7 * vIsOcean;
          gl_FragColor = vec4(diffuse + vec3(1.0, 0.97, 0.90) * spec, 1.0);
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

    // Glow sprite behind sun orb
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = 128;
    glowCanvas.height = 128;
    const ctx = glowCanvas.getContext('2d')!;
    const grad = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
    grad.addColorStop(0, 'rgba(255, 220, 160, 0.6)');
    grad.addColorStop(0.3, 'rgba(255, 180, 80, 0.3)');
    grad.addColorStop(1, 'rgba(255, 140, 40, 0.0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 128, 128);
    const glowTex = new THREE.CanvasTexture(glowCanvas);
    const glowMat = new THREE.SpriteMaterial({
      map: glowTex,
      blending: THREE.AdditiveBlending,
      transparent: true,
      depthWrite: false,
    });
    sunGlow = new THREE.Sprite(glowMat);
    sunGlow.scale.set(56, 56, 1);
    sunOrb.add(sunGlow);

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
      updateTemperatureColors(data, displayMonth);
    }

    // Create blue marble globe if layerData is available
    if (layerData) {
      blueMarbleGlobe = createBlueMarbleGlobe(layerData);
      blueMarbleGlobe.visible = activeLayer === 'blue-marble';
      scene.add(blueMarbleGlobe);
      updateBlueMarbleColors(layerData, displayMonth);
      initWindParticles();
      initTreeInstances();
      initCloudInstances();
    }

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
      loadBorders().then(group => {
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

    // Smooth interpolation toward target month progress
    // Handle wraparound (e.g., 11.5 -> 0.5 should go forward, not backward)
    let delta = monthProgress - displayMonthProgress;
    if (delta > 6) delta -= 12;
    if (delta < -6) delta += 12;
    displayMonthProgress += delta * 0.05;
    // Keep in 0-12 range
    if (displayMonthProgress < 0) displayMonthProgress += 12;
    if (displayMonthProgress >= 12) displayMonthProgress -= 12;

    // When not auto-rotating, orbit the sun around the globe at 4x auto-rotate speed
    // This gives 1 day per ~60 seconds
    if (!isAutoRotating && lastAnimateTime !== null && time !== undefined) {
      const dt = (time - lastAnimateTime) / 1000;
      sunOrbitAngle += dt * controls.autoRotateSpeed * (2 * Math.PI / 60) * 2;
    }

    // Update wind particles
    if (windParticles && time !== undefined && lastAnimateTime !== null) {
      const dt = (time - lastAnimateTime) / 1000;
      windParticles.setMonth(Math.floor(displayMonthProgress) % 12);
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
        (sunGlow.material as THREE.SpriteMaterial).map?.dispose();
        (sunGlow.material as THREE.Material).dispose();
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
