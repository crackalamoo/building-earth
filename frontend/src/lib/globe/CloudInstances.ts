import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { ClimateLayerData } from './loadBinaryData';

const DEG2RAD = Math.PI / 180;
const CLOUD_RADIUS = 1.025;
const SEED = 7919;

const MIN_FRACTION = 0.12;
const MAX_CLOUDS = 500;

// Puff size range
const PUFF_BASE = 0.008;
const PUFF_VARY = 0.006;

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const Y_UP = new THREE.Vector3(0, 1, 0);

function latLonToNormal(lat: number, lon: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = lon * DEG2RAD;
  return new THREE.Vector3(
    -Math.sin(phi) * Math.cos(theta),
    Math.cos(phi),
    Math.sin(phi) * Math.sin(theta),
  );
}

function latLonToPosition(lat: number, lon: number, radius: number): THREE.Vector3 {
  return latLonToNormal(lat, lon).multiplyScalar(radius);
}

// Cloud shape types for variety
function buildCloudGeometry(rand: () => number, shapeType: number): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];

  // Different shapes: 0=small cumulus, 1=big cumulus, 2=elongated, 3=towering, 4=wispy, 5=cluster
  let numPuffs: number;
  let spreadX: number;
  let spreadZ: number;
  let heightRange: number;
  let sizeMin: number;
  let sizeMax: number;

  switch (shapeType) {
    case 0: // small cumulus — compact dome
      numPuffs = 4 + Math.floor(rand() * 2);
      spreadX = 0.008; spreadZ = 0.008; heightRange = 0.008;
      sizeMin = 0.006; sizeMax = 0.012;
      break;
    case 1: // big cumulus — large puffy dome
      numPuffs = 8 + Math.floor(rand() * 4);
      spreadX = 0.014; spreadZ = 0.014; heightRange = 0.014;
      sizeMin = 0.008; sizeMax = 0.016;
      break;
    case 2: // elongated — stretched along one axis
      numPuffs = 6 + Math.floor(rand() * 3);
      spreadX = 0.025; spreadZ = 0.006; heightRange = 0.005;
      sizeMin = 0.006; sizeMax = 0.011;
      break;
    case 3: // towering — tall cumulonimbus-like
      numPuffs = 10 + Math.floor(rand() * 4);
      spreadX = 0.010; spreadZ = 0.010; heightRange = 0.020;
      sizeMin = 0.007; sizeMax = 0.014;
      break;
    case 4: // wispy — few scattered small puffs
      numPuffs = 3 + Math.floor(rand() * 2);
      spreadX = 0.018; spreadZ = 0.012; heightRange = 0.003;
      sizeMin = 0.004; sizeMax = 0.008;
      break;
    default: // cluster — group of distinct blobs
      numPuffs = 7 + Math.floor(rand() * 3);
      spreadX = 0.020; spreadZ = 0.016; heightRange = 0.010;
      sizeMin = 0.005; sizeMax = 0.013;
      break;
  }

  for (let i = 0; i < numPuffs; i++) {
    const r = sizeMin + rand() * (sizeMax - sizeMin);
    const sphere = new THREE.IcosahedronGeometry(r, 2);

    const pos = sphere.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    colors.fill(1.0);
    sphere.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    // Position: bigger puffs near center and top (dome shape)
    const angle = rand() * Math.PI * 2;
    const distFrac = rand();
    const px = Math.cos(angle) * distFrac * spreadX;
    const pz = Math.sin(angle) * distFrac * spreadZ;
    // Higher puffs near center, lower at edges
    const py = (1.0 - distFrac * 0.7) * heightRange * rand();
    sphere.translate(px, py, pz);

    parts.push(sphere);
  }

  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  return merged!;
}

// ---- Cloud placement ----

interface CloudInfo {
  baseLat: number;
  baseLon: number;
  driftLat: number;
  driftLon: number;
  windU: number; // current interpolated wind in deg/s
  windV: number;
  cellI: number;
  cellJ: number;
  currentOpacity: number;
  targetOpacity: number;
  index: number; // index within its InstancedMesh
  meshIndex: number; // which InstancedMesh template
}

// Custom shader for clouds: Lambert + sun-aware + opacity
const cloudVertexShader = `
  attribute float instanceOpacity;
  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  varying float vOpacity;

  void main() {
    vColor = color;
    vNormal = normalize(mat3(modelMatrix) * normal);
    vec4 worldPos = modelMatrix * instanceMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    vOpacity = instanceOpacity;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const cloudFragmentShader = `
  uniform vec3 sunDirection;

  varying vec3 vColor;
  varying vec3 vNormal;
  varying vec3 vWorldPos;
  varying float vOpacity;

  void main() {
    if (vOpacity < 0.01) discard;

    vec3 normal = normalize(vNormal);
    vec3 cloudDir = normalize(vWorldPos); // radial direction from globe center

    // Sun illumination based on cloud's position on globe
    float sunDot = dot(cloudDir, sunDirection);
    float daylight = smoothstep(-0.15, 0.3, sunDot);

    // Diffuse lighting on the puff surface
    float NdotL = max(dot(normal, sunDirection), 0.0);
    float ambient = 0.45; // clouds are bright even in shadow
    float diffuse = ambient + (1.0 - ambient) * NdotL;

    // Warm sunlit → cool shadow color
    vec3 litColor = vec3(1.0, 0.99, 0.95);
    vec3 shadowColor = vec3(0.65, 0.70, 0.82);
    vec3 baseColor = mix(shadowColor, litColor, daylight);

    // Night dimming
    float brightness = 0.12 + 0.88 * daylight;

    if (vOpacity < 0.01) discard;
    gl_FragColor = vec4(baseColor * diffuse * brightness, vOpacity);
  }
`;

export class CloudInstances {
  private group: THREE.Group;
  private clouds: CloudInfo[] = [];
  private sunDir: THREE.Vector3 = new THREE.Vector3(1, 0, 0);
  private nlat: number = 0;
  private nlon: number = 0;
  private lastMonthProgress: number = -1;
  private meshes: THREE.InstancedMesh[] = [];
  private opacityAttrs: THREE.InstancedBufferAttribute[] = [];
  private dummy = new THREE.Object3D();

  constructor(layerData: ClimateLayerData) {
    this.group = new THREE.Group();

    const totalField = layerData.cloud_fraction;
    if (!totalField) return;

    this.nlat = totalField.shape[1];
    this.nlon = totalField.shape[2];
    const nlat = this.nlat;
    const nlon = this.nlon;
    const latStep = 180 / nlat;
    const lonStep = 360 / nlon;
    const data = totalField.data as Float32Array;

    // Compute annual means per cell
    const annualMean = new Float32Array(nlat * nlon);
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nlon; j++) {
        let csum = 0;
        for (let m = 0; m < 12; m++) {
          csum += data[m * nlat * nlon + i * nlon + j];
        }
        annualMean[i * nlon + j] = csum / 12;
      }
    }

    const rand = mulberry32(SEED);

    // Collect cloudy cells
    const cells: { i: number; j: number; mean: number }[] = [];
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nlon; j++) {
        const mean = annualMean[i * nlon + j];
        if (mean >= MIN_FRACTION) {
          cells.push({ i, j, mean });
        }
      }
    }
    // Shuffle
    for (let n = cells.length - 1; n > 0; n--) {
      const swap = Math.floor(rand() * (n + 1));
      [cells[n], cells[swap]] = [cells[swap], cells[n]];
    }

    // Build several varied cloud geometry templates
    const NUM_TEMPLATES = 6;
    const cloudGeos: THREE.BufferGeometry[] = [];
    for (let t = 0; t < NUM_TEMPLATES; t++) {
      cloudGeos.push(buildCloudGeometry(mulberry32(12345 + t * 997), t));
    }

    // Place clouds, assigning each to a random template
    const cloudInfos: CloudInfo[] = [];
    const perTemplate: number[] = new Array(NUM_TEMPLATES).fill(0);
    let placed = 0;
    for (const cell of cells) {
      if (placed >= MAX_CLOUDS) break;
      const { i, j, mean } = cell;

      const expected = mean * 3.5;
      let count = Math.floor(expected);
      if (rand() < (expected - count)) count++;
      if (count < 1) count = 1;

      for (let k = 0; k < count && placed < MAX_CLOUDS; k++) {
        const lat = -90 + i * latStep + rand() * latStep;
        const lon = (j * lonStep + rand() * lonStep) % 360;

        // Wind will be set dynamically in setMonth()
        const windU = 0;
        const windV = 0;

        const meshIdx = Math.floor(rand() * NUM_TEMPLATES);
        const idxInMesh = perTemplate[meshIdx];
        perTemplate[meshIdx]++;

        cloudInfos.push({
          baseLat: lat, baseLon: lon,
          driftLat: 0, driftLon: 0,
          windU, windV,
          cellI: i, cellJ: j,
          currentOpacity: 0, targetOpacity: 0,
          index: idxInMesh,
          meshIndex: meshIdx,
        });
        placed++;
      }
    }
    this.clouds = cloudInfos;

    // Create one InstancedMesh per template
    for (let t = 0; t < NUM_TEMPLATES; t++) {
      const count = perTemplate[t];
      if (count === 0) continue;

      const material = new THREE.ShaderMaterial({
        vertexShader: cloudVertexShader,
        fragmentShader: cloudFragmentShader,
        uniforms: {
          sunDirection: { value: this.sunDir },
        },
        vertexColors: true,
        transparent: true,
        depthWrite: false,
        side: THREE.FrontSide,
      });

      const mesh = new THREE.InstancedMesh(cloudGeos[t], material, count);
      mesh.renderOrder = 10;

      const opacities = new Float32Array(count);
      const attr = new THREE.InstancedBufferAttribute(opacities, 1);
      mesh.geometry.setAttribute('instanceOpacity', attr);

      this.meshes.push(mesh);
      this.opacityAttrs.push(attr);
      this.group.add(mesh);
    }

    // Map template index to mesh array index (some templates may have 0 clouds)
    // Build lookup: meshIndex -> index in this.meshes
    const templateToMeshIdx: number[] = [];
    let meshIdx = 0;
    for (let t = 0; t < NUM_TEMPLATES; t++) {
      if (perTemplate[t] > 0) {
        templateToMeshIdx.push(meshIdx++);
      } else {
        templateToMeshIdx.push(-1);
      }
    }
    // Remap cloud meshIndex to actual mesh array index
    for (const cloud of this.clouds) {
      cloud.meshIndex = templateToMeshIdx[cloud.meshIndex];
    }

    // Set initial transforms
    for (const cloud of this.clouds) {
      this.updateCloudTransform(cloud);
    }
    for (const mesh of this.meshes) {
      mesh.instanceMatrix.needsUpdate = true;
    }
  }

  private updateCloudTransform(cloud: CloudInfo): void {
    const lon = ((cloud.baseLon + cloud.driftLon) % 360 + 360) % 360;
    const lat = Math.max(-89, Math.min(89, cloud.baseLat + cloud.driftLat));
    const r = CLOUD_RADIUS;

    const normal = latLonToNormal(lat, lon);
    const pos = normal.clone().multiplyScalar(r);

    this.dummy.position.copy(pos);
    // Orient cloud to face outward from globe
    this.dummy.quaternion.setFromUnitVectors(Y_UP, normal);
    // Scale variation per cloud
    const hash = (cloud.index * 7 + cloud.meshIndex * 13) % 17;
    const s = 0.75 + hash * 0.04;
    this.dummy.scale.set(s, s * 0.8, s);
    this.dummy.updateMatrix();
    this.meshes[cloud.meshIndex].setMatrixAt(cloud.index, this.dummy.matrix);
  }

  setMonth(monthProgress: number, layerData: ClimateLayerData): void {
    if (Math.abs(monthProgress - this.lastMonthProgress) < 0.005) return;
    this.lastMonthProgress = monthProgress;

    const mp = ((monthProgress % 12) + 12) % 12;
    const m0 = Math.floor(mp) % 12;
    const m1 = (m0 + 1) % 12;
    const frac = mp - Math.floor(mp);

    const nlat = this.nlat;
    const nlon = this.nlon;

    const totalField = layerData.cloud_fraction;
    if (!totalField) return;
    const data = totalField.data as Float32Array;

    const windUField = layerData.wind_u_10m;
    const windVField = layerData.wind_v_10m;
    const windUData = windUField?.data as Float32Array | undefined;
    const windVData = windVField?.data as Float32Array | undefined;
    const windScale = 5000.0;

    const latStep = 180 / nlat;
    const lonStep = 360 / nlon;

    for (const cloud of this.clouds) {
      // Use current drifted position for wind/cloud lookups
      const curLat = Math.max(-89, Math.min(89, cloud.baseLat + cloud.driftLat));
      const curLon = ((cloud.baseLon + cloud.driftLon) % 360 + 360) % 360;
      const ci = Math.min(nlat - 1, Math.max(0, Math.floor((curLat + 90) / latStep)));
      const cj = Math.min(nlon - 1, Math.floor(curLon / lonStep));

      const idx0 = m0 * nlat * nlon + ci * nlon + cj;
      const idx1 = m1 * nlat * nlon + ci * nlon + cj;

      const v0 = data[idx0];
      const v1 = data[idx1];
      const fraction = Math.max(0, Math.min(1, v0 + (v1 - v0) * frac));
      cloud.targetOpacity = Math.min(1, fraction * 1.4);

      // Sample wind at current position
      if (windUData && windVData) {
        const cosLat = Math.cos(curLat * DEG2RAD);
        const u = windUData[idx0] + (windUData[idx1] - windUData[idx0]) * frac;
        const v = windVData[idx0] + (windVData[idx1] - windVData[idx0]) * frac;
        cloud.windU = (u / (111000 * Math.max(cosLat, 0.15))) * windScale;
        cloud.windV = (v / 111000) * windScale;
      }
    }
  }

  update(dt: number, camera?: THREE.Camera): void {
    if (this.meshes.length === 0) return;

    for (const cloud of this.clouds) {
      // Smooth opacity
      const diff = cloud.targetOpacity - cloud.currentOpacity;
      if (Math.abs(diff) > 0.001) {
        cloud.currentOpacity += Math.sign(diff) * Math.min(Math.abs(diff), 0.08 * dt);
      }

      this.opacityAttrs[cloud.meshIndex].array[cloud.index] = cloud.currentOpacity;

      // Drift
      cloud.driftLon += cloud.windU * dt;
      cloud.driftLat += cloud.windV * dt;

      this.updateCloudTransform(cloud);
    }

    for (let i = 0; i < this.meshes.length; i++) {
      this.meshes[i].instanceMatrix.needsUpdate = true;
      this.opacityAttrs[i].needsUpdate = true;
    }
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.sunDir.copy(dir);
  }

  getObject(): THREE.Object3D {
    return this.group;
  }

  dispose(): void {
    for (const mesh of this.meshes) {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    }
    this.meshes = [];
    this.opacityAttrs = [];
    this.clouds = [];
  }
}
