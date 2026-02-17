import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { ClimateLayerData } from './loadBinaryData';
import { ELEVATION_SCALE, sampleElevation } from './elevation';
import { sampleBilinear } from './gridSampling';

const GLOBE_RADIUS = 1.001;
const TREE_HEIGHT = 0.025;
const SEED = 42;

// Mulberry32 seeded PRNG
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
const DEG2RAD = Math.PI / 180;

function latLonToNormal(lat: number, lon: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = lon * DEG2RAD;
  return new THREE.Vector3(
    -Math.sin(phi) * Math.cos(theta),
    Math.cos(phi),
    Math.sin(phi) * Math.sin(theta),
  );
}

interface TreeInstance {
  type: 0 | 1 | 2; // 0=conifer, 1=broadleaf, 2=palm
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
  scaleXZ: number;
  scaleY: number;
  lat: number;
  lon: number;
  cellI: number;
  cellJ: number;
}

/** Convert to non-indexed, keep only position+normal+color for merge compatibility. */
function prepareForMerge(
  geom: THREE.BufferGeometry,
  r: number, g: number, b: number,
): THREE.BufferGeometry {
  const ni = geom.toNonIndexed();
  geom.dispose();
  ni.deleteAttribute('uv');
  ni.computeVertexNormals();
  const count = ni.attributes.position.count;
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    colors[i * 3] = r;
    colors[i * 3 + 1] = g;
    colors[i * 3 + 2] = b;
  }
  ni.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return ni;
}

function buildConiferGeometry(): THREE.BufferGeometry {
  const h = TREE_HEIGHT;
  const trunk = prepareForMerge(
    new THREE.CylinderGeometry(h * 0.06, h * 0.08, h * 0.35, 5, 1),
    0.40, 0.26, 0.13,
  );
  trunk.translate(0, h * 0.175, 0);
  const foliage = prepareForMerge(
    new THREE.ConeGeometry(h * 0.28, h * 0.65, 6, 1),
    0.10, 0.45, 0.12,
  );
  foliage.translate(0, h * 0.35 + h * 0.325, 0);
  const merged = mergeGeometries([trunk, foliage], false);
  trunk.dispose();
  foliage.dispose();
  return merged!;
}

function buildBroadleafTrunkGeometry(): THREE.BufferGeometry {
  const h = TREE_HEIGHT;
  const trunk = prepareForMerge(
    new THREE.CylinderGeometry(h * 0.05, h * 0.07, h * 0.4, 5, 1),
    0.45, 0.30, 0.15,
  );
  trunk.translate(0, h * 0.2, 0);
  return trunk;
}

function buildBroadleafFoliageGeometry(): THREE.BufferGeometry {
  const h = TREE_HEIGHT;
  // White vertex colors — actual color is controlled entirely by instance color
  // so we can set it to green, orange, red, or brown seasonally
  const foliage = prepareForMerge(
    new THREE.IcosahedronGeometry(h * 0.30, 1),
    1.0, 1.0, 1.0,
  );
  foliage.translate(0, h * 0.4 + h * 0.25, 0);
  return foliage;
}

function buildPalmGeometry(): THREE.BufferGeometry {
  const h = TREE_HEIGHT * 1.8;
  const crownY = h * 0.78;
  const parts: THREE.BufferGeometry[] = [];
  const trunk = prepareForMerge(
    new THREE.CylinderGeometry(h * 0.02, h * 0.045, h * 0.76, 4, 1),
    0.55, 0.40, 0.22,
  );
  trunk.translate(0, h * 0.38, 0);
  parts.push(trunk);
  const hub = prepareForMerge(
    new THREE.SphereGeometry(h * 0.12, 5, 4),
    0.08, 0.50, 0.08,
  );
  hub.translate(0, crownY, 0);
  parts.push(hub);
  const numFronds = 5;
  for (let i = 0; i < numFronds; i++) {
    const angle = (i / numFronds) * Math.PI * 2;
    const frond = prepareForMerge(
      new THREE.TetrahedronGeometry(h * 0.17, 0),
      0.06, 0.52, 0.06,
    );
    frond.scale(0.5, 0.5, 2.2);
    frond.rotateX(0.35);
    frond.rotateY(angle);
    frond.translate(0, crownY, 0);
    parts.push(frond);
  }
  const merged = mergeGeometries(parts, false);
  for (const part of parts) part.dispose();
  return merged!;
}

function computeTypeProbabilities(
  coldestMonth: number,
  warmestMonth: number,
  annualSoilMoisture: number,
): [number, number, number] {
  let palm = 0;
  if (coldestMonth > 10 && warmestMonth > 24) {
    const frostFree = Math.min((coldestMonth - 10) / 8, 1);
    palm = frostFree * (0.3 + 0.7 * annualSoilMoisture) * 1.5;
  }
  let conifer = 0;
  {
    const coldWinter = coldestMonth < 8
      ? Math.min((8 - coldestMonth) / 30, 1)
      : 0;
    const coolSummer = warmestMonth < 24
      ? Math.min((24 - warmestMonth) / 8, 1)
      : 0;
    const tropicalFade = coldestMonth > 15
      ? Math.max(0, 1 - (coldestMonth - 15) / 5)
      : 1;
    conifer = (coldWinter + coolSummer * 1.2) * tropicalFade;
    conifer *= (0.4 + 0.6 * annualSoilMoisture);
  }
  let broadleaf = 0;
  if (warmestMonth > 15 && coldestMonth > -15) {
    const summerWarmth = Math.min((warmestMonth - 15) / 10, 1);
    const winterSurvival = Math.min((coldestMonth + 15) / 15, 1);
    broadleaf = summerWarmth * winterSurvival * (0.4 + 0.6 * annualSoilMoisture);
  }
  if (coldestMonth > 18 && annualSoilMoisture > 0.4) {
    const tropicalBroad = Math.min((coldestMonth - 18) / 5, 1) * annualSoilMoisture;
    broadleaf = Math.max(broadleaf, tropicalBroad * 1.2);
  }
  const total = palm + conifer + broadleaf;
  if (total < 1e-6) return [0, 1, 0];
  return [conifer / total, broadleaf / total, palm / total];
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/**
 * Pre-compute 12 monthly samples of foliageScale and autumnTint for a grid cell.
 * setMonth() lerps between adjacent months — no phase boundaries, no discontinuities.
 *
 * Logic per month:
 * - senescence (leaf loss drive): how far temp is below 15°C AND days shortening
 * - coldStress: how far temp is below 5°C (accelerates leaf loss)
 * - growthDrive: how far temp is above 10°C AND days lengthening
 * - foliageScale: accumulated from growth vs loss
 * - autumnTint: present when foliage is declining, fades when growing
 */
function computeCellCurves(
  tempData: Float32Array,
  nlat: number, nlon: number,
  cellI: number, cellJ: number,
  lat: number,
  cloudData: Float32Array | undefined,
  cloudNlat: number, cloudNlon: number,
  cloudCellI?: number, cloudCellJ?: number,
): {
  foliageCurve: Float32Array;   // 12 values, 0-1
  autumnTintCurve: Float32Array; // 12 values, 0-1
  colorQuality: number;
  evergreen: boolean;
} {
  // Get 12 monthly temperatures
  const temps: number[] = [];
  for (let m = 0; m < 12; m++) {
    temps.push(tempData[m * nlat * nlon + cellI * nlon + cellJ]);
  }

  let warmestTemp = -Infinity, coldestTemp = Infinity;
  let warmestMonth = 0;
  for (let m = 0; m < 12; m++) {
    if (temps[m] > warmestTemp) { warmestTemp = temps[m]; warmestMonth = m; }
    if (temps[m] < coldestTemp) { coldestTemp = temps[m]; }
  }

  // Evergreen: never drops below 15°C
  if (coldestTemp > 15) {
    const fc = new Float32Array(12).fill(1);
    const at = new Float32Array(12).fill(0);
    return { foliageCurve: fc, autumnTintCurve: at, colorQuality: 0, evergreen: true };
  }

  // Minimum foliage: mild winters don't lose all leaves
  const minFoliage = clamp((coldestTemp - 5) / 10, 0, 0.85);

  // Compute color quality from autumn conditions
  // Find the month with steepest cooling after warmest
  let peakAutumnMonth = (warmestMonth + 3) % 12;
  let steepest = 0;
  for (let step = 1; step < 6; step++) {
    const m0 = (warmestMonth + step) % 12;
    const m1 = (m0 + 1) % 12;
    const drop = temps[m0] - temps[m1];
    if (drop > steepest) { steepest = drop; peakAutumnMonth = m1; }
  }
  const autumnTemp = temps[peakAutumnMonth];
  const coolNightFactor = clamp((7 - autumnTemp) / 7, 0, 1) * clamp((autumnTemp + 5) / 10, 0, 1);
  let clearSkyFactor = 0.5;
  if (cloudData && cloudNlat > 0) {
    const ci = cloudCellI ?? cellI, cj = cloudCellJ ?? cellJ;
    const clouds = cloudData[peakAutumnMonth * cloudNlat * cloudNlon + ci * cloudNlon + cj];
    clearSkyFactor = 1 - clamp(clouds, 0, 1);
  }
  const colorQuality = clamp(coolNightFactor * 0.6 + clearSkyFactor * 0.4, 0, 1);

  // Compute per-month signals by iterating around the year starting from warmest month
  // (when foliage is at max). We simulate foliage state forward.
  const foliageCurve = new Float32Array(12);
  const autumnTintCurve = new Float32Array(12);

  // Start at warmest month: full green canopy
  let foliage = 1.0;
  let tint = 0.0;

  // Two passes: first pass builds the curve, second pass corrects
  // so the cycle is periodic (ends where it started)
  for (let pass = 0; pass < 3; pass++) {
    for (let step = 0; step < 12; step++) {
      const m = (warmestMonth + step) % 12;
      const temp = temps[m];
      const nextTemp = temps[(m + 1) % 12];
      const cooling = temp > nextTemp; // temperature trending down

      // Senescence drive: temp below 15°C and cooling
      const senescence = cooling ? clamp((15 - temp) / 10, 0, 1) : 0;

      // Cold stress: accelerates leaf loss below 5°C
      const coldStress = clamp((5 - temp) / 10, 0, 0.5);

      // Growth drive: temp above 10°C and warming (or spring conditions)
      const warming = !cooling;
      const growthDrive = warming ? clamp((temp - 5) / 10, 0, 1) : 0;

      // Update foliage
      const loss = (senescence + coldStress) * 0.4;
      const gain = growthDrive * 0.35;
      foliage = clamp(foliage - loss + gain, 0, 1);

      // Remap so it never goes below minFoliage
      const remapped = minFoliage + (1 - minFoliage) * foliage;

      // Autumn tint: builds when losing foliage, fades when gaining
      if (loss > gain && foliage < 0.9) {
        tint = clamp(tint + (loss - gain) * 2, 0, 1);
      } else if (gain > loss) {
        tint = clamp(tint - gain * 1.5, 0, 1);
      }

      foliageCurve[m] = remapped;
      autumnTintCurve[m] = tint;
    }
  }

  return { foliageCurve, autumnTintCurve, colorQuality, evergreen: false };
}

export class TreeInstances {
  private group: THREE.Group;
  private meshes: THREE.InstancedMesh[] = [];

  // Broadleaf seasonal data
  private broadleafFoliageMesh: THREE.InstancedMesh | null = null;
  private broadleafOrigScaleXZ: Float32Array = new Float32Array(0);
  private broadleafOrigScaleY: Float32Array = new Float32Array(0);
  private broadleafBaseColors: Float32Array = new Float32Array(0); // per-instance brightness
  private broadleafPositions: Float32Array = new Float32Array(0); // per-instance [x,y,z]
  private broadleafQuaternions: Float32Array = new Float32Array(0); // per-instance [x,y,z,w]
  // Pre-computed monthly curves per cell (shared across instances in same cell)
  // Stored as flat arrays: instance k uses cellCurveIndex[k] to index into curves
  private cellFoliageCurves: Float32Array = new Float32Array(0);   // [numCells * 12]
  private cellAutumnTintCurves: Float32Array = new Float32Array(0); // [numCells * 12]
  private cellColorQuality: Float32Array = new Float32Array(0);     // [numCells]
  private broadleafCellIndex: Uint16Array = new Uint16Array(0);     // per-instance cell index
  private broadleafEvergreen: Uint8Array = new Uint8Array(0);       // 1 = tropical evergreen
  private lastMonthProgress: number = -1;
  private sunDirUniform: { value: THREE.Vector3 } = { value: new THREE.Vector3(1, 0, 0) };

  constructor(layerData: ClimateLayerData) {
    this.group = new THREE.Group();

    const rand = mulberry32(SEED);

    const surfaceData = layerData.surface.data as Float32Array;
    const t2mData = layerData.temperature_2m.data as Float32Array;
    const t2mNlat = layerData.temperature_2m.shape[1];
    const t2mNlon = layerData.temperature_2m.shape[2];
    const vegData = layerData.vegetation_fraction!.data as Float32Array;
    const soilData = layerData.soil_moisture!.data as Float32Array;
    const landMaskNative = layerData.land_mask_native!.data as Uint8Array;
    const landMaskHiRes = layerData.land_mask.data as Uint8Array;
    const cloudData = layerData.cloud_fraction?.data as Float32Array | undefined;
    const cloudField = layerData.cloud_fraction;

    const nativeNlat = layerData.surface.shape[1];
    const nativeNlon = layerData.surface.shape[2];
    const hiNlat = layerData.land_mask.shape[0];
    const hiNlon = layerData.land_mask.shape[1];

    const latStep = 180 / nativeNlat;
    const lonStep = 360 / nativeNlon;

    // Collect instances per type
    const instances: [TreeInstance[], TreeInstance[], TreeInstance[]] = [[], [], []];

    for (let i = 0; i < nativeNlat; i++) {
      for (let j = 0; j < nativeNlon; j++) {
        // Coarse cell stats for tree type probabilities (only if coarse land)
        const isCoarseLand = landMaskNative[i * nativeNlon + j] === 1;
        let coarseVegSum = 0, coarseSoilSum = 0;
        let monthsAbove10 = 0;
        if (isCoarseLand) {
          // Use T2m at coarse cell center for growing season estimate
          const t2mCI = Math.floor((i + 0.5) * t2mNlat / nativeNlat);
          const t2mCJ = Math.floor((j + 0.5) * t2mNlon / nativeNlon);
          for (let m = 0; m < 12; m++) {
            const t2mTemp = t2mData[m * t2mNlat * t2mNlon + t2mCI * t2mNlon + t2mCJ];
            if (t2mTemp > 10) monthsAbove10++;
            const offset = m * nativeNlat * nativeNlon + i * nativeNlon + j;
            coarseVegSum += vegData[offset];
            coarseSoilSum += soilData[offset];
          }
        }
        const coarseAnnualVeg = isCoarseLand ? coarseVegSum / 12 : 0;
        const coarseAnnualSoil = isCoarseLand ? Math.min(coarseSoilSum / 12, 1) : 0.5;

        // Tree density: use minimum thresholds (palm/conifer), reject per-type later
        // Global gate is permissive; per-type survival checks happen after type selection
        const vegU = Math.max(0, Math.min(1, (coarseAnnualVeg - 0.1) / 0.6));
        const vegGate = vegU * vegU * (3 - 2 * vegU); // smoothstep 0.1→0.7
        const moistU = Math.max(0, Math.min(1, (coarseAnnualSoil - 0.05) / 0.2));
        const moistGate = moistU * moistU * (3 - 2 * moistU); // smoothstep 0.05→0.25
        const gsU = Math.max(0, Math.min(1, (monthsAbove10 - 3) / 3));
        const treeGrowingSeason = gsU * gsU * (3 - 2 * gsU); // hermite
        const treeDensity = vegGate * moistGate * treeGrowingSeason;

        // Max possible trees per coarse cell — try more candidates for
        // cells that are partially ocean at coarse res but have hi-res land
        const maxCandidates = isCoarseLand
          ? Math.floor(treeDensity * 7 + rand())
          : 2; // small budget for coastal spillover
        if (maxCandidates <= 0) { rand(); continue; } // consume one rand for determinism

        const cellLatSouth = -90 + i * latStep;
        const cellLonWest = j * lonStep;

        for (let t = 0; t < maxCandidates; t++) {
          let lat: number = 0, lon: number = 0;
          let valid = false;

          for (let attempt = 0; attempt < 3; attempt++) {
            lat = cellLatSouth + rand() * latStep;
            lon = cellLonWest + rand() * lonStep;
            if (lon >= 360) lon -= 360;

            const hiRenderI = Math.floor(((90 - lat) / 180) * hiNlat);
            const hiDataI = hiNlat - 1 - Math.min(hiRenderI, hiNlat - 1);
            const hiJ = Math.floor((lon / 360) * hiNlon) % hiNlon;
            const hiIdx = hiDataI * hiNlon + hiJ;
            if (landMaskHiRes[hiIdx] === 1) {
              valid = true;
              break;
            }
          }

          if (!valid) continue;

          // Sample T2m at this tree's position for treeline check and type selection
          const hiRenderI = Math.floor(((90 - lat) / 180) * hiNlat);
          const hiDataI = hiNlat - 1 - Math.min(hiRenderI, hiNlat - 1);
          const hiJ = Math.floor((lon / 360) * hiNlon) % hiNlon;
          // Map hi-res pixel to T2m grid
          const t2mI = Math.floor(hiDataI * t2mNlat / hiNlat);
          const t2mJ = Math.floor(hiJ * t2mNlon / hiNlon);
          let t2mWarmest = -Infinity, t2mColdest = Infinity;
          let t2mWarmCount = 0;
          for (let m = 0; m < 12; m++) {
            const t = t2mData[m * t2mNlat * t2mNlon + t2mI * t2mNlon + t2mJ];
            if (t > t2mWarmest) t2mWarmest = t;
            if (t < t2mColdest) t2mColdest = t;
            if (t > 10) t2mWarmCount++;
          }
          // Treeline: reject if warmest month < 10°C (no trees above alpine treeline)
          if (t2mWarmest < 10) continue;

          // For non-coarse-land cells, also check veg density
          if (!isCoarseLand) {
            let vegAtPoint = 0, localSoilSum = 0;
            for (let m = 0; m < 12; m++) {
              const mOff = m * nativeNlat * nativeNlon;
              vegAtPoint += sampleBilinear(
                vegData, nativeNlat, nativeNlon, mOff,
                hiDataI, hiJ, hiNlat, hiNlon,
                true, landMaskNative,
              );
              localSoilSum += sampleBilinear(
                soilData, nativeNlat, nativeNlon, mOff,
                hiDataI, hiJ, hiNlat, hiNlon,
                true, landMaskNative,
              );
            }
            vegAtPoint /= 12;
            const vU = Math.max(0, Math.min(1, (vegAtPoint - 0.25) / 0.5));
            const vS = vU * vU * (3 - 2 * vU);
            const localSoil = Math.min(localSoilSum / 12, 1);
            const mU = Math.max(0, Math.min(1, (localSoil - 0.1) / 0.2));
            const mG = mU * mU * (3 - 2 * mU);
            const gsU2 = Math.max(0, Math.min(1, (t2mWarmCount - 3) / 3));
            const td = vS * mG * gsU2 * gsU2 * (3 - 2 * gsU2);
            if (td < 0.01 || rand() > td) continue;
          }
          // Type selection from T2m at tree position
          const localSoilForType = isCoarseLand ? coarseAnnualSoil : 0.5;
          const [localPConifer, localPBroadleaf] = computeTypeProbabilities(t2mColdest, t2mWarmest, localSoilForType);

          const r = rand();
          let type: 0 | 1 | 2;
          if (r < localPConifer) type = 0;
          else if (r < localPConifer + localPBroadleaf) type = 1;
          else type = 2;

          // Per-type survival: different veg/moisture thresholds
          // Conifer: drought-tolerant, grows on thin dry soils (boreal)
          // Broadleaf: needs most moisture and ground cover
          // Palm: low veg ok (oases), just needs warmth + some moisture
          const localMoist = isCoarseLand ? coarseAnnualSoil : 0.5;
          const localVeg = coarseAnnualVeg;
          if (type === 0) { // conifer
            if (localMoist < 0.05 || localVeg < 0.15) continue;
          } else if (type === 1) { // broadleaf
            if (localMoist < 0.15 || localVeg < 0.3) continue;
          } else { // palm
            if (localMoist < 0.05 || localVeg < 0.08) continue;
          }

          const normal = latLonToNormal(lat, lon);
          const position = normal.clone().multiplyScalar(GLOBE_RADIUS);

          if (layerData.elevation) {
            const ed = layerData.elevation.data as Float32Array;
            const enl = layerData.elevation.shape[0];
            const eno = layerData.elevation.shape[1];
            const elev = sampleElevation(ed, enl, eno, lat, lon);
            position.addScaledVector(normal, elev * ELEVATION_SCALE);
          }

          const quaternion = new THREE.Quaternion();
          quaternion.setFromUnitVectors(Y_UP, normal);
          const twist = new THREE.Quaternion();
          twist.setFromAxisAngle(normal, rand() * Math.PI * 2);
          quaternion.premultiply(twist);

          let scaleXZ: number, scaleY: number;
          if (type === 0) {
            scaleXZ = 0.85 + rand() * 0.3;
            scaleY = 0.8 + rand() * 0.4;
          } else if (type === 1) {
            scaleXZ = 0.7 + rand() * 0.6;
            scaleY = 0.6 + rand() * 0.8;
          } else {
            scaleXZ = 0.8 + rand() * 0.3;
            scaleY = 0.5 + rand() * 1.0;
          }

          instances[type].push({ type, position, quaternion, scaleXZ, scaleY, lat, lon, cellI: i, cellJ: j });
        }
      }
    }

    // Build geometries: conifer (merged), broadleaf trunk, broadleaf foliage, palm (merged)
    const coniferGeom = buildConiferGeometry();
    const broadleafTrunkGeom = buildBroadleafTrunkGeometry();
    const broadleafFoliageGeom = buildBroadleafFoliageGeometry();
    const palmGeom = buildPalmGeometry();

    const material = new THREE.MeshLambertMaterial({ vertexColors: true });
    // Darken trees on the night side of the globe by modulating vColor.
    // We pass the sun direction as a uniform and compute
    // dot(globeSurfaceNormal, sunDir) to dim night-side trees.
    this.sunDirUniform = { value: new THREE.Vector3(1, 0, 0) };
    material.onBeforeCompile = (shader) => {
      shader.uniforms.sunDir = this.sunDirUniform;
      shader.vertexShader = 'uniform vec3 sunDir;\n' + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(
        '#include <color_vertex>',
        `
        #include <color_vertex>
        vec3 treeWorldPos = (modelMatrix * instanceMatrix * vec4(position, 1.0)).xyz;
        vec3 globeNormal = normalize(treeWorldPos);
        float surfaceDot = dot(globeNormal, sunDir);
        float surfaceLight = smoothstep(-0.05, 0.15, surfaceDot);
        vColor.rgb *= mix(0.08, 1.0, surfaceLight);
        `
      );
    };

    // --- Conifer mesh (type 0) ---
    const coniferList = instances[0];
    if (coniferList.length > 0) {
      const mesh = new THREE.InstancedMesh(coniferGeom, material, coniferList.length);
      const dummy = new THREE.Matrix4();
      const color = new THREE.Color();
      for (let k = 0; k < coniferList.length; k++) {
        const inst = coniferList[k];
        dummy.compose(inst.position, inst.quaternion, new THREE.Vector3(inst.scaleXZ, inst.scaleY, inst.scaleXZ));
        mesh.setMatrixAt(k, dummy);
        const variation = 0.85 + rand() * 0.30;
        color.setRGB(variation, variation, variation);
        mesh.setColorAt(k, color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.frustumCulled = false;

      this.meshes.push(mesh);
      this.group.add(mesh);
    }

    // --- Broadleaf: separate trunk + foliage meshes (type 1) ---
    const broadleafList = instances[1];
    if (broadleafList.length > 0) {
      const count = broadleafList.length;

      // Store per-instance data for seasonal updates
      this.broadleafOrigScaleXZ = new Float32Array(count);
      this.broadleafOrigScaleY = new Float32Array(count);
      this.broadleafBaseColors = new Float32Array(count);
      this.broadleafPositions = new Float32Array(count * 3);
      this.broadleafQuaternions = new Float32Array(count * 4);
      this.broadleafCellIndex = new Uint16Array(count);
      this.broadleafEvergreen = new Uint8Array(count);

      const trunkMesh = new THREE.InstancedMesh(broadleafTrunkGeom, material, count);
      const foliageMesh = new THREE.InstancedMesh(broadleafFoliageGeom, material, count);

      const dummy = new THREE.Matrix4();
      const color = new THREE.Color();

      // Pre-compute monthly curves per native cell
      const curveCache = new Map<number, {
        foliageCurve: Float32Array; autumnTintCurve: Float32Array;
        colorQuality: number; evergreen: boolean;
      }>();
      const cellKeyToIndex = new Map<number, number>();
      let nextCellIndex = 0;

      // First pass: collect unique cells
      for (let k = 0; k < count; k++) {
        const inst = broadleafList[k];
        const cellKey = inst.cellI * nativeNlon + inst.cellJ;
        if (!curveCache.has(cellKey)) {
          // Use T2m (air temp) for leaf shedding — sample at coarse cell center
          const t2mCellI = Math.floor((inst.cellI + 0.5) * t2mNlat / nativeNlat);
          const t2mCellJ = Math.floor((inst.cellJ + 0.5) * t2mNlon / nativeNlon);
          const curves = computeCellCurves(
            t2mData, t2mNlat, t2mNlon, t2mCellI, t2mCellJ,
            inst.lat, cloudData, cloudField?.shape[1] ?? 0, cloudField?.shape[2] ?? 0,
            inst.cellI, inst.cellJ,
          );
          curveCache.set(cellKey, curves);
          cellKeyToIndex.set(cellKey, nextCellIndex++);
        }
      }

      // Allocate curve storage
      const numCells = nextCellIndex;
      this.cellFoliageCurves = new Float32Array(numCells * 12);
      this.cellAutumnTintCurves = new Float32Array(numCells * 12);
      this.cellColorQuality = new Float32Array(numCells);

      // Fill curve arrays
      for (const [cellKey, curves] of curveCache) {
        const ci = cellKeyToIndex.get(cellKey)!;
        this.cellFoliageCurves.set(curves.foliageCurve, ci * 12);
        this.cellAutumnTintCurves.set(curves.autumnTintCurve, ci * 12);
        this.cellColorQuality[ci] = curves.colorQuality;
      }

      // Second pass: set up instances
      for (let k = 0; k < count; k++) {
        const inst = broadleafList[k];
        dummy.compose(inst.position, inst.quaternion, new THREE.Vector3(inst.scaleXZ, inst.scaleY, inst.scaleXZ));
        trunkMesh.setMatrixAt(k, dummy);
        foliageMesh.setMatrixAt(k, dummy);

        const variation = 0.85 + rand() * 0.30;
        color.setRGB(variation, variation, variation);
        trunkMesh.setColorAt(k, color);
        const greenVar = 0.85 + rand() * 0.3;
        color.setRGB(0.15 * greenVar, 0.55 * greenVar, 0.10 * greenVar);
        foliageMesh.setColorAt(k, color);

        this.broadleafOrigScaleXZ[k] = inst.scaleXZ;
        this.broadleafOrigScaleY[k] = inst.scaleY;
        this.broadleafBaseColors[k] = greenVar;
        this.broadleafPositions[k * 3] = inst.position.x;
        this.broadleafPositions[k * 3 + 1] = inst.position.y;
        this.broadleafPositions[k * 3 + 2] = inst.position.z;
        this.broadleafQuaternions[k * 4] = inst.quaternion.x;
        this.broadleafQuaternions[k * 4 + 1] = inst.quaternion.y;
        this.broadleafQuaternions[k * 4 + 2] = inst.quaternion.z;
        this.broadleafQuaternions[k * 4 + 3] = inst.quaternion.w;

        const cellKey = inst.cellI * nativeNlon + inst.cellJ;
        this.broadleafCellIndex[k] = cellKeyToIndex.get(cellKey)!;
        this.broadleafEvergreen[k] = curveCache.get(cellKey)!.evergreen ? 1 : 0;
      }

      trunkMesh.instanceMatrix.needsUpdate = true;
      foliageMesh.instanceMatrix.needsUpdate = true;
      if (trunkMesh.instanceColor) trunkMesh.instanceColor.needsUpdate = true;
      if (foliageMesh.instanceColor) foliageMesh.instanceColor.needsUpdate = true;
      trunkMesh.frustumCulled = false;

      foliageMesh.frustumCulled = false;


      this.broadleafFoliageMesh = foliageMesh;
      this.meshes.push(trunkMesh);
      this.meshes.push(foliageMesh);
      this.group.add(trunkMesh);
      this.group.add(foliageMesh);
    }

    // --- Palm mesh (type 2) ---
    const palmList = instances[2];
    if (palmList.length > 0) {
      const mesh = new THREE.InstancedMesh(palmGeom, material, palmList.length);
      const dummy = new THREE.Matrix4();
      const color = new THREE.Color();
      for (let k = 0; k < palmList.length; k++) {
        const inst = palmList[k];
        dummy.compose(inst.position, inst.quaternion, new THREE.Vector3(inst.scaleXZ, inst.scaleY, inst.scaleXZ));
        mesh.setMatrixAt(k, dummy);
        const variation = 0.85 + rand() * 0.30;
        color.setRGB(variation, variation, variation);
        mesh.setColorAt(k, color);
      }
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
      mesh.frustumCulled = false;

      this.meshes.push(mesh);
      this.group.add(mesh);
    }

    // Clean up base geometries
    coniferGeom.dispose();
    broadleafTrunkGeom.dispose();
    broadleafFoliageGeom.dispose();
    palmGeom.dispose();
  }

  /**
   * Update broadleaf foliage by interpolating pre-computed monthly curves.
   * No phase boundaries — just smooth lerp between adjacent month samples.
   */
  setMonth(monthProgress: number, _layerData: ClimateLayerData): void {
    const foliage = this.broadleafFoliageMesh;
    if (!foliage || foliage.count === 0) return;

    if (Math.abs(monthProgress - this.lastMonthProgress) < 0.01) return;
    this.lastMonthProgress = monthProgress;

    const dummy = new THREE.Matrix4();
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scaleVec = new THREE.Vector3();
    const color = new THREE.Color();
    const count = foliage.count;

    // Interpolation indices
    const mp = ((monthProgress % 12) + 12) % 12;
    const m0 = Math.floor(mp) % 12;
    const m1 = (m0 + 1) % 12;
    const frac = mp - Math.floor(mp);

    for (let k = 0; k < count; k++) {
      const v = this.broadleafBaseColors[k];

      // Evergreen: always green, full foliage
      if (this.broadleafEvergreen[k]) {
        color.setRGB(0.15 * v, 0.55 * v, 0.10 * v);
        foliage.setColorAt(k, color);
        continue;
      }

      // Look up pre-computed curves for this cell
      const ci = this.broadleafCellIndex[k];
      const base = ci * 12;

      // Lerp foliage scale between months
      const fs0 = this.cellFoliageCurves[base + m0];
      const fs1 = this.cellFoliageCurves[base + m1];
      const foliageScale = fs0 + (fs1 - fs0) * frac;

      // Lerp autumn tint between months
      const at0 = this.cellAutumnTintCurves[base + m0];
      const at1 = this.cellAutumnTintCurves[base + m1];
      const autumnTint = at0 + (at1 - at0) * frac;

      const cq = this.cellColorQuality[ci];

      // Color: blend green and autumn based on tint
      const greenR = 0.15 * v, greenG = 0.55 * v, greenB = 0.10 * v;
      const vividR = 0.70 * v, vividG = 0.25 * v, vividB = 0.05 * v;
      const dullR = 0.45 * v, dullG = 0.35 * v, dullB = 0.12 * v;
      const autR = dullR + (vividR - dullR) * cq;
      const autG = dullG + (vividG - dullG) * cq;
      const autB = dullB + (vividB - dullB) * cq;

      const cr = greenR * (1 - autumnTint) + autR * autumnTint;
      const cg = greenG * (1 - autumnTint) + autG * autumnTint;
      const cb = greenB * (1 - autumnTint) + autB * autumnTint;

      color.setRGB(cr, cg, cb);
      foliage.setColorAt(k, color);

      // Scale
      const origSXZ = this.broadleafOrigScaleXZ[k];
      const origSY = this.broadleafOrigScaleY[k];

      pos.set(
        this.broadleafPositions[k * 3],
        this.broadleafPositions[k * 3 + 1],
        this.broadleafPositions[k * 3 + 2],
      );
      quat.set(
        this.broadleafQuaternions[k * 4],
        this.broadleafQuaternions[k * 4 + 1],
        this.broadleafQuaternions[k * 4 + 2],
        this.broadleafQuaternions[k * 4 + 3],
      );
      scaleVec.set(
        origSXZ * foliageScale,
        origSY * foliageScale,
        origSXZ * foliageScale,
      );
      dummy.compose(pos, quat, scaleVec);
      foliage.setMatrixAt(k, dummy);
    }

    foliage.instanceMatrix.needsUpdate = true;
    if (foliage.instanceColor) foliage.instanceColor.needsUpdate = true;
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.sunDirUniform.value.copy(dir);
  }

  getObject(): THREE.Object3D {
    return this.group;
  }

  dispose(): void {
    for (const mesh of this.meshes) {
      mesh.geometry.dispose();
      if (mesh.material instanceof THREE.Material) {
        mesh.material.dispose();
      }
    }
    this.meshes = [];
    this.broadleafFoliageMesh = null;
  }
}
