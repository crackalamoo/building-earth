import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { ClimateLayerData } from './loadBinaryData';
import { ELEVATION_SCALE, sampleElevation } from './elevation';

const GLOBE_RADIUS = 1.001;
const TREE_HEIGHT = 0.017;
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

function latLonToNormal(lat: number, lon: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = lon * (Math.PI / 180);
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
  scaleXZ: number; // width/girth
  scaleY: number;  // height
}

/** Convert to non-indexed, keep only position+normal+color for merge compatibility. */
function prepareForMerge(
  geom: THREE.BufferGeometry,
  r: number, g: number, b: number,
): THREE.BufferGeometry {
  // Expand index buffer into flat vertex list
  const ni = geom.toNonIndexed();
  geom.dispose();
  // Strip uv (cylinders/cones have it, icosahedrons don't)
  ni.deleteAttribute('uv');
  // Ensure normals exist
  ni.computeVertexNormals();
  // Bake uniform vertex color
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

function buildBroadleafGeometry(): THREE.BufferGeometry {
  const h = TREE_HEIGHT;

  const trunk = prepareForMerge(
    new THREE.CylinderGeometry(h * 0.05, h * 0.07, h * 0.4, 5, 1),
    0.45, 0.30, 0.15,
  );
  trunk.translate(0, h * 0.2, 0);

  const foliage = prepareForMerge(
    new THREE.IcosahedronGeometry(h * 0.30, 1),
    0.15, 0.55, 0.10,
  );
  foliage.translate(0, h * 0.4 + h * 0.25, 0);

  const merged = mergeGeometries([trunk, foliage], false);
  trunk.dispose();
  foliage.dispose();
  return merged!;
}

function buildPalmGeometry(): THREE.BufferGeometry {
  // Palm: tall thin trunk with radiating frond spikes
  const h = TREE_HEIGHT * 1.8; // taller than other trees
  const crownY = h * 0.78;

  const parts: THREE.BufferGeometry[] = [];

  // Thin trunk
  const trunk = prepareForMerge(
    new THREE.CylinderGeometry(h * 0.02, h * 0.045, h * 0.76, 4, 1),
    0.55, 0.40, 0.22,
  );
  trunk.translate(0, h * 0.38, 0);
  parts.push(trunk);

  // Center hub — balances body with spiky fronds
  const hub = prepareForMerge(
    new THREE.SphereGeometry(h * 0.12, 5, 4),
    0.08, 0.50, 0.08,
  );
  hub.translate(0, crownY, 0);
  parts.push(hub);

  // 5 fronds as spikes pointing outward and drooping
  const numFronds = 5;
  for (let i = 0; i < numFronds; i++) {
    const angle = (i / numFronds) * Math.PI * 2;

    // Tetrahedron stretched into spike
    const frond = prepareForMerge(
      new THREE.TetrahedronGeometry(h * 0.17, 0),
      0.06, 0.52, 0.06,
    );
    // Stretch into spike
    frond.scale(0.5, 0.5, 2.2);
    // Tilt to point outward and slightly down
    frond.rotateX(0.35);
    // Rotate around Y to position around the crown
    frond.rotateY(angle);
    // Move to crown position
    frond.translate(0, crownY, 0);

    parts.push(frond);
  }

  const merged = mergeGeometries(parts, false);
  for (const part of parts) part.dispose();
  return merged!;
}

function computeTypeProbabilities(
  annualTemp: number,
  annualSoilMoisture: number,
): [number, number, number] {
  // Palm: tropical, needs warmth (>20°C) and moisture
  let palm = 0;
  if (annualTemp > 20) {
    palm = ((annualTemp - 20) / 10) * (0.2 + 0.8 * annualSoilMoisture);
  }

  // Conifer: dominates boreal/cold zones, strong below 10°C, fades by 18°C
  let conifer = 0;
  if (annualTemp < 18) {
    const t = Math.max(0, Math.min(1, (18 - annualTemp) / 18));
    // Quadratic ramp — heavy dominance when cold
    conifer = t * t * 2.0;
  }

  // Broadleaf: temperate sweet spot, peaks 12–22°C, needs some warmth and moisture
  let broadleaf = 0;
  if (annualTemp > 2) {
    // Ramps up from 2°C, peaks around 18°C
    const warmth = Math.min((annualTemp - 2) / 16, 1);
    // Falls off in very hot (>25°C) tropical zones where palm takes over
    const heatFade = annualTemp > 25 ? Math.max(0, 1 - (annualTemp - 25) / 10) : 1;
    broadleaf = warmth * heatFade * (0.4 + 0.6 * annualSoilMoisture);
  }

  // Normalize
  const total = palm + conifer + broadleaf;
  if (total < 1e-6) return [0, 1, 0]; // fallback broadleaf
  return [conifer / total, broadleaf / total, palm / total];
}

export class TreeInstances {
  private group: THREE.Group;
  private meshes: THREE.InstancedMesh[] = [];

  constructor(layerData: ClimateLayerData) {
    this.group = new THREE.Group();

    const rand = mulberry32(SEED);

    const surfaceData = layerData.surface.data as Float32Array;
    const vegData = layerData.vegetation_fraction!.data as Float32Array;
    const soilData = layerData.soil_moisture!.data as Float32Array;
    const landMaskNative = layerData.land_mask_native!.data as Uint8Array;
    const landMaskHiRes = layerData.land_mask.data as Uint8Array;

    const nativeNlat = layerData.surface.shape[1]; // 36
    const nativeNlon = layerData.surface.shape[2]; // 72
    const hiNlat = layerData.land_mask.shape[0]; // 720
    const hiNlon = layerData.land_mask.shape[1]; // 1440

    const latStep = 180 / nativeNlat;
    const lonStep = 360 / nativeNlon;

    // Collect instances per type
    const instances: [TreeInstance[], TreeInstance[], TreeInstance[]] = [[], [], []];

    for (let i = 0; i < nativeNlat; i++) {
      for (let j = 0; j < nativeNlon; j++) {
        // Skip ocean cells
        if (landMaskNative[i * nativeNlon + j] === 0) continue;

        // Compute annual averages
        let tempSum = 0, vegSum = 0, soilSum = 0;
        for (let m = 0; m < 12; m++) {
          const offset = m * nativeNlat * nativeNlon + i * nativeNlon + j;
          tempSum += surfaceData[offset];
          vegSum += vegData[offset];
          soilSum += soilData[offset];
        }
        const annualTemp = tempSum / 12;
        const annualVeg = vegSum / 12;
        const annualSoil = Math.min(soilSum / 12, 1);

        // Skip low vegetation
        if (annualVeg < 0.05) continue;

        // Tree count per cell (stochastic rounding)
        const treeCount = Math.floor(annualVeg * 15 + rand());
        if (treeCount <= 0) continue;

        const [pConifer, pBroadleaf, pPalm] = computeTypeProbabilities(annualTemp, annualSoil);

        // Cell lat/lon bounds (native grid: i=0 is south pole row)
        const cellLatSouth = -90 + i * latStep;
        const cellLatNorth = cellLatSouth + latStep;
        const cellLonWest = j * lonStep;

        for (let t = 0; t < treeCount; t++) {
          // Random position within cell
          let lat: number, lon: number;
          let valid = false;

          for (let attempt = 0; attempt < 3; attempt++) {
            lat = cellLatSouth + rand() * latStep;
            lon = cellLonWest + rand() * lonStep;
            if (lon >= 360) lon -= 360;

            // Validate against hi-res land mask (data row 0 = south pole)
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

          // Sample tree type
          const r = rand();
          let type: 0 | 1 | 2;
          if (r < pConifer) type = 0;
          else if (r < pConifer + pBroadleaf) type = 1;
          else type = 2;

          const normal = latLonToNormal(lat!, lon!);
          const position = normal.clone().multiplyScalar(GLOBE_RADIUS);

          // Offset tree to match terrain displacement
          if (layerData.elevation) {
            const ed = layerData.elevation.data as Float32Array;
            const enl = layerData.elevation.shape[0];
            const eno = layerData.elevation.shape[1];
            const elev = sampleElevation(ed, enl, eno, lat!, lon!);
            position.addScaledVector(normal, elev * ELEVATION_SCALE);
          }

          // Orient: Y_UP → surface normal, plus random twist
          const quaternion = new THREE.Quaternion();
          quaternion.setFromUnitVectors(Y_UP, normal);
          const twist = new THREE.Quaternion();
          twist.setFromAxisAngle(normal, rand() * Math.PI * 2);
          quaternion.premultiply(twist);

          // Per-type variance: conifers uniform, broadleaf moderate, palms tall-variable
          let scaleXZ: number, scaleY: number;
          if (type === 0) {
            // Conifer: narrow, uniform height (dense even-aged stands)
            scaleXZ = 0.85 + rand() * 0.3;
            scaleY = 0.8 + rand() * 0.4;
          } else if (type === 1) {
            // Broadleaf: wide canopy variance, moderate height variance
            scaleXZ = 0.7 + rand() * 0.6;
            scaleY = 0.6 + rand() * 0.8;
          } else {
            // Palm: consistently thin, high height variance
            scaleXZ = 0.8 + rand() * 0.3;
            scaleY = 0.5 + rand() * 1.0;
          }

          instances[type].push({ type, position, quaternion, scaleXZ, scaleY });
        }
      }
    }

    // Build geometries
    const geometries = [
      buildConiferGeometry(),
      buildBroadleafGeometry(),
      buildPalmGeometry(),
    ];

    const material = new THREE.MeshLambertMaterial({ vertexColors: true });

    for (let type = 0; type < 3; type++) {
      const list = instances[type];
      if (list.length === 0) continue;

      const mesh = new THREE.InstancedMesh(geometries[type], material, list.length);
      const dummy = new THREE.Matrix4();
      const color = new THREE.Color();

      for (let k = 0; k < list.length; k++) {
        const inst = list[k];
        dummy.compose(inst.position, inst.quaternion, new THREE.Vector3(inst.scaleXZ, inst.scaleY, inst.scaleXZ));
        mesh.setMatrixAt(k, dummy);

        // Slight color variation
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
    for (const geom of geometries) {
      geom.dispose();
    }
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
  }
}
