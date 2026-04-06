import * as THREE from 'three';
import type { ClimateLayerData } from './loadBinaryData';

const DEG2RAD = Math.PI / 180;
const CLOUD_RADIUS = 1.015;
const SEED = 7919;

const MIN_FRACTION = 0.05;
const MAX_CLOUDS = 800;
const NUM_STYLES = 4; // texture atlas columns

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function latLonToXYZ(lat: number, lon: number, radius: number): [number, number, number] {
  const phi = (90 - lat) * DEG2RAD;
  const theta = lon * DEG2RAD;
  return [
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  ];
}

// ---- Texture atlas: 4 styles packed horizontally ----

type CloudStyle = 'smallPuff' | 'bigCumulus' | 'towering' | 'cluster';

function createCloudAtlas(): THREE.Texture {
  const tileSize = 256;
  const canvas = document.createElement('canvas');
  canvas.width = tileSize * NUM_STYLES;
  canvas.height = tileSize;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const styles: CloudStyle[] = ['smallPuff', 'bigCumulus', 'towering', 'cluster'];

  for (let s = 0; s < styles.length; s++) {
    const ox = s * tileSize;
    const cx = ox + tileSize / 2;
    const cy = tileSize / 2;
    const rand = mulberry32(10000 + s * 777);

    const blobs: { x: number; y: number; r: number; a: number }[] = [];

    if (styles[s] === 'smallPuff') {
      blobs.push(
        { x: cx, y: cy, r: 38, a: 0.4 },
        { x: cx - 12, y: cy + 4, r: 30, a: 0.3 },
        { x: cx + 14, y: cy + 2, r: 32, a: 0.3 },
        { x: cx, y: cy - 14, r: 26, a: 0.28 },
      );
      for (let i = 0; i < 6; i++) {
        blobs.push({
          x: cx + (rand() - 0.5) * 50,
          y: cy + (rand() - 0.5) * 40,
          r: 10 + rand() * 16,
          a: 0.12 + rand() * 0.12,
        });
      }
    } else if (styles[s] === 'bigCumulus') {
      blobs.push(
        { x: cx, y: cy + 5, r: 52, a: 0.38 },
        { x: cx - 24, y: cy + 8, r: 44, a: 0.32 },
        { x: cx + 26, y: cy + 6, r: 46, a: 0.32 },
        { x: cx - 8, y: cy - 18, r: 34, a: 0.28 },
        { x: cx + 10, y: cy - 22, r: 30, a: 0.26 },
        { x: cx, y: cy - 32, r: 24, a: 0.22 },
        { x: cx - 40, y: cy + 10, r: 32, a: 0.22 },
        { x: cx + 42, y: cy + 8, r: 30, a: 0.22 },
      );
      for (let i = 0; i < 14; i++) {
        blobs.push({
          x: cx + (rand() - 0.5) * 80,
          y: cy + (rand() - 0.5) * 60 - 5,
          r: 14 + rand() * 22,
          a: 0.1 + rand() * 0.15,
        });
      }
    } else if (styles[s] === 'towering') {
      blobs.push(
        { x: cx, y: cy + 10, r: 40, a: 0.36 },
        { x: cx - 14, y: cy - 8, r: 36, a: 0.32 },
        { x: cx + 12, y: cy - 12, r: 34, a: 0.32 },
        { x: cx, y: cy - 30, r: 30, a: 0.28 },
        { x: cx - 8, y: cy - 44, r: 24, a: 0.24 },
        { x: cx + 6, y: cy - 50, r: 20, a: 0.2 },
      );
      for (let i = 0; i < 8; i++) {
        blobs.push({
          x: cx + (rand() - 0.5) * 50,
          y: cy + (rand() - 0.5) * 80 - 10,
          r: 10 + rand() * 18,
          a: 0.1 + rand() * 0.14,
        });
      }
    } else {
      // cluster
      for (let i = 0; i < 5; i++) {
        const bx = (rand() - 0.5) * 80;
        const by = (rand() - 0.5) * 60;
        blobs.push({ x: cx + bx, y: cy + by, r: 20 + rand() * 16, a: 0.3 + rand() * 0.1 });
        for (let k = 0; k < 3; k++) {
          blobs.push({
            x: cx + bx + (rand() - 0.5) * 30,
            y: cy + by + (rand() - 0.5) * 24,
            r: 8 + rand() * 12,
            a: 0.12 + rand() * 0.1,
          });
        }
      }
    }

    for (const b of blobs) {
      const grad = ctx.createRadialGradient(b.x, b.y, 0, b.x, b.y, b.r);
      grad.addColorStop(0, `rgba(255,255,255,${b.a})`);
      grad.addColorStop(0.3, `rgba(255,255,255,${b.a * 0.8})`);
      grad.addColorStop(0.6, `rgba(248,250,255,${b.a * 0.4})`);
      grad.addColorStop(0.85, `rgba(240,244,255,${b.a * 0.1})`);
      grad.addColorStop(1, 'rgba(235,240,250,0)');
      ctx.fillStyle = grad;
      // Clip to this tile
      ctx.save();
      ctx.beginPath();
      ctx.rect(ox, 0, tileSize, tileSize);
      ctx.clip();
      ctx.fillRect(ox, 0, tileSize, tileSize);
      ctx.restore();
    }

    // Bottom shadow per tile
    ctx.save();
    ctx.beginPath();
    ctx.rect(ox, 0, tileSize, tileSize);
    ctx.clip();
    const bellyGrad = ctx.createLinearGradient(0, 0, 0, tileSize);
    bellyGrad.addColorStop(0, 'rgba(0,0,0,0)');
    bellyGrad.addColorStop(0.6, 'rgba(0,0,0,0)');
    bellyGrad.addColorStop(1, 'rgba(30,40,60,0.08)');
    ctx.globalCompositeOperation = 'source-atop';
    ctx.fillStyle = bellyGrad;
    ctx.fillRect(ox, 0, tileSize, tileSize);
    ctx.restore();
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

// ---- Cloud data (SoA layout for cache-friendly updates) ----

interface CloudData {
  count: number;
  baseLat: Float32Array;
  baseLon: Float32Array;
  driftLat: Float32Array;
  driftLon: Float32Array;
  windU: Float32Array;
  windV: Float32Array;
  radius: Float32Array;
  cellI: Uint16Array;
  cellJ: Uint16Array;
  currentOpacity: Float32Array;
  targetOpacity: Float32Array;
}

export class CloudInstances {
  private group: THREE.Group;
  private data: CloudData | null = null;
  private points: THREE.Points | null = null;
  private sunDir: THREE.Vector3 = new THREE.Vector3(1, 0, 0);
  private nlat: number = 0;
  private nlon: number = 0;
  private lastMonthProgress: number = -1;
  private atlas: THREE.Texture | null = null;
  private windUData: Float32Array | null = null;
  private windVData: Float32Array | null = null;
  private monthFrac: number = 0;
  private m0: number = 0;
  private m1: number = 0;

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
    const cfData = totalField.data as Float32Array;

    // Compute annual means per cell
    const annualMean = new Float32Array(nlat * nlon);
    const annualWindU = new Float32Array(nlat * nlon);
    const annualWindV = new Float32Array(nlat * nlon);
    const windUData = layerData.wind_u_10m?.data as Float32Array | undefined;
    const windVData = layerData.wind_v_10m?.data as Float32Array | undefined;
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nlon; j++) {
        let csum = 0, usum = 0, vsum = 0;
        for (let m = 0; m < 12; m++) {
          const idx = m * nlat * nlon + i * nlon + j;
          csum += cfData[idx];
          if (windUData) usum += windUData[idx];
          if (windVData) vsum += windVData[idx];
        }
        const ci = i * nlon + j;
        annualMean[ci] = csum / 12;
        annualWindU[ci] = usum / 12;
        annualWindV[ci] = vsum / 12;
      }
    }

    // Create texture atlas
    this.atlas = createCloudAtlas();

    const rand = mulberry32(SEED);

    // Collect and shuffle candidate cells
    const cells: { i: number; j: number; mean: number }[] = [];
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nlon; j++) {
        const mean = annualMean[i * nlon + j];
        if (mean >= MIN_FRACTION) {
          cells.push({ i, j, mean });
        }
      }
    }
    for (let n = cells.length - 1; n > 0; n--) {
      const swap = Math.floor(rand() * (n + 1));
      [cells[n], cells[swap]] = [cells[swap], cells[n]];
    }

    // First pass: collect cloud params into temp arrays
    const tmpLat: number[] = [];
    const tmpLon: number[] = [];
    const tmpRadius: number[] = [];
    const tmpCellI: number[] = [];
    const tmpCellJ: number[] = [];
    const tmpWindU: number[] = [];
    const tmpWindV: number[] = [];
    const tmpSize: number[] = [];
    const tmpStyle: number[] = [];
    const tmpAspect: number[] = []; // scaleX / scaleY ratio

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
        const r = CLOUD_RADIUS + rand() * 0.01;

        const styleRoll = rand();
        let styleIdx: number;
        if (styleRoll < 0.35) styleIdx = 0;       // smallPuff
        else if (styleRoll < 0.65) styleIdx = 1;   // bigCumulus
        else if (styleRoll < 0.85) styleIdx = 2;   // towering
        else styleIdx = 3;                          // cluster

        const baseSize = 0.06 + mean * 0.06 + rand() * 0.03;
        const aspectJitter = 0.85 + rand() * 0.3;
        let scaleX = baseSize * aspectJitter * 1.3;
        let scaleY = baseSize / aspectJitter * 0.9;
        if (styleIdx === 2) { scaleX *= 0.8; scaleY *= 1.3; }
        if (styleIdx === 3) { scaleX *= 1.2; scaleY *= 1.1; }

        const cosLat = Math.cos(lat * DEG2RAD);
        const ci = i * nlon + j;
        const windScale = 5000.0;
        const wU = (annualWindU[ci] / (111000 * Math.max(cosLat, 0.15))) * windScale;
        const wV = (annualWindV[ci] / 111000) * windScale;

        tmpLat.push(lat);
        tmpLon.push(lon);
        tmpRadius.push(r);
        tmpCellI.push(i);
        tmpCellJ.push(j);
        tmpWindU.push(wU);
        tmpWindV.push(wV);
        tmpSize.push(Math.max(scaleX, scaleY)); // point size = max dimension
        tmpStyle.push(styleIdx);
        tmpAspect.push(scaleX / scaleY); // >1 = wider, <1 = taller
        placed++;
      }
    }

    const N = placed;
    if (N === 0) return;

    // Allocate SoA
    const data: CloudData = {
      count: N,
      baseLat: new Float32Array(tmpLat),
      baseLon: new Float32Array(tmpLon),
      driftLat: new Float32Array(N),
      driftLon: new Float32Array(N),
      windU: new Float32Array(tmpWindU),
      windV: new Float32Array(tmpWindV),
      radius: new Float32Array(tmpRadius),
      cellI: new Uint16Array(tmpCellI),
      cellJ: new Uint16Array(tmpCellJ),
      currentOpacity: new Float32Array(N),
      targetOpacity: new Float32Array(N),
    };
    this.data = data;

    // Build geometry
    const positions = new Float32Array(N * 3);
    const sizes = new Float32Array(tmpSize);
    const styles = new Float32Array(tmpStyle);
    const opacities = new Float32Array(N); // starts at 0
    const aspects = new Float32Array(tmpAspect);

    for (let idx = 0; idx < N; idx++) {
      const [x, y, z] = latLonToXYZ(data.baseLat[idx], data.baseLon[idx], data.radius[idx]);
      positions[idx * 3] = x;
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = z;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    geom.setAttribute('aStyle', new THREE.BufferAttribute(styles, 1));
    geom.setAttribute('aOpacity', new THREE.BufferAttribute(opacities, 1));
    geom.setAttribute('aAspect', new THREE.BufferAttribute(aspects, 1));

    const mat = new THREE.ShaderMaterial({
      uniforms: {
        atlas: { value: this.atlas },
        sunDir: { value: this.sunDir },
        viewportHeight: { value: 1.0 },
      },
      vertexShader: `
        attribute float aSize;
        attribute float aStyle;
        attribute float aOpacity;
        attribute float aAspect;
        varying float vOpacity;
        varying float vStyle;
        varying float vAspect;
        varying float vBrightness;
        uniform vec3 sunDir;
        uniform float viewportHeight;
        void main() {
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mvPos;
          // Size in pixels: scale by distance
          gl_PointSize = aSize * viewportHeight / -mvPos.z;
          vOpacity = aOpacity;
          vStyle = aStyle;
          vAspect = aAspect;
          // Day/night brightness — transform to world space to match sunDir
          vec3 worldNormal = normalize((modelMatrix * vec4(position, 0.0)).xyz);
          float surfaceDot = dot(worldNormal, sunDir);
          vBrightness = 0.12 + 0.88 * clamp((surfaceDot + 0.05) / 0.2, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D atlas;
        varying float vOpacity;
        varying float vStyle;
        varying float vAspect;
        varying float vBrightness;
        void main() {
          if (vOpacity < 0.005) discard;
          vec2 uv = gl_PointCoord;
          // Apply aspect ratio: stretch UV to sample correctly
          // For wide clouds (aspect>1): compress x sampling
          // For tall clouds (aspect<1): compress y sampling
          if (vAspect > 1.0) {
            uv.y = 0.5 + (uv.y - 0.5) * vAspect;
          } else {
            uv.x = 0.5 + (uv.x - 0.5) / vAspect;
          }
          // Discard if UV out of 0-1 range (outside the cloud shape)
          if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) discard;
          // Map to atlas tile
          float styleIdx = floor(vStyle + 0.5);
          float atlasU = (styleIdx + uv.x) / ${NUM_STYLES}.0;
          vec4 texColor = texture2D(atlas, vec2(atlasU, uv.y));
          if (texColor.a < 0.01) discard;
          vec3 color = texColor.rgb * vBrightness;
          gl_FragColor = vec4(color, texColor.a * vOpacity);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });

    this.points = new THREE.Points(geom, mat);
    this.points.renderOrder = 10;
    this.points.frustumCulled = false;
    this.group.add(this.points);
  }

  setMonth(monthProgress: number, layerData: ClimateLayerData): void {
    if (!this.data) return;
    if (Math.abs(monthProgress - this.lastMonthProgress) < 0.005) return;
    this.lastMonthProgress = monthProgress;

    const mp = ((monthProgress % 12) + 12) % 12;
    this.m0 = Math.floor(mp) % 12;
    this.m1 = (this.m0 + 1) % 12;
    this.monthFrac = mp - Math.floor(mp);

    const nlat = this.nlat;
    const nlon = this.nlon;

    const totalField = layerData.cloud_fraction;
    if (!totalField) return;
    const cfData = totalField.data as Float32Array;

    this.windUData = layerData.wind_u_10m?.data as Float32Array ?? null;
    this.windVData = layerData.wind_v_10m?.data as Float32Array ?? null;

    const d = this.data;
    for (let idx = 0; idx < d.count; idx++) {
      const v0 = cfData[this.m0 * nlat * nlon + d.cellI[idx] * nlon + d.cellJ[idx]];
      const v1 = cfData[this.m1 * nlat * nlon + d.cellI[idx] * nlon + d.cellJ[idx]];
      const fraction = Math.max(0, Math.min(1, v0 + (v1 - v0) * this.monthFrac));
      d.targetOpacity[idx] = Math.min(1, fraction * 1.3);
    }
  }

  update(dt: number): void {
    if (!this.data || !this.points) return;

    const d = this.data;
    const geom = this.points.geometry;
    const posAttr = geom.getAttribute('position') as THREE.BufferAttribute;
    const opAttr = geom.getAttribute('aOpacity') as THREE.BufferAttribute;
    const positions = posAttr.array as Float32Array;
    const opacities = opAttr.array as Float32Array;

    const nlat = this.nlat;
    const nlon = this.nlon;
    const latStep = 180 / nlat;
    const lonStep = 360 / nlon;
    const hasWind = this.windUData !== null && this.windVData !== null;

    for (let idx = 0; idx < d.count; idx++) {
      // Smooth opacity
      const diff = d.targetOpacity[idx] - d.currentOpacity[idx];
      if (Math.abs(diff) > 0.001) {
        d.currentOpacity[idx] += Math.sign(diff) * Math.min(Math.abs(diff), 0.08 * dt);
      }
      opacities[idx] = d.currentOpacity[idx];

      if (d.currentOpacity[idx] < 0.005) continue;

      // Re-sample wind at drifted position
      if (hasWind) {
        const curLat = Math.max(-89, Math.min(89, d.baseLat[idx] + d.driftLat[idx]));
        const curLon = ((d.baseLon[idx] + d.driftLon[idx]) % 360 + 360) % 360;
        const ci = Math.min(nlat - 1, Math.max(0, Math.floor((curLat + 90) / latStep)));
        const cj = Math.min(nlon - 1, Math.floor(curLon / lonStep));
        const i0 = this.m0 * nlat * nlon + ci * nlon + cj;
        const i1 = this.m1 * nlat * nlon + ci * nlon + cj;
        const u = this.windUData![i0] + (this.windUData![i1] - this.windUData![i0]) * this.monthFrac;
        const v = this.windVData![i0] + (this.windVData![i1] - this.windVData![i0]) * this.monthFrac;
        const cosLat = Math.cos(curLat * DEG2RAD);
        d.windU[idx] = (u / (111000 * Math.max(cosLat, 0.15))) * 5000.0;
        d.windV[idx] = (v / 111000) * 5000.0;
      }

      // Drift
      d.driftLon[idx] += d.windU[idx] * dt;
      d.driftLat[idx] += d.windV[idx] * dt;

      const lon = ((d.baseLon[idx] + d.driftLon[idx]) % 360 + 360) % 360;
      const lat = Math.max(-89, Math.min(89, d.baseLat[idx] + d.driftLat[idx]));
      const [x, y, z] = latLonToXYZ(lat, lon, d.radius[idx]);
      positions[idx * 3] = x;
      positions[idx * 3 + 1] = y;
      positions[idx * 3 + 2] = z;
    }

    posAttr.needsUpdate = true;
    opAttr.needsUpdate = true;
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.sunDir.copy(dir);
    if (this.points) {
      (this.points.material as THREE.ShaderMaterial).uniforms.sunDir.value.copy(dir);
    }
  }

  getObject(): THREE.Object3D {
    return this.group;
  }

  /** Call when viewport size or camera FOV changes for correct point sizing. */
  setViewportHeight(h: number, fovDeg: number = 45): void {
    if (this.points) {
      // projectionScale converts world-space size to pixel size at unit distance
      const projScale = h / (2 * Math.tan((fovDeg * DEG2RAD) / 2));
      (this.points.material as THREE.ShaderMaterial).uniforms.viewportHeight.value = projScale;
    }
  }

  dispose(): void {
    if (this.points) {
      this.points.geometry.dispose();
      (this.points.material as THREE.ShaderMaterial).dispose();
    }
    if (this.atlas) this.atlas.dispose();
    this.data = null;
    this.atlas = null;
  }
}
