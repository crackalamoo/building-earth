import * as THREE from 'three';
import type { ClimateLayerData } from './loadBinaryData';

const DEG2RAD = Math.PI / 180;
const CLOUD_RADIUS = 1.015;
const SEED = 7919;

const MIN_FRACTION = 0.05;
const MAX_CLOUDS = 800;

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function latLonToPosition(lat: number, lon: number, radius: number): THREE.Vector3 {
  const phi = (90 - lat) * DEG2RAD;
  const theta = lon * DEG2RAD;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

// ---- Multiple cloud texture variants ----

type CloudStyle = 'smallPuff' | 'bigCumulus' | 'elongated' | 'towering' | 'cluster';

function createCloudTexture(style: CloudStyle, seed: number): THREE.Texture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, size, size);

  const cx = size / 2;
  const cy = size / 2;
  const rand = mulberry32(seed);

  const blobs: { x: number; y: number; r: number; a: number }[] = [];

  if (style === 'smallPuff') {
    // Compact round puff
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
  } else if (style === 'bigCumulus') {
    // Large puffy cumulus with prominent top bumps
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
  } else if (style === 'elongated') {
    // Wide stratus shelf — still puffy but spread horizontally
    blobs.push(
      { x: cx, y: cy, r: 46, a: 0.35 },
      { x: cx - 30, y: cy + 4, r: 40, a: 0.3 },
      { x: cx + 32, y: cy + 2, r: 42, a: 0.3 },
      { x: cx - 52, y: cy + 6, r: 34, a: 0.24 },
      { x: cx + 54, y: cy + 4, r: 32, a: 0.24 },
      { x: cx - 10, y: cy - 12, r: 30, a: 0.25 },
      { x: cx + 14, y: cy - 14, r: 28, a: 0.24 },
    );
    for (let i = 0; i < 10; i++) {
      blobs.push({
        x: cx + (rand() - 0.5) * 100,
        y: cy + (rand() - 0.5) * 50,
        r: 14 + rand() * 20,
        a: 0.1 + rand() * 0.14,
      });
    }
  } else if (style === 'towering') {
    // Tall cumulonimbus-like
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
    // cluster: scattered small puffs
    for (let i = 0; i < 5; i++) {
      const ox = (rand() - 0.5) * 80;
      const oy = (rand() - 0.5) * 60;
      blobs.push({ x: cx + ox, y: cy + oy, r: 20 + rand() * 16, a: 0.3 + rand() * 0.1 });
      for (let k = 0; k < 3; k++) {
        blobs.push({
          x: cx + ox + (rand() - 0.5) * 30,
          y: cy + oy + (rand() - 0.5) * 24,
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
    ctx.fillRect(0, 0, size, size);
  }

  // Bottom shadow
  const bellyGrad = ctx.createLinearGradient(0, 0, 0, size);
  bellyGrad.addColorStop(0, 'rgba(0,0,0,0)');
  bellyGrad.addColorStop(0.6, 'rgba(0,0,0,0)');
  bellyGrad.addColorStop(1, 'rgba(30,40,60,0.08)');
  ctx.globalCompositeOperation = 'source-atop';
  ctx.fillStyle = bellyGrad;
  ctx.fillRect(0, 0, size, size);

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

// ---- Cloud placement ----

interface CloudInfo {
  sprite: THREE.Sprite;
  baseLat: number;
  baseLon: number;
  driftLat: number;
  driftLon: number;
  windU: number;
  windV: number;
  radius: number;
  cellI: number;
  cellJ: number;
  currentOpacity: number;
  targetOpacity: number;
}

export class CloudInstances {
  private group: THREE.Group;
  private clouds: CloudInfo[] = [];
  private sunDir: THREE.Vector3 = new THREE.Vector3(1, 0, 0);
  private nlat: number = 0;
  private nlon: number = 0;
  private lastMonthProgress: number = -1;
  private textures: THREE.Texture[] = [];
  private materials: THREE.SpriteMaterial[] = [];
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
    const data = totalField.data as Float32Array;

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
          csum += data[idx];
          if (windUData) usum += windUData[idx];
          if (windVData) vsum += windVData[idx];
        }
        const ci = i * nlon + j;
        annualMean[ci] = csum / 12;
        annualWindU[ci] = usum / 12;
        annualWindV[ci] = vsum / 12;
      }
    }

    // Create texture variants — shared materials to save memory
    const styles: CloudStyle[] = ['smallPuff', 'bigCumulus', 'elongated', 'towering', 'cluster'];
    for (let s = 0; s < styles.length; s++) {
      const tex = createCloudTexture(styles[s], 10000 + s * 777);
      this.textures.push(tex);
      this.materials.push(new THREE.SpriteMaterial({
        map: tex,
        transparent: true,
        depthWrite: false,
        opacity: 0,
        blending: THREE.NormalBlending,
      }));
    }

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

        // Pick a style — weight toward cumulus types, fewer clusters/towering
        const styleRoll = rand();
        let styleIdx: number;
        if (styleRoll < 0.35) styleIdx = 0;       // smallPuff
        else if (styleRoll < 0.65) styleIdx = 1;   // bigCumulus
        else if (styleRoll < 0.85) styleIdx = 3;   // towering
        else styleIdx = 4;                          // cluster

        // Each sprite gets its own material clone (needed for per-sprite opacity)
        const mat = this.materials[styleIdx].clone();

        const sprite = new THREE.Sprite(mat);
        // Size varies by style
        const baseSize = 0.06 + mean * 0.06 + rand() * 0.03;
        const aspectJitter = 0.85 + rand() * 0.3;
        let scaleX = baseSize * aspectJitter * 1.3;
        let scaleY = baseSize / aspectJitter * 0.9;
        // Elongated clouds are wider, towering are taller
        if (styleIdx === 3) { scaleX *= 0.8; scaleY *= 1.3; }
        if (styleIdx === 4) { scaleX *= 1.2; scaleY *= 1.1; }
        sprite.scale.set(scaleX, scaleY, 1);
        sprite.position.copy(latLonToPosition(lat, lon, r));
        sprite.visible = false;
        sprite.renderOrder = 10;

        const cosLat = Math.cos(lat * DEG2RAD);
        const ci = i * nlon + j;
        const windScale = 5000.0;
        const windU = (annualWindU[ci] / (111000 * Math.max(cosLat, 0.15))) * windScale;
        const windV = (annualWindV[ci] / 111000) * windScale;

        this.clouds.push({
          sprite, baseLat: lat, baseLon: lon,
          driftLat: 0, driftLon: 0,
          windU, windV,
          radius: r,
          cellI: i, cellJ: j,
          currentOpacity: 0, targetOpacity: 0,
        });
        this.group.add(sprite);
        placed++;
      }
    }
  }

  setMonth(monthProgress: number, layerData: ClimateLayerData): void {
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
    const data = totalField.data as Float32Array;

    this.windUData = layerData.wind_u_10m?.data as Float32Array ?? null;
    this.windVData = layerData.wind_v_10m?.data as Float32Array ?? null;

    for (const cloud of this.clouds) {
      const v0 = data[this.m0 * nlat * nlon + cloud.cellI * nlon + cloud.cellJ];
      const v1 = data[this.m1 * nlat * nlon + cloud.cellI * nlon + cloud.cellJ];
      const fraction = Math.max(0, Math.min(1, v0 + (v1 - v0) * this.monthFrac));
      cloud.targetOpacity = Math.min(1, fraction * 1.3);
    }
  }

  update(dt: number): void {
    for (const cloud of this.clouds) {
      const diff = cloud.targetOpacity - cloud.currentOpacity;
      if (Math.abs(diff) > 0.001) {
        cloud.currentOpacity += Math.sign(diff) * Math.min(Math.abs(diff), 0.08 * dt);
      }

      const mat = cloud.sprite.material as THREE.SpriteMaterial;
      mat.opacity = cloud.currentOpacity;

      if (cloud.currentOpacity < 0.005) {
        cloud.sprite.visible = false;
        continue;
      }

      cloud.sprite.visible = true;

      // Re-sample wind at current drifted position every frame
      if (this.windUData && this.windVData) {
        const nlat = this.nlat;
        const nlon = this.nlon;
        const latStep = 180 / nlat;
        const lonStep = 360 / nlon;
        const curLat = Math.max(-89, Math.min(89, cloud.baseLat + cloud.driftLat));
        const curLon = ((cloud.baseLon + cloud.driftLon) % 360 + 360) % 360;
        const ci = Math.min(nlat - 1, Math.max(0, Math.floor((curLat + 90) / latStep)));
        const cj = Math.min(nlon - 1, Math.floor(curLon / lonStep));
        const idx0 = this.m0 * nlat * nlon + ci * nlon + cj;
        const idx1 = this.m1 * nlat * nlon + ci * nlon + cj;
        const u = this.windUData[idx0] + (this.windUData[idx1] - this.windUData[idx0]) * this.monthFrac;
        const v = this.windVData[idx0] + (this.windVData[idx1] - this.windVData[idx0]) * this.monthFrac;
        const cosLat = Math.cos(curLat * DEG2RAD);
        const windScale = 5000.0;
        cloud.windU = (u / (111000 * Math.max(cosLat, 0.15))) * windScale;
        cloud.windV = (v / 111000) * windScale;
      }

      // Drift with local wind
      cloud.driftLon += cloud.windU * dt;
      cloud.driftLat += cloud.windV * dt;

      const lon = ((cloud.baseLon + cloud.driftLon) % 360 + 360) % 360;
      const lat = Math.max(-89, Math.min(89, cloud.baseLat + cloud.driftLat));
      cloud.sprite.position.copy(latLonToPosition(lat, lon, cloud.radius));

      // Day/night dimming
      const normal = cloud.sprite.position.clone().normalize();
      const surfaceDot = normal.dot(this.sunDir);
      const brightness = 0.12 + 0.88 * Math.max(0, Math.min(1, (surfaceDot + 0.05) / 0.2));
      mat.color.setRGB(brightness, brightness, brightness);
    }
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.sunDir.copy(dir);
  }

  getObject(): THREE.Object3D {
    return this.group;
  }

  dispose(): void {
    for (const cloud of this.clouds) {
      (cloud.sprite.material as THREE.SpriteMaterial).dispose();
    }
    for (const tex of this.textures) tex.dispose();
    for (const mat of this.materials) mat.dispose();
    this.textures = [];
    this.materials = [];
    this.clouds = [];
  }
}
