import * as THREE from 'three';
import type { ClimateLayerData, FieldData } from './loadBinaryData';

const DEG2RAD = Math.PI / 180;
const CLOUD_RADIUS = 1.015;
const SEED = 7919;

// Only place clouds in cells with annual-mean fraction above this
const MIN_FRACTION = 0.05;
// Max clouds total (budget for performance)
const MAX_CLOUDS = 600;

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

// ---- High-quality procedural cloud texture ----

function createCloudTexture(): THREE.Texture {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, size, size);

  const cx = size / 2;
  const cy = size / 2;

  // Build up a cloud from many soft overlapping circles
  // This creates a natural puffy shape with soft edges
  const rand = mulberry32(12345);

  // Large base blobs
  const blobs: { x: number; y: number; r: number; a: number }[] = [
    // Core mass
    { x: cx, y: cy, r: 48, a: 0.35 },
    { x: cx - 20, y: cy + 5, r: 40, a: 0.3 },
    { x: cx + 22, y: cy + 3, r: 42, a: 0.3 },
    { x: cx + 5, y: cy - 10, r: 36, a: 0.28 },
    // Puffy bumps on top
    { x: cx - 10, y: cy - 20, r: 30, a: 0.25 },
    { x: cx + 12, y: cy - 18, r: 28, a: 0.25 },
    { x: cx, y: cy - 28, r: 22, a: 0.2 },
    // Wider base
    { x: cx - 35, y: cy + 8, r: 30, a: 0.2 },
    { x: cx + 38, y: cy + 6, r: 28, a: 0.2 },
  ];

  // Add some random smaller puffs for detail
  for (let i = 0; i < 12; i++) {
    blobs.push({
      x: cx + (rand() - 0.5) * 70,
      y: cy + (rand() - 0.5) * 50 - 5,
      r: 12 + rand() * 20,
      a: 0.1 + rand() * 0.15,
    });
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

  // Slight bottom darkening (belly shadow)
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
  driftLat: number; // accumulated lat drift in degrees
  driftLon: number; // accumulated lon drift in degrees
  windU: number;    // annual-mean zonal wind (deg/s, positive = eastward)
  windV: number;    // annual-mean meridional wind (deg/s, positive = northward)
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
  private texture: THREE.Texture;

  constructor(layerData: ClimateLayerData) {
    this.group = new THREE.Group();

    // Determine which field to use for placement (combined cloud_fraction or sum of components)
    const totalField = layerData.cloud_fraction;
    if (!totalField) return;

    this.nlat = totalField.shape[1];
    this.nlon = totalField.shape[2];
    const nlat = this.nlat;
    const nlon = this.nlon;
    const latStep = 180 / nlat;
    const lonStep = 360 / nlon;
    const data = totalField.data as Float32Array;

    // Compute annual means per cell: cloud fraction + wind
    const annualMean = new Float32Array(nlat * nlon);
    const annualWindU = new Float32Array(nlat * nlon); // m/s, positive = eastward
    const annualWindV = new Float32Array(nlat * nlon); // m/s, positive = northward
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

    this.texture = createCloudTexture();
    const rand = mulberry32(SEED);

    // Collect all candidate cells, then shuffle for even global coverage
    const cells: { i: number; j: number; mean: number }[] = [];
    for (let i = 0; i < nlat; i++) {
      for (let j = 0; j < nlon; j++) {
        const mean = annualMean[i * nlon + j];
        if (mean >= MIN_FRACTION) {
          cells.push({ i, j, mean });
        }
      }
    }
    // Fisher-Yates shuffle
    for (let n = cells.length - 1; n > 0; n--) {
      const swap = Math.floor(rand() * (n + 1));
      [cells[n], cells[swap]] = [cells[swap], cells[n]];
    }

    let placed = 0;
    for (const cell of cells) {
      if (placed >= MAX_CLOUDS) break;

      const { i, j, mean } = cell;

      // Expected count scales with fraction
      const expected = mean * 3.0;
      let count = Math.floor(expected);
      if (rand() < (expected - count)) count++;
      if (count < 1) count = 1;

      for (let k = 0; k < count && placed < MAX_CLOUDS; k++) {
          const lat = -90 + i * latStep + rand() * latStep;
          const lon = (j * lonStep + rand() * lonStep) % 360;
          const r = CLOUD_RADIUS + rand() * 0.01;
          const baseSize = 0.06 + mean * 0.06 + rand() * 0.03;

          const mat = new THREE.SpriteMaterial({
            map: this.texture,
            transparent: true,
            depthWrite: false,
            opacity: 0,
            blending: THREE.NormalBlending,
          });

          const sprite = new THREE.Sprite(mat);
          const aspectJitter = 0.85 + rand() * 0.3;
          sprite.scale.set(baseSize * aspectJitter * 1.3, baseSize / aspectJitter * 0.9, 1);
          sprite.position.copy(latLonToPosition(lat, lon, r));
          sprite.visible = false;
          sprite.renderOrder = 10;

          // Convert wind m/s to degrees/s
          // 1 deg latitude ≈ 111km, 1 deg longitude ≈ 111km * cos(lat)
          const cosLat = Math.cos(lat * DEG2RAD);
          const ci = i * nlon + j;
          const windScale = 8.0; // speed up for visual drama
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
    const m0 = Math.floor(mp) % 12;
    const m1 = (m0 + 1) % 12;
    const frac = mp - Math.floor(mp);

    const nlat = this.nlat;
    const nlon = this.nlon;

    const totalField = layerData.cloud_fraction;
    if (!totalField) return;
    const data = totalField.data as Float32Array;

    for (const cloud of this.clouds) {
      const v0 = data[m0 * nlat * nlon + cloud.cellI * nlon + cloud.cellJ];
      const v1 = data[m1 * nlat * nlon + cloud.cellI * nlon + cloud.cellJ];
      const fraction = Math.max(0, Math.min(1, v0 + (v1 - v0) * frac));

      // Target opacity tracks the cloud fraction directly
      // Scale it up a bit since we're only showing some cells
      cloud.targetOpacity = Math.min(1, fraction * 1.3);
    }
  }

  /** Call each frame with dt in seconds. */
  update(dt: number): void {
    for (const cloud of this.clouds) {
      // Smooth opacity fade (~3s transition)
      const diff = cloud.targetOpacity - cloud.currentOpacity;
      if (Math.abs(diff) > 0.001) {
        cloud.currentOpacity += Math.sign(diff) * Math.min(Math.abs(diff), 0.35 * dt);
      }

      const mat = cloud.sprite.material as THREE.SpriteMaterial;
      mat.opacity = cloud.currentOpacity;

      if (cloud.currentOpacity < 0.005) {
        cloud.sprite.visible = false;
        continue;
      }

      cloud.sprite.visible = true;

      // Drift with local wind
      cloud.driftLon += cloud.windU * dt;
      cloud.driftLat += cloud.windV * dt;

      // Wrap drift back when too far from base cell (smooth pullback)
      const maxDrift = 15;
      if (Math.abs(cloud.driftLon) > maxDrift) cloud.driftLon *= 0.5;
      if (Math.abs(cloud.driftLat) > maxDrift) cloud.driftLat *= 0.5;

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
    this.texture.dispose();
    this.clouds = [];
  }
}
