/**
 * Animated wind particle system with comet-like trails on the globe surface.
 *
 * Each particle stores a trail of recent positions. The trail is rendered as
 * line segments that fade from bright (head) to transparent (tail).
 */

import * as THREE from 'three';
import type { FieldData } from './loadBinaryData';

const PARTICLE_COUNT = 15000;
const TRAIL_LENGTH = 30; // positions per particle — longer comet tails
const GLOBE_RADIUS = 1.002;
const MIN_LIFETIME = 3.0;
const MAX_LIFETIME = 7.0;
const FADE_FRACTION = 0.2;
const ADVECTION_SPEED = 0.30; // degrees per (m/s * second)

// Each trail is TRAIL_LENGTH points → (TRAIL_LENGTH - 1) segments → 2 vertices per segment
const VERTS_PER_PARTICLE = (TRAIL_LENGTH - 1) * 2;
const TOTAL_VERTS = PARTICLE_COUNT * VERTS_PER_PARTICLE;

interface WindFields {
  wind_u_10m: FieldData;
  wind_v_10m: FieldData;
  wind_speed_10m: FieldData;
}

export class WindParticles {
  private mesh: THREE.LineSegments;
  private positions: Float32Array;
  private alphas: Float32Array;

  // Per-particle state: trail of lat/lon positions, newest at index 0
  private trailLats: Float32Array; // [PARTICLE_COUNT * TRAIL_LENGTH]
  private trailLons: Float32Array;
  private ages: Float32Array;
  private lifetimes: Float32Array;

  private windFields: WindFields;
  private monthIndex: number = 0;
  private nlat: number;
  private nlon: number;

  private spawnCDF: Float32Array | null = null;
  private spawnMonth: number = -1;

  constructor(windFields: WindFields) {
    this.windFields = windFields;
    const shape = windFields.wind_u_10m.shape;
    this.nlat = shape[1];
    this.nlon = shape[2];

    this.positions = new Float32Array(TOTAL_VERTS * 3);
    this.alphas = new Float32Array(TOTAL_VERTS);
    this.trailLats = new Float32Array(PARTICLE_COUNT * TRAIL_LENGTH);
    this.trailLons = new Float32Array(PARTICLE_COUNT * TRAIL_LENGTH);
    this.ages = new Float32Array(PARTICLE_COUNT);
    this.lifetimes = new Float32Array(PARTICLE_COUNT);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      this.respawnParticle(i, true);
    }
    this.rebuildBuffers();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    geometry.setAttribute('alpha', new THREE.BufferAttribute(this.alphas, 1));

    const material = new THREE.ShaderMaterial({
      vertexShader: `
        attribute float alpha;
        varying float vAlpha;
        void main() {
          vAlpha = alpha;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying float vAlpha;
        void main() {
          gl_FragColor = vec4(0.7, 0.85, 1.0, vAlpha);
        }
      `,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.mesh = new THREE.LineSegments(geometry, material);
  }

  getObject(): THREE.Object3D {
    return this.mesh;
  }

  setMonth(monthIndex: number): void {
    this.monthIndex = Math.floor(monthIndex) % 12;
  }

  update(dt: number): void {
    const month = this.monthIndex;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      this.ages[i] += dt;

      if (this.ages[i] >= this.lifetimes[i]) {
        this.respawnParticle(i, false);
        continue;
      }

      // Shift trail: move all positions back by one slot
      const base = i * TRAIL_LENGTH;
      for (let t = TRAIL_LENGTH - 1; t > 0; t--) {
        this.trailLats[base + t] = this.trailLats[base + t - 1];
        this.trailLons[base + t] = this.trailLons[base + t - 1];
      }

      // Advect head position
      const headLat = this.trailLats[base];
      const headLon = this.trailLons[base];
      const { u, v } = this.sampleWind(month, headLat, headLon);

      const latRad = headLat * (Math.PI / 180);
      const cosLat = Math.cos(latRad);
      const dlat = v * ADVECTION_SPEED * dt;
      const dlon = cosLat > 0.05 ? (u * ADVECTION_SPEED * dt) / cosLat : 0;

      this.trailLats[base] = Math.max(-89, Math.min(89, headLat + dlat));
      this.trailLons[base] = ((headLon + dlon) % 360 + 360) % 360;
    }

    this.rebuildBuffers();
    (this.mesh.geometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
    (this.mesh.geometry.attributes.alpha as THREE.BufferAttribute).needsUpdate = true;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }

  // ── Private ──────────────────────────────────────────────────────

  private respawnParticle(i: number, randomAge: boolean): void {
    const { lat, lon } = this.sampleSpawnPosition();
    const base = i * TRAIL_LENGTH;
    // Set all trail points to the spawn position (no trail yet)
    for (let t = 0; t < TRAIL_LENGTH; t++) {
      this.trailLats[base + t] = lat;
      this.trailLons[base + t] = lon;
    }
    this.lifetimes[i] = MIN_LIFETIME + Math.random() * (MAX_LIFETIME - MIN_LIFETIME);
    this.ages[i] = randomAge ? Math.random() * this.lifetimes[i] : 0;
  }

  private rebuildBuffers(): void {
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const trailBase = i * TRAIL_LENGTH;
      const vertBase = i * VERTS_PER_PARTICLE;

      // Compute particle-level alpha from age
      const ageFrac = this.ages[i] / this.lifetimes[i];
      let particleAlpha: number;
      if (ageFrac < FADE_FRACTION) {
        particleAlpha = ageFrac / FADE_FRACTION;
      } else if (ageFrac > 1 - FADE_FRACTION) {
        particleAlpha = (1 - ageFrac) / FADE_FRACTION;
      } else {
        particleAlpha = 1;
      }

      // Scale by wind speed
      const headLat = this.trailLats[trailBase];
      const headLon = this.trailLons[trailBase];
      const speed = this.sampleWindSpeed(this.monthIndex, headLat, headLon);
      const speedFactor = Math.min(speed / 10.0, 1.0);
      particleAlpha *= (0.3 + 0.7 * speedFactor);

      // Build line segments: each segment connects trail[t] → trail[t+1]
      for (let t = 0; t < TRAIL_LENGTH - 1; t++) {
        const segIdx = vertBase + t * 2;
        // Trail fades: head (t=0) is brightest, tail is transparent
        const headFade = 1.0 - t / (TRAIL_LENGTH - 1);
        const tailFade = 1.0 - (t + 1) / (TRAIL_LENGTH - 1);

        // Start vertex of segment
        const lat0 = this.trailLats[trailBase + t];
        const lon0 = this.trailLons[trailBase + t];
        const [x0, y0, z0] = this.latLonToXYZ(lat0, lon0);
        this.positions[segIdx * 3] = x0;
        this.positions[segIdx * 3 + 1] = y0;
        this.positions[segIdx * 3 + 2] = z0;
        this.alphas[segIdx] = particleAlpha * headFade;

        // End vertex of segment
        const lat1 = this.trailLats[trailBase + t + 1];
        const lon1 = this.trailLons[trailBase + t + 1];
        const [x1, y1, z1] = this.latLonToXYZ(lat1, lon1);
        this.positions[(segIdx + 1) * 3] = x1;
        this.positions[(segIdx + 1) * 3 + 1] = y1;
        this.positions[(segIdx + 1) * 3 + 2] = z1;
        this.alphas[segIdx + 1] = particleAlpha * tailFade;
      }
    }
  }

  private latLonToXYZ(lat: number, lon: number): [number, number, number] {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = lon * (Math.PI / 180);
    return [
      -GLOBE_RADIUS * Math.sin(phi) * Math.cos(theta),
      GLOBE_RADIUS * Math.cos(phi),
      GLOBE_RADIUS * Math.sin(phi) * Math.sin(theta),
    ];
  }

  private sampleSpawnPosition(): { lat: number; lon: number } {
    if (this.spawnMonth !== this.monthIndex || !this.spawnCDF) {
      this.buildSpawnCDF();
    }
    const r = Math.random();
    const cdf = this.spawnCDF!;
    let lo = 0, hi = cdf.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < r) lo = mid + 1;
      else hi = mid;
    }
    const latIdx = Math.floor(lo / this.nlon);
    const lonIdx = lo % this.nlon;
    const latStep = 180 / this.nlat;
    const lonStep = 360 / this.nlon;
    const lat = -90 + (latIdx + Math.random()) * latStep;
    const lon = (lonIdx + Math.random()) * lonStep;
    return { lat: Math.max(-89, Math.min(89, lat)), lon: lon % 360 };
  }

  private buildSpawnCDF(): void {
    const n = this.nlat * this.nlon;
    const cdf = new Float32Array(n);
    const speedData = this.windFields.wind_speed_10m.data as Float32Array;
    const base = this.monthIndex * n;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.max(speedData[base + i], 0.5);
      total += w;
      cdf[i] = total;
    }
    for (let i = 0; i < n; i++) {
      cdf[i] /= total;
    }
    this.spawnCDF = cdf;
    this.spawnMonth = this.monthIndex;
  }

  /** Bilinear interpolation of a wind component field at (lat, lon). */
  private sampleFieldBilinear(field: Float32Array, month: number, lat: number, lon: number): number {
    const latStep = 180 / this.nlat;
    const lonStep = 360 / this.nlon;
    // Continuous grid coordinates (cell centers at 0.5, 1.5, ...)
    const latF = (lat + 90) / latStep - 0.5;
    const lonF = lon / lonStep - 0.5;

    const lat0 = Math.max(0, Math.min(this.nlat - 1, Math.floor(latF)));
    const lat1 = Math.min(this.nlat - 1, lat0 + 1);
    const lon0 = ((Math.floor(lonF) % this.nlon) + this.nlon) % this.nlon;
    const lon1 = (lon0 + 1) % this.nlon;

    const tLat = Math.max(0, Math.min(1, latF - Math.floor(latF)));
    const tLon = Math.max(0, Math.min(1, lonF - Math.floor(lonF)));

    const base = month * this.nlat * this.nlon;
    const v00 = field[base + lat0 * this.nlon + lon0];
    const v10 = field[base + lat1 * this.nlon + lon0];
    const v01 = field[base + lat0 * this.nlon + lon1];
    const v11 = field[base + lat1 * this.nlon + lon1];

    return (
      v00 * (1 - tLat) * (1 - tLon) +
      v10 * tLat * (1 - tLon) +
      v01 * (1 - tLat) * tLon +
      v11 * tLat * tLon
    );
  }

  private sampleWind(month: number, lat: number, lon: number): { u: number; v: number } {
    return {
      u: this.sampleFieldBilinear(this.windFields.wind_u_10m.data as Float32Array, month, lat, lon),
      v: this.sampleFieldBilinear(this.windFields.wind_v_10m.data as Float32Array, month, lat, lon),
    };
  }

  private sampleWindSpeed(month: number, lat: number, lon: number): number {
    return this.sampleFieldBilinear(this.windFields.wind_speed_10m.data as Float32Array, month, lat, lon);
  }
}
