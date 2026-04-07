import * as THREE from 'three';
import { ELEVATION_SCALE, sampleElevation } from './elevation';

const GLOBE_RADIUS = 1.001;

const vertexShader = `
  attribute float aSize;
  uniform vec3 sunDirection;
  varying float vNightAlpha;
  varying float vSize;

  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vec3 worldDir = normalize(worldPos.xyz);

    // Nightside: dot with -sunDirection → 1 on dark side, 0 on lit side
    float nightDot = dot(worldDir, -sunDirection);
    vNightAlpha = smoothstep(-0.1, 0.2, nightDot);

    vSize = aSize;
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (200.0 / -mvPos.z);
    gl_Position = projectionMatrix * mvPos;
  }
`;

const fragmentShader = `
  varying float vNightAlpha;
  varying float vSize;

  void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    float glow = 1.0 - smoothstep(0.0, 0.4, dist);

    // Warm yellow-orange
    vec3 color = vec3(1.0, 0.85, 0.5);

    float alpha = glow * vNightAlpha * 0.7;
    if (alpha < 0.005) discard;
    gl_FragColor = vec4(color * alpha, alpha);
  }
`;

export class CityLights {
  private points: THREE.Points | null = null;
  private sunDirUniform = { value: new THREE.Vector3(1, 0, 0) };
  private group = new THREE.Group();
  private disposed = false;
  private elevData: Float32Array | null;
  private elevNlat: number;
  private elevNlon: number;

  constructor(elevData?: Float32Array, elevNlat?: number, elevNlon?: number) {
    this.elevData = elevData ?? null;
    this.elevNlat = elevNlat ?? 0;
    this.elevNlon = elevNlon ?? 0;
    this.load();
  }

  private async load(): Promise<void> {
    try {
      const resp = await fetch('data/cities.bin');
      const buf = await resp.arrayBuffer();
      if (this.disposed) return;

      const view = new DataView(buf);
      const count = view.getUint32(0, true);

      const positions = new Float32Array(count * 3);
      const sizes = new Float32Array(count);

      for (let i = 0; i < count; i++) {
        const offset = 4 + i * 12;
        const lon = view.getFloat32(offset, true);
        const lat = view.getFloat32(offset + 4, true);
        const pop = view.getFloat32(offset + 8, true);

        // Lift the city above the displaced terrain in blue marble mode so
        // it isn't buried inside a mountain. The same lookup the mesh uses.
        let radius = GLOBE_RADIUS;
        if (this.elevData) {
          const elev = sampleElevation(this.elevData, this.elevNlat, this.elevNlon, lat, lon);
          radius += Math.max(0, elev) * ELEVATION_SCALE;
        }

        // Convert lat/lon to 3D position on globe
        const phi = (90 - lat) * (Math.PI / 180);
        const theta = lon * (Math.PI / 180);
        positions[i * 3] = -radius * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = radius * Math.cos(phi);
        positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);

        // Size from log(population): range ~2 (small towns) to ~8 (megacities)
        // Use sqrt(population) for wide dynamic range:
        // 10k → 0.3, 100k → 1.0, 1M → 3.2, 10M → 10, 37M → 19
        // cbrt: 10k→0.27, 100k→0.58, 1M→1.25, 10M→2.7
        sizes[i] = Math.cbrt(pop) / 800;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1));

      const material = new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: {
          sunDirection: this.sunDirUniform,
        },
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });

      this.points = new THREE.Points(geometry, material);
      this.group.add(this.points);
    } catch (e) {
      console.error('Failed to load city lights:', e);
    }
  }

  setSunDirection(dir: THREE.Vector3): void {
    this.sunDirUniform.value.copy(dir);
  }

  getObject(): THREE.Object3D {
    return this.group;
  }

  dispose(): void {
    this.disposed = true;
    if (this.points) {
      this.points.geometry.dispose();
      (this.points.material as THREE.Material).dispose();
    }
  }
}
