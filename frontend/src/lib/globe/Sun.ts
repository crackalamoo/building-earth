/**
 * Sun orb, screen-space bloom, and declination helpers.
 */
import * as THREE from 'three';

// ── Helpers ──────────────────────────────────────────────────────────

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

// ── Sun declination ──────────────────────────────────────────────────

/** Compute solar declination in radians for a given month value (0-12 continuous). */
export function getSunDeclination(monthValue: number): number {
  const declinationDeg = 23.5 * Math.sin((monthValue - 2) / 12 * 2 * Math.PI);
  return declinationDeg * (Math.PI / 180);
}

// ── Sun orb ──────────────────────────────────────────────────────────

/** Create the visible sun disc with limb darkening. */
export function createSunOrb(): THREE.Mesh {
  const geo = new THREE.SphereGeometry(7.2, 32, 32);
  const mat = new THREE.ShaderMaterial({
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
  return new THREE.Mesh(geo, mat);
}

// ── Screen-space sun bloom ───────────────────────────────────────────

/** Create the fullscreen bloom overlay quad. */
export function createSunBloom(): THREE.Mesh {
  const mat = new THREE.ShaderMaterial({
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec2 sunScreenPos;  // sun position in screen UV space (0-1)
      uniform float sunVisible;   // 0 = behind camera, 1 = in front
      uniform float sunIntensity; // how directly we're facing the sun
      uniform float aspectRatio;  // width / height
      uniform vec2 globeScreenPos;  // globe center in screen UV space
      uniform float globeScreenRadius; // globe's apparent radius in UV units (corrected for aspect)
      varying vec2 vUv;

      void main() {
        if (sunVisible < 0.01) discard;

        // Fade bloom behind globe disc
        vec2 toGlobe = vUv - globeScreenPos;
        toGlobe.x *= aspectRatio;
        float globeDist = length(toGlobe);
        float globeMask = smoothstep(globeScreenRadius - 0.003, globeScreenRadius + 0.003, globeDist);
        if (globeMask < 0.001) discard;

        // Distance from sun's screen position, aspect-corrected
        vec2 delta = vUv - sunScreenPos;
        delta.x *= aspectRatio;
        float d = length(delta);

        // Core bloom — intense near sun position
        float core = exp(-d * d * 120.0) * 2.5;

        // Mid glow — warm spread
        float mid = exp(-d * d * 15.0) * 0.6;

        // Wide wash — entire screen tint when facing sun
        float wash = exp(-d * d * 2.5) * 0.2;

        // Subtle anamorphic horizontal streak (lens artifact)
        float streak = exp(-abs(delta.y) * 40.0) * exp(-delta.x * delta.x * 3.0) * 0.12;

        float intensity = (core + mid + wash + streak) * sunIntensity * sunVisible * globeMask;

        // White-hot center → warm amber edge
        vec3 white = vec3(1.0, 1.0, 0.95);
        vec3 amber = vec3(1.0, 0.75, 0.35);
        vec3 col = mix(white, amber, smoothstep(0.0, 0.3, d));

        gl_FragColor = vec4(col * intensity, intensity);
      }
    `,
    uniforms: {
      sunScreenPos: { value: new THREE.Vector2(0.5, 0.5) },
      sunVisible: { value: 0.0 },
      sunIntensity: { value: 0.0 },
      aspectRatio: { value: 1.0 },
      globeScreenPos: { value: new THREE.Vector2(0.5, 0.5) },
      globeScreenRadius: { value: 0.3 },
    },
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
    depthTest: false,
  });
  const geo = new THREE.PlaneGeometry(2, 2);
  const quad = new THREE.Mesh(geo, mat);
  quad.frustumCulled = false;
  quad.renderOrder = 999;
  return quad;
}

/**
 * Update screen-space sun bloom uniforms each frame.
 * Handles camera-facing check, globe occlusion, and screen-space projection.
 */
// Reusable scratch vectors so updateSunBloom does zero per-frame allocations
// (it runs every frame; new Vector3s here pile up GC pressure fast).
const _sbSunNDC = new THREE.Vector3();
const _sbCamDir = new THREE.Vector3();
const _sbToSun = new THREE.Vector3();
const _sbToSunDir = new THREE.Vector3();
const _sbToGlobeDir = new THREE.Vector3();
const _sbGlobeCenter = new THREE.Vector3();

export function updateSunBloom(
  bloomQuad: THREE.Mesh,
  sunOrb: THREE.Mesh,
  camera: THREE.PerspectiveCamera,
): void {
  const mat = bloomQuad.material as THREE.ShaderMaterial;
  _sbSunNDC.copy(sunOrb.position).project(camera);

  // Check if sun is in front of camera
  camera.getWorldDirection(_sbCamDir);
  _sbToSun.copy(sunOrb.position).normalize();
  const facing = _sbCamDir.dot(_sbToSun);

  // Ray-sphere occlusion: check if sun disc is blocked by globe
  const sunAngularRadius = 7.2 / 360; // ~0.02 rad
  const globeRadius = 1.0;
  const camPos = camera.position;
  const camDist = camPos.length();
  const globeAngularRadius = Math.asin(Math.min(1, globeRadius / camDist));
  _sbToSunDir.copy(sunOrb.position).sub(camPos).normalize();
  _sbToGlobeDir.copy(camPos).negate().normalize();
  const angleBetween = Math.acos(Math.min(1, Math.max(-1, _sbToSunDir.dot(_sbToGlobeDir))));
  const clearAngle = globeAngularRadius + sunAngularRadius;
  const visibility = smoothstep(clearAngle - 0.04, clearAngle + 0.02, angleBetween);

  if (facing > -0.2 && visibility > 0.001) {
    mat.uniforms.sunVisible.value = smoothstep(-0.2, 0.1, facing) * visibility;
    mat.uniforms.sunScreenPos.value.set(
      _sbSunNDC.x * 0.5 + 0.5,
      _sbSunNDC.y * 0.5 + 0.5,
    );
    mat.uniforms.sunIntensity.value = 1.0;
    mat.uniforms.aspectRatio.value = camera.aspect;

    // Globe screen-space disc for masking
    _sbGlobeCenter.set(0, 0, 0).project(camera);
    const gcx = _sbGlobeCenter.x * 0.5 + 0.5;
    const gcy = _sbGlobeCenter.y * 0.5 + 0.5;
    mat.uniforms.globeScreenPos.value.set(gcx, gcy);
    const cDist = camera.position.length();
    const R = 1.0;
    const limbAngularRadius = Math.asin(Math.min(1, R / cDist));
    const vFov = camera.fov * Math.PI / 180;
    const screenRadiusY = Math.tan(limbAngularRadius) / Math.tan(vFov / 2);
    mat.uniforms.globeScreenRadius.value = screenRadiusY * 0.5;
  } else {
    mat.uniforms.sunVisible.value = 0.0;
  }
}

/** Dispose sun orb and bloom quad geometries and materials. */
export function disposeSun(sunOrb: THREE.Mesh | null, bloomQuad: THREE.Object3D | null): void {
  if (sunOrb) {
    sunOrb.geometry.dispose();
    (sunOrb.material as THREE.Material).dispose();
  }
  if (bloomQuad) {
    ((bloomQuad as THREE.Mesh).geometry as THREE.BufferGeometry)?.dispose();
    ((bloomQuad as THREE.Mesh).material as THREE.Material)?.dispose();
  }
}
