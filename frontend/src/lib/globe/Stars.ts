/**
 * Star field and Milky Way for the globe scene.
 *
 * Star data: HYG Database (astronexus/HYG-Database), CC-BY-SA 4.0
 * https://github.com/astronexus/HYG-Database
 */
import * as THREE from 'three';

const STAR_RADIUS = 200;
const MILKY_WAY_RADIUS = 190;

// ---------- B-V color index → RGB ----------

function bvToColor(ci: number): [number, number, number] {
  // B-V → color temperature → sRGB, based on Ballesteros 2012 formula
  // T = 4600 * (1/(0.92*BV + 1.7) + 1/(0.92*BV + 0.62))
  const bv = Math.max(-0.4, Math.min(2.0, ci));
  const temp = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));

  // Attempt at Planckian locus mapping to sRGB, keeping most stars
  // looking white-ish with subtle tints (matching real night sky perception).
  // Hot stars (>10000K): bluish-white. Sun-like (~5800K): pure white.
  // Cool stars (<4000K): warm yellow to soft orange. Only the coolest look red.
  let r: number, g: number, b: number;

  if (temp >= 10000) {
    // Hot blue-white (O, B stars)
    const t = Math.min(1, (temp - 10000) / 20000);
    r = 0.85 - 0.15 * t;
    g = 0.90 - 0.05 * t;
    b = 1.0;
  } else if (temp >= 6500) {
    // White to blue-white (A, F stars)
    const t = (temp - 6500) / 3500;
    r = 1.0 - 0.15 * t;
    g = 1.0 - 0.10 * t;
    b = 1.0;
  } else if (temp >= 5000) {
    // White to warm white (G stars, including the Sun at ~5800K)
    const t = (6500 - temp) / 1500;
    r = 1.0;
    g = 1.0 - 0.05 * t;
    b = 1.0 - 0.12 * t;
  } else if (temp >= 3700) {
    // Warm yellow-white (K stars)
    const t = (5000 - temp) / 1300;
    r = 1.0;
    g = 0.95 - 0.15 * t;
    b = 0.88 - 0.28 * t;
  } else {
    // Soft orange (M stars — only the very coolest look distinctly colored)
    const t = Math.min(1, (3700 - temp) / 1200);
    r = 1.0;
    g = 0.80 - 0.15 * t;
    b = 0.60 - 0.20 * t;
  }

  return [r, g, b];
}

// ---------- Star Points ----------

const starVertexShader = `
  attribute float size;
  varying vec3 vColor;
  void main() {
    vColor = color;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size;
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const starFragmentShader = `
  varying vec3 vColor;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    // Additive blending: output = src.rgb * src.a + dst.rgb
    // Put full color in rgb, use alpha for falloff shape
    float shape = 1.0 - d * d;
    gl_FragColor = vec4(vColor, shape);
  }
`;

type StarData = [number, number, number, number][]; // [ra_hours, dec_deg, mag, ci]

function createStarPoints(stars: StarData): THREE.Points {
  const count = stars.length;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const [ra, dec, mag, ci] = stars[i];
    // RA in hours → radians, Dec in degrees → radians
    const raRad = ra * (Math.PI / 12);
    const decRad = dec * (Math.PI / 180);

    // Equatorial → cartesian (right-handed, y=north pole)
    const cosDec = Math.cos(decRad);
    positions[i * 3] = STAR_RADIUS * cosDec * Math.cos(raRad);
    positions[i * 3 + 1] = STAR_RADIUS * Math.sin(decRad);
    positions[i * 3 + 2] = -STAR_RADIUS * cosDec * Math.sin(raRad);

    // Normalized brightness: mag -1.5 (Sirius) → 1.0, mag 5 → 0.0
    const t = Math.max(0, Math.min(1, (5.0 - mag) / 6.5));

    // Size: 2px for dimmest, 8px for brightest (Sirius)
    sizes[i] = 2.0 + t * t * 6.0;

    const [r, g, b] = bvToColor(ci);
    // Brightness: dim stars still clearly visible, bright stars saturate
    const lum = 0.7 + 1.3 * t;
    colors[i * 3] = r * lum;
    colors[i * 3 + 1] = g * lum;
    colors[i * 3 + 2] = b * lum;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader: starVertexShader,
    fragmentShader: starFragmentShader,
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  return new THREE.Points(geometry, material);
}

// ---------- Milky Way ----------

const milkyWayVertexShader = `
  varying vec3 vPos;
  void main() {
    vPos = position; // local-space position (on the sphere)
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const milkyWayFragmentShader = `
  uniform float rotationY;
  varying vec3 vPos;

  // Equatorial → galactic coordinate rotation matrix (J2000)
  const mat3 eqToGal = mat3(
    -0.0548756, -0.8734371, -0.4838350,
     0.4941094, -0.4448296,  0.7469823,
    -0.8676661, -0.1980764,  0.4559838
  );

  // --- Noise functions ---
  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(269.5, 183.3))) * 43758.5453);
  }

  float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(hash(i), hash(i + vec2(1,0)), u.x),
      mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
      u.y
    );
  }

  float fbm(vec2 p, int octaves) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8, 0.6, -0.6, 0.8); // domain rotation to reduce axis alignment
    for (int i = 0; i < 6; i++) {
      if (i >= octaves) break;
      v += a * noise(p);
      p = rot * p * 2.0;
      a *= 0.5;
    }
    return v;
  }

  void main() {
    // Apply Y-axis rotation in the shader (matches star group rotation)
    vec3 p = normalize(vPos);
    float c = cos(rotationY), s = sin(rotationY);
    // Match THREE.js Y-axis rotation: (x,z) → (x·cos - z·sin, x·sin + z·cos)
    vec3 dir = vec3(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
    // Convert our convention (x=cos δ cos α, y=sin δ, z=-cos δ sin α)
    // to standard astronomical (x=cos δ cos α, y=cos δ sin α, z=sin δ)
    vec3 eqStd = vec3(dir.x, -dir.z, dir.y);
    vec3 gal = eqToGal * eqStd;
    float b = asin(clamp(gal.y, -1.0, 1.0)); // galactic latitude
    float l = atan(gal.z, gal.x);             // galactic longitude

    // UV for noise sampling — stretched along the band
    vec2 uv = vec2(l, b);

    // === Layer 1: Broad diffuse halo ===
    float sigmaWide = 0.35; // ~20°
    float halo = exp(-b * b / (2.0 * sigmaWide * sigmaWide));
    // Gentle large-scale modulation
    halo *= 0.7 + 0.3 * fbm(uv * 2.0, 3);

    // === Layer 2: Narrow bright core band ===
    float sigmaCore = 0.12; // ~7°
    float core = exp(-b * b / (2.0 * sigmaCore * sigmaCore));
    // Wavy displacement of the core (the real MW band isn't perfectly straight)
    float warp = 0.04 * sin(l * 2.5 + 1.0) + 0.02 * sin(l * 5.0 + 3.0);
    float bWarped = b - warp;
    core = exp(-bWarped * bWarped / (2.0 * sigmaCore * sigmaCore));
    // Fine structure in the core
    float coreNoise = fbm(uv * vec2(6.0, 20.0), 5);
    core *= 0.5 + 0.5 * coreNoise;

    // === Layer 3: Dust lanes (dark absorption) ===
    // Narrow dark filaments slightly offset from center
    float dust1 = exp(-(bWarped - 0.03) * (bWarped - 0.03) / (2.0 * 0.04 * 0.04));
    float dust2 = exp(-(bWarped + 0.02) * (bWarped + 0.02) / (2.0 * 0.03 * 0.03));
    float dustNoise = fbm(uv * vec2(8.0, 30.0) + 5.0, 5);
    float dustMask = (dust1 + dust2 * 0.6) * (0.3 + 0.7 * dustNoise);
    dustMask = clamp(dustMask * 0.7, 0.0, 0.8);

    // === Layer 4: Star clouds (bright clumps) ===
    float clouds = fbm(uv * vec2(10.0, 25.0) + 10.0, 5);
    clouds = smoothstep(0.45, 0.75, clouds); // threshold to get discrete clumps
    float cloudBand = exp(-b * b / (2.0 * 0.18 * 0.18));
    clouds *= cloudBand;

    // === Layer 5: Galactic center bulge ===
    // Wrap longitude so l=0 (galactic center) doesn't split at ±π
    float lc = l;
    if (lc > 3.0) lc -= 6.2832;
    if (lc < -3.0) lc += 6.2832;
    float bulgeR = sqrt(lc * lc * 1.5 + b * b * 4.0);
    float bulge = exp(-bulgeR * bulgeR / (2.0 * 0.18 * 0.18));
    // Bulge has its own mottled texture
    float bulgeNoise = fbm(uv * 15.0 + 20.0, 4);
    bulge *= 0.7 + 0.3 * bulgeNoise;

    // === Longitude falloff: brightest near galactic center, fades toward anti-center ===
    // lc=0 is center, lc=±π is anti-center
    float lonFade = 0.3 + 0.7 * exp(-lc * lc / (2.0 * 1.2 * 1.2));

    // === Combine density ===
    float density = (halo * 0.35 + core * 0.8 + clouds * 0.3 + bulge * 0.6) * lonFade;
    // Apply dust absorption
    density *= 1.0 - dustMask;
    density = clamp(density, 0.0, 1.0);

    // === Color ===
    // Base: cool blue-white for disk
    vec3 diskColor = vec3(0.65, 0.7, 0.85);
    // Core: warmer white
    vec3 coreColor = vec3(0.85, 0.82, 0.78);
    // Bulge: warm golden
    vec3 bulgeColor = vec3(0.95, 0.85, 0.65);
    // Star clouds: slightly blue-white
    vec3 cloudColor = vec3(0.8, 0.82, 0.9);

    float coreWeight = core / max(density, 0.01);
    float bulgeWeight = bulge * 0.6 / max(density, 0.01);
    float cloudWeight = clouds * 0.3 / max(density, 0.01);

    vec3 color = diskColor;
    color = mix(color, coreColor, clamp(coreWeight, 0.0, 1.0));
    color = mix(color, bulgeColor, clamp(bulgeWeight, 0.0, 1.0));
    color = mix(color, cloudColor, clamp(cloudWeight * 0.5, 0.0, 1.0));

    // Dust-reddened regions (where absorption is high, remaining light is redder)
    vec3 dustTint = vec3(0.7, 0.5, 0.35);
    color = mix(color, dustTint, dustMask * 0.3);

    float alpha = density * 0.15;
    gl_FragColor = vec4(color, alpha);
  }
`;

function createMilkyWay(): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(MILKY_WAY_RADIUS, 96, 48);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      rotationY: { value: 0.0 },
    },
    vertexShader: milkyWayVertexShader,
    fragmentShader: milkyWayFragmentShader,
    transparent: true,
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  return new THREE.Mesh(geometry, material);
}

// ---------- Public API ----------

export interface StarField {
  group: THREE.Group;
  milkyWay: THREE.Mesh;
  dispose: () => void;
}

export async function createStarField(): Promise<StarField> {
  const resp = await fetch('data/stars.json');
  const stars: StarData = await resp.json();

  const group = new THREE.Group();

  const points = createStarPoints(stars);
  group.add(points);

  // Milky Way is NOT added to the group — it stays at the scene root
  // and rotation is applied via shader uniform to avoid double-rotation.
  const milkyWay = createMilkyWay();

  return {
    group,
    milkyWay,
    dispose() {
      points.geometry.dispose();
      (points.material as THREE.Material).dispose();
      milkyWay.geometry.dispose();
      (milkyWay.material as THREE.Material).dispose();
    },
  };
}

/**
 * Rotate stars so they match the sun's position for the current month.
 * Sun's RA advances ~30° per month starting from ~0h at March equinox.
 */
export function updateStarRotation(
  sf: StarField,
  sunOrbitAngle: number,
  monthProgress: number,
): void {
  // Sun ecliptic longitude: 0° at March equinox (month 2), +30°/month
  const eclLon = ((monthProgress - 2) / 12) * 2 * Math.PI;
  const obliquity = 23.44 * (Math.PI / 180);
  // Ecliptic → equatorial RA (accounts for obliquity, not just linear)
  const sunRA = Math.atan2(Math.sin(eclLon) * Math.cos(obliquity), Math.cos(eclLon));

  // Align the star field so the sun's RA matches the sun orb's orbit angle.
  // After THREE.js Y-rotation by θ, a star at RA ends up at world angle -(θ+RA).
  // Sun orb is at world angle sunOrbitAngle, so: -(θ+sunRA) = sunOrbitAngle
  const rotation = -sunOrbitAngle - sunRA;
  sf.group.rotation.y = rotation;
  // Milky Way is a separate scene object — rotation via shader uniform
  const mwMat = sf.milkyWay.material as THREE.ShaderMaterial;
  mwMat.uniforms.rotationY.value = rotation;
}
