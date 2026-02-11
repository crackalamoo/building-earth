import * as THREE from 'three';

const ATMOS_RADIUS = 1.15;
const SCALE_HEIGHT = 0.05;

const vertexShader = `
  varying vec3 vWorldPos;
  void main() {
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  uniform vec3 sunDirection;
  varying vec3 vWorldPos;

  const float R = 1.0;
  const float atmosTop = ${ATMOS_RADIUS.toFixed(2)};
  const float H = ${SCALE_HEIGHT.toFixed(2)};

  // Ray-sphere intersection: returns (tNear, tFar), negative if no hit
  vec2 raySphere(vec3 ro, vec3 rd, float radius) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) return vec2(-1.0);
    float s = sqrt(disc);
    return vec2(-b - s, -b + s);
  }

  void main() {
    vec3 rayOrigin = cameraPosition;
    vec3 rayDir = normalize(vWorldPos - cameraPosition);

    // Find ray entry/exit through atmosphere shell
    vec2 tAtmos = raySphere(rayOrigin, rayDir, atmosTop);
    if (tAtmos.x < 0.0 && tAtmos.y < 0.0) discard;

    // Find ray intersection with planet
    vec2 tPlanet = raySphere(rayOrigin, rayDir, R);

    // Determine the atmosphere segment we can see
    float tStart = max(tAtmos.x, 0.0);
    float tEnd = tAtmos.y;

    // If ray hits the planet, clip the segment
    if (tPlanet.x > 0.0) {
      tEnd = min(tEnd, tPlanet.x);
    }

    if (tEnd <= tStart) discard;

    // Sample point: midpoint of visible atmosphere segment
    float tMid = (tStart + tEnd) * 0.5;
    vec3 samplePos = rayOrigin + rayDir * tMid;
    float sampleAlt = length(samplePos) - R;

    // Closest approach of ray to planet center (within visible segment)
    float b = dot(rayOrigin, rayDir);
    float closest = length(rayOrigin + rayDir * clamp(-b, tStart, tEnd));
    float minAlt = closest - R;

    // Column density: path length through atmosphere weighted by density
    float pathLength = tEnd - tStart;
    float columnDensity = pathLength * exp(-max(minAlt, 0.0) / H) * 0.5;

    // Soften peak near the surface
    float peakClamp = smoothstep(0.0, 0.015, minAlt);
    columnDensity = mix(columnDensity * 0.7, columnDensity, peakClamp);

    // Normalize to keep brightness similar to original
    columnDensity = min(columnDensity * 3.0, 1.5);

    float opacity = columnDensity;

    // Sun illumination at the sample point (not the shell surface)
    vec3 sampleDir = normalize(samplePos);
    float sunDot = dot(sampleDir, sunDirection);

    // Rayleigh scattering: blue scatters ~5.5x more than red
    float sunPathLength = 1.0 / max(sunDot, 0.02);
    float scatterScale = 0.18;
    vec3 tau = vec3(scatterScale, scatterScale * 2.5, scatterScale * 5.5);
    vec3 transmittance = exp(-tau * sunPathLength);

    // Scattered light color
    vec3 scatteredColor = vec3(1.0, 0.95, 0.9) * (vec3(1.0) - transmittance);
    float maxC = max(scatteredColor.r, max(scatteredColor.g, scatteredColor.b));
    if (maxC > 0.01) scatteredColor /= maxC;

    // Sunset band: orange/red concentrated at low altitude near the terminator
    float sunsetAngular = smoothstep(-0.1, 0.0, sunDot) * smoothstep(0.3, 0.08, sunDot);
    float sunsetAltitude = exp(-minAlt / (H * 0.4));
    float sunsetStrength = sunsetAngular * sunsetAltitude;
    vec3 sunsetColor = vec3(1.0, 0.35, 0.1);
    scatteredColor = mix(scatteredColor, sunsetColor, sunsetStrength * 0.7);

    // Day/night brightness
    float dayBrightness = smoothstep(-0.2, 0.5, sunDot);
    float nightFloor = 0.3;
    float brightness = mix(nightFloor, 1.0, dayBrightness);

    // Night side: cool blue-purple
    vec3 nightColor = vec3(0.1, 0.15, 0.35);
    vec3 color = mix(nightColor, scatteredColor, dayBrightness);

    float sunsetBoost = 1.0 + sunsetStrength * 0.8;

    gl_FragColor = vec4(color, opacity * brightness * sunsetBoost);
  }
`;

/** Create the atmosphere glow mesh (rendered on BackSide with additive blending). */
export function createAtmosphere(): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(ATMOS_RADIUS, 64, 64);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      sunDirection: { value: new THREE.Vector3(1, 0, 0) },
    },
    vertexShader,
    fragmentShader,
    transparent: true,
    side: THREE.BackSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  return new THREE.Mesh(geometry, material);
}

/** Update the sun direction uniform on an atmosphere mesh. */
export function updateAtmosphereSunDirection(mesh: THREE.Mesh, sunDir: THREE.Vector3): void {
  const material = mesh.material as THREE.ShaderMaterial;
  material.uniforms.sunDirection.value.copy(sunDir);
}
