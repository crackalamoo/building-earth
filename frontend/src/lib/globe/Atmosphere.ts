import * as THREE from 'three';

const ATMOS_RADIUS = 1.15;
const SCALE_HEIGHT = 0.05;

const vertexShader = `
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;
  void main() {
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    vWorldNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const fragmentShader = `
  uniform vec3 sunDirection;
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;

  const float R = 1.0;
  const float atmosTop = ${ATMOS_RADIUS.toFixed(2)};
  const float H = ${SCALE_HEIGHT.toFixed(2)};

  void main() {
    vec3 rayDir = normalize(vWorldPos - cameraPosition);

    // Closest approach of viewing ray to planet center
    float b = dot(cameraPosition, rayDir);
    float closest = length(cameraPosition - b * rayDir);
    float minAlt = closest - R;

    // Column density falls off exponentially with altitude
    float columnDensity = exp(-max(minAlt, 0.0) / H);

    // Soften peak near the surface
    float peakClamp = smoothstep(0.0, 0.015, minAlt);
    columnDensity = mix(columnDensity * 0.7, columnDensity, peakClamp);

    // Fade beyond atmosphere top
    float outerFade = smoothstep(atmosTop, R, closest);

    if (minAlt < -0.001) discard;

    float opacity = columnDensity * outerFade;

    // Sun illumination
    vec3 pointDir = normalize(vWorldPos);
    float sunDot = dot(pointDir, sunDirection);

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
