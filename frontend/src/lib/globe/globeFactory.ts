import * as THREE from 'three';
import { ELEVATION_SCALE, NORMAL_BLEND, sampleElevation, displacedNormal, computeHillshadeGrid } from './elevation';
import type { ClimateLayerData } from './loadBinaryData';

/**
 * Build a globe mesh with per-face vertex colors, optional elevation displacement,
 * and optional per-vertex isOcean attribute.
 *
 * Called once at construction time (not per-frame).
 */
export function createGlobeMesh(
  latCount: number, lonCount: number, radius: number,
  elevationData?: Float32Array, elevNlat?: number, elevNlon?: number,
  landMaskData?: Uint8Array, maskNlat?: number, maskNlon?: number,
): THREE.Mesh {
  const positions: number[] = [];
  const colors: number[] = [];
  const blendedNormals: number[] = [];

  function getVertex(lat: number, lon: number): [number, number, number] {
    let r = radius;
    if (elevationData && elevNlat && elevNlon) {
      const elev = sampleElevation(elevationData, elevNlat, elevNlon, lat, lon);
      r += Math.max(0, elev) * ELEVATION_SCALE;
    }
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = lon * (Math.PI / 180);
    return [
      -r * Math.sin(phi) * Math.cos(theta),
      r * Math.cos(phi),
      r * Math.sin(phi) * Math.sin(theta)
    ];
  }

  function addVertex(lat: number, lon: number, r: number, g: number, b: number, isLand: boolean): void {
    const [x, y, z] = getVertex(lat, lon);
    positions.push(x, y, z);
    colors.push(r, g, b);

    const phi = (90 - lat) * (Math.PI / 180);
    const theta = lon * (Math.PI / 180);
    const snx = -Math.sin(phi) * Math.cos(theta);
    const sny = Math.cos(phi);
    const snz = Math.sin(phi) * Math.sin(theta);

    // Ocean: use pure sphere normals (bathymetry shouldn't affect lighting)
    if (elevationData && elevNlat && elevNlon && isLand) {
      const [dnx, dny, dnz] = displacedNormal(elevationData, elevNlat, elevNlon, lat, lon, radius);
      let nx = snx + NORMAL_BLEND * (dnx - snx);
      let ny = sny + NORMAL_BLEND * (dny - sny);
      let nz = snz + NORMAL_BLEND * (dnz - snz);
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
      blendedNormals.push(nx / len, ny / len, nz / len);
    } else {
      blendedNormals.push(snx, sny, snz);
    }
  }

  const latStep = 180 / latCount;
  const lonStep = 360 / lonCount;

  // Pre-compute per-cell hillshade if elevation data provided
  const hillshadeGrid = (elevationData && elevNlat && elevNlon)
    ? computeHillshadeGrid(elevationData, elevNlat, elevNlon, latCount, lonCount)
    : null;

  for (let i = 0; i < latCount; i++) {
    const lat0 = 90 - i * latStep;
    const lat1 = 90 - (i + 1) * latStep;
    const dataLatIdx = latCount - 1 - i;

    for (let j = 0; j < lonCount; j++) {
      const lon0 = j * lonStep;
      const lon1 = (j + 1) * lonStep;

      const cellIsLand = (landMaskData && maskNlat && maskNlon)
        ? landMaskData[dataLatIdx * maskNlon + j] === 1
        : false;

      // Default color (will be updated immediately)
      const r = 0.1, g = 0.1, b = 0.1;

      addVertex(lat0, lon0, r, g, b, cellIsLand);
      addVertex(lat1, lon0, r, g, b, cellIsLand);
      addVertex(lat1, lon1, r, g, b, cellIsLand);

      addVertex(lat0, lon0, r, g, b, cellIsLand);
      addVertex(lat1, lon1, r, g, b, cellIsLand);
      addVertex(lat0, lon1, r, g, b, cellIsLand);
    }
  }

  // Build per-vertex isOcean attribute if land mask provided
  const isOceanArr: number[] = [];
  if (landMaskData && maskNlat && maskNlon) {
    for (let i = 0; i < latCount; i++) {
      const dataLatIdx = latCount - 1 - i;
      for (let j = 0; j < lonCount; j++) {
        const isLand = landMaskData[dataLatIdx * maskNlon + j] === 1;
        const val = isLand ? 0.0 : 1.0;
        for (let v = 0; v < 6; v++) isOceanArr.push(val);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(blendedNormals, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  if (isOceanArr.length > 0) {
    geometry.setAttribute('isOcean', new THREE.Float32BufferAttribute(isOceanArr, 1));
  }

  const material = new THREE.MeshLambertMaterial({
    vertexColors: true,
    side: THREE.FrontSide,
  });

  const mesh = new THREE.Mesh(geometry, material);
  if (hillshadeGrid) {
    (mesh as any)._hillshadeGrid = hillshadeGrid;
  }
  return mesh;
}

/** Create a simple temperature globe (no elevation, no land mask). */
export function createTemperatureGlobe(climateData: number[][][]): THREE.Mesh {
  const latCount = climateData[0].length;
  const lonCount = climateData[0][0].length;
  return createGlobeMesh(latCount, lonCount, 1);
}

/** Create the ocean specular shader material for the blue marble globe. */
export function createOceanSpecularMaterial(): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: `
      attribute float isOcean;
      varying vec3 vColor;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying float vIsOcean;
      void main() {
        vColor = color;
        vNormal = mat3(modelMatrix) * normal;
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vIsOcean = isOcean;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      uniform vec3 sunDirection;
      uniform float ambientIntensity;
      varying vec3 vColor;
      varying vec3 vNormal;
      varying vec3 vWorldPos;
      varying float vIsOcean;

      void main() {
        vec3 normal = normalize(vNormal);
        float rawNdotL = dot(normal, sunDirection);
        float NdotL = vIsOcean > 0.01
          ? smoothstep(-0.15, 0.3, rawNdotL)
          : max(rawNdotL, 0.0);
        vec3 diffuse = vColor * (ambientIntensity + NdotL * (1.0 - ambientIntensity));

        vec3 viewDir = normalize(cameraPosition - vWorldPos);

        float fresnel = 1.0 - max(dot(viewDir, normal), 0.0);
        fresnel = fresnel * fresnel;
        float fresnelStrength = fresnel * 0.4 * vIsOcean;
        vec3 fresnelColor = vec3(0.5, 0.65, 0.85) * fresnelStrength;

        vec3 specColor = vec3(0.0);
        if (vIsOcean > 0.01) {
          vec3 halfVec = normalize(sunDirection + viewDir);
          float specSharp = pow(max(dot(normal, halfVec), 0.0), 60.0) * 0.6;
          float specBroad = pow(max(dot(normal, halfVec), 0.0), 8.0) * 0.15;
          float specFade = smoothstep(-0.05, 0.15, rawNdotL);
          specColor = vec3(1.0, 0.97, 0.90) * (specSharp + specBroad) * vIsOcean * specFade;
        }

        gl_FragColor = vec4(diffuse + specColor + fresnelColor, 1.0);
      }
    `,
    uniforms: {
      sunDirection: { value: new THREE.Vector3(1, 0, 0) },
      ambientIntensity: { value: 0.15 },
    },
    vertexColors: true,
    side: THREE.FrontSide,
  });
}

/** Create blue marble globe with elevation, land mask, and ocean specular material. */
export function createBlueMarbleGlobe(ld: ClimateLayerData): THREE.Mesh {
  const bmNlat = ld.land_mask.shape[0];
  const bmNlon = ld.land_mask.shape[1];
  const elevData = ld.elevation?.data as Float32Array | undefined;
  const elevNlat = ld.elevation?.shape[0];
  const elevNlon = ld.elevation?.shape[1];
  const landMaskData = ld.land_mask.data as Uint8Array;
  const mesh = createGlobeMesh(bmNlat, bmNlon, 1, elevData, elevNlat, elevNlon, landMaskData, bmNlat, bmNlon);
  mesh.material = createOceanSpecularMaterial();
  return mesh;
}

/** Create the dark primordial globe shown before the reveal animation. */
export function createPrimordialGlobe(landMask: { data: Uint8Array; nlat: number; nlon: number }): THREE.Mesh {
  const { data: mask, nlat: latCount, nlon: lonCount } = landMask;
  const positions: number[] = [];
  const colors: number[] = [];
  const radius = 1;
  const latStep = 180 / latCount;
  const lonStep = 360 / lonCount;

  for (let i = 0; i < latCount; i++) {
    const lat0 = 90 - i * latStep;
    const lat1 = 90 - (i + 1) * latStep;
    const dataLatIdx = latCount - 1 - i;

    for (let j = 0; j < lonCount; j++) {
      const lon0 = j * lonStep;
      const lon1 = (j + 1) * lonStep;
      const isLand = mask[dataLatIdx * lonCount + j] === 1;
      const [r, g, b] = isLand ? [0.003, 0.003, 0.0025] : [0.001, 0.001, 0.003];

      for (const [la, lo] of [[lat0, lon0], [lat1, lon0], [lat1, lon1], [lat0, lon0], [lat1, lon1], [lat0, lon1]]) {
        const phi = (90 - la) * (Math.PI / 180);
        const theta = lo * (Math.PI / 180);
        positions.push(
          -radius * Math.sin(phi) * Math.cos(theta),
          radius * Math.cos(phi),
          radius * Math.sin(phi) * Math.sin(theta)
        );
        colors.push(r, g, b);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshBasicMaterial({ vertexColors: true });
  return new THREE.Mesh(geometry, material);
}
