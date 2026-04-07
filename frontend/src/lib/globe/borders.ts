/**
 * Load country borders and coastlines from TopoJSON files
 * and create Three.js line objects for globe overlay.
 */

import * as THREE from 'three';
import * as topojson from 'topojson-client';
import type { Topology, GeometryCollection } from 'topojson-specification';
import { ELEVATION_SCALE, sampleElevation } from './elevation';

/** Shared elevation state set before building lines. */
let _elevData: Float32Array | null = null;
let _elevNlat = 0;
let _elevNlon = 0;

/**
 * Sample the maximum elevation in a small neighborhood around (lat, lon).
 * Border vertices use this so they sit above any nearby peak — a plain
 * point sample can dip under the displaced mesh face on the back side
 * of a mountain, even when the offset above the surface is generous.
 */
function sampleMaxElevation(lat: number, lon: number): number {
  if (!_elevData) return 0;
  const dLat = 180 / _elevNlat;
  const dLon = 360 / _elevNlon;
  let maxElev = -Infinity;
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      const la = Math.max(-89.9, Math.min(89.9, lat + di * dLat));
      const lo = ((lon + dj * dLon) % 360 + 360) % 360;
      const e = sampleElevation(_elevData, _elevNlat, _elevNlon, la, lo);
      if (e > maxElev) maxElev = e;
    }
  }
  return maxElev;
}

function latLonToVector3(lat: number, lon: number, r: number): THREE.Vector3 {
  let radius = r;
  if (_elevData) {
    const elev = sampleMaxElevation(lat, lon);
    radius += Math.max(0, elev) * ELEVATION_SCALE;
  }
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = lon * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

function createLineFromCoords(
  coords: number[][], r: number, darkColor: number, lightColor: number,
): THREE.Line[] {
  const points: THREE.Vector3[] = [];
  for (const [lon, lat] of coords) {
    points.push(latLonToVector3(lat, lon, r));
  }
  const geom = new THREE.BufferGeometry().setFromPoints(points);
  const dark = new THREE.Line(geom, new THREE.LineBasicMaterial({
    color: darkColor,
    transparent: true,
    opacity: 1.0,
  }));
  const light = new THREE.Line(geom.clone(), new THREE.LineBasicMaterial({
    color: lightColor,
    transparent: true,
    opacity: 0.35,
  }));
  return [dark, light];
}

function processMultiLineString(
  coords: number[][][], r: number, darkColor: number, lightColor: number,
): THREE.Line[] {
  return coords.flatMap(lineCoords => createLineFromCoords(lineCoords, r, darkColor, lightColor));
}

function processPolygon(
  coords: number[][][], r: number, darkColor: number, lightColor: number,
): THREE.Line[] {
  return coords.flatMap(ring => createLineFromCoords(ring, r, darkColor, lightColor));
}

function processMultiPolygon(
  coords: number[][][][], r: number, darkColor: number, lightColor: number,
): THREE.Line[] {
  const lines: THREE.Line[] = [];
  for (const polygon of coords) {
    lines.push(...processPolygon(polygon, r, darkColor, lightColor));
  }
  return lines;
}

async function buildBordersAtRadius(radius: number): Promise<THREE.Group> {
  const group = new THREE.Group();

  // Country borders
  const response = await fetch('/countries-110m.json');
  const topology = await response.json() as Topology;
  const countries = topology.objects.countries as GeometryCollection;
  const mesh = topojson.mesh(topology, countries);

  if (mesh.type === 'MultiLineString') {
    const lines = processMultiLineString(mesh.coordinates, radius, 0x000000, 0xcccccc);
    lines.forEach(line => group.add(line));
  }

  // Coastlines
  const landResponse = await fetch('/land-110m.json');
  const landTopology = await landResponse.json() as Topology;
  const land = landTopology.objects.land as GeometryCollection;
  const landFeature = topojson.feature(landTopology, land);

  const features = 'features' in landFeature
    ? landFeature.features
    : [landFeature as GeoJSON.Feature];
  for (const feature of features) {
    const geom = feature.geometry;
    if (geom.type === 'Polygon') {
      const lines = processPolygon(geom.coordinates, radius, 0x000000, 0xffffff);
      lines.forEach(line => group.add(line));
    } else if (geom.type === 'MultiPolygon') {
      const lines = processMultiPolygon(geom.coordinates, radius, 0x000000, 0xffffff);
      lines.forEach(line => group.add(line));
    }
  }

  return group;
}

/**
 * Load country borders and coastlines, returning two THREE.Groups: one
 * built without elevation displacement (for the smooth temperature /
 * precipitation globes) and one built on top of the displaced terrain
 * (for blue marble). Globe.svelte toggles visibility based on the
 * active layer.
 */
export async function loadBorders(
  elevData?: Float32Array, elevNlat?: number, elevNlon?: number,
): Promise<{ flat: THREE.Group; terrain: THREE.Group }> {
  // Flat variant: no elevation sampling (smooth sphere underneath).
  _elevData = null;
  _elevNlat = 0;
  _elevNlon = 0;
  const flat = await buildBordersAtRadius(1.002);

  // Terrain-aware variant: lifts each vertex above the displaced mesh.
  if (elevData && elevNlat && elevNlon) {
    _elevData = elevData;
    _elevNlat = elevNlat;
    _elevNlon = elevNlon;
  }
  const terrain = await buildBordersAtRadius(1.002);

  // Reset shared state so future calls without elevation start clean.
  _elevData = null;
  _elevNlat = 0;
  _elevNlon = 0;

  return { flat, terrain };
}
