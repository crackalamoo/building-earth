/**
 * Load country borders and coastlines from TopoJSON files
 * and create Three.js line objects for globe overlay.
 */

import * as THREE from 'three';
import * as topojson from 'topojson-client';
import type { Topology, GeometryCollection } from 'topojson-specification';

function latLonToVector3(lat: number, lon: number, r: number): THREE.Vector3 {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = lon * (Math.PI / 180);
  return new THREE.Vector3(
    -r * Math.sin(phi) * Math.cos(theta),
    r * Math.cos(phi),
    r * Math.sin(phi) * Math.sin(theta),
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
    opacity: 0.8,
  }));
  const light = new THREE.Line(geom.clone(), new THREE.LineBasicMaterial({
    color: lightColor,
    transparent: true,
    opacity: 0.3,
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

/**
 * Load country borders and coastlines, returning a THREE.Group.
 * Fetches /countries-110m.json and /land-110m.json.
 */
export async function loadBorders(): Promise<THREE.Group> {
  const group = new THREE.Group();
  const radius = 1.002; // slightly above globe surface

  // Country borders
  const response = await fetch('/countries-110m.json');
  const topology = await response.json() as Topology;
  const countries = topology.objects.countries as GeometryCollection;
  const mesh = topojson.mesh(topology, countries);

  if (mesh.type === 'MultiLineString') {
    const lines = processMultiLineString(mesh.coordinates, radius, 0x333333, 0xcccccc);
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
