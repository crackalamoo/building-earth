<script lang="ts">
  import type { ClimateLayerData } from './loadBinaryData';

  export let lat: number;
  export let lon: number;
  export let screenX: number;
  export let screenY: number;
  export let monthProgress: number;
  export let temperatureData: number[][][] | null;
  export let layerData: ClimateLayerData | null;

  function sampleT2m(lat: number, lon: number): number {
    if (!temperatureData) return 0;
    const nlat = temperatureData[0].length;
    const nlon = temperatureData[0][0].length;
    const latIdx = (lat + 90) / 180 * nlat - 0.5;
    const lonNorm = ((lon % 360) + 360) % 360;
    const lonIdx = lonNorm / 360 * nlon;

    const m0 = Math.floor(monthProgress) % 12;
    const m1 = (m0 + 1) % 12;
    const t = monthProgress - Math.floor(monthProgress);

    function sample(month: number): number {
      const li = Math.max(0, Math.min(nlat - 1, Math.floor(latIdx)));
      const li2 = Math.min(nlat - 1, li + 1);
      const lj = Math.floor(((lonIdx % nlon) + nlon) % nlon);
      const lj2 = (lj + 1) % nlon;
      const ft = latIdx - li;
      const fl = lonIdx - lj;
      const v00 = temperatureData![month][li][lj];
      const v01 = temperatureData![month][li][lj2];
      const v10 = temperatureData![month][li2][lj];
      const v11 = temperatureData![month][li2][lj2];
      return (1 - ft) * ((1 - fl) * v00 + fl * v01) + ft * ((1 - fl) * v10 + fl * v11);
    }

    return (1 - t) * sample(m0) + t * sample(m1);
  }

  function sampleOceanInfo(lat: number, lon: number): { isOcean: boolean; sstC: number | null } {
    if (!layerData?.land_mask_native || !layerData?.surface) return { isOcean: false, sstC: null };
    const lmData = layerData.land_mask_native.data as Uint8Array;
    const nlat = layerData.land_mask_native.shape[0];
    const nlon = layerData.land_mask_native.shape[1];
    const li = Math.max(0, Math.min(nlat - 1, Math.round((lat + 90) / 180 * nlat - 0.5)));
    const lonNorm = ((lon % 360) + 360) % 360;
    const lj = Math.floor(lonNorm / 360 * nlon) % nlon;
    const isOcean = lmData[li * nlon + lj] === 0;
    if (!isOcean) return { isOcean: false, sstC: null };

    const surfData = layerData.surface.data as Float32Array;
    const sNlat = layerData.surface.shape[1];
    const sNlon = layerData.surface.shape[2];
    const m0 = Math.floor(monthProgress) % 12;
    const m1 = (m0 + 1) % 12;
    const t = monthProgress - Math.floor(monthProgress);
    const si = Math.max(0, Math.min(sNlat - 1, Math.round((lat + 90) / 180 * sNlat - 0.5)));
    const sj = Math.floor(lonNorm / 360 * sNlon) % sNlon;
    const v0 = surfData[m0 * sNlat * sNlon + si * sNlon + sj];
    const v1 = surfData[m1 * sNlat * sNlon + si * sNlon + sj];
    return { isOcean: true, sstC: (1 - t) * v0 + t * v1 };
  }

  function sampleWind(lat: number, lon: number): { speed: number; dir: number } {
    if (!layerData?.wind_u_10m || !layerData?.wind_v_10m) return { speed: 0, dir: 0 };
    const uData = layerData.wind_u_10m.data as Float32Array;
    const vData = layerData.wind_v_10m.data as Float32Array;
    const nlat = layerData.wind_u_10m.shape[1];
    const nlon = layerData.wind_u_10m.shape[2];
    const n = nlat * nlon;
    const li = Math.max(0, Math.min(nlat - 1, Math.round((lat + 90) / 180 * nlat - 0.5)));
    const lonNorm = ((lon % 360) + 360) % 360;
    const lj = Math.floor(lonNorm / 360 * nlon) % nlon;
    const m0 = Math.floor(monthProgress) % 12;
    const m1 = (m0 + 1) % 12;
    const t = monthProgress - Math.floor(monthProgress);
    const u = (1 - t) * uData[m0 * n + li * nlon + lj] + t * uData[m1 * n + li * nlon + lj];
    const v = (1 - t) * vData[m0 * n + li * nlon + lj] + t * vData[m1 * n + li * nlon + lj];
    const speed = Math.sqrt(u * u + v * v);
    const dir = (Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360;
    return { speed, dir };
  }

  // Reference monthProgress explicitly so Svelte re-runs when it changes
  function sampleElevation(lat: number, lon: number): number | null {
    if (!layerData?.elevation) return null;
    const data = layerData.elevation.data as Float32Array;
    const nlat = layerData.elevation.shape[0];
    const nlon = layerData.elevation.shape[1];
    const li = Math.max(0, Math.min(nlat - 1, Math.round((lat + 90) / 180 * nlat - 0.5)));
    const lonNorm = ((lon % 360) + 360) % 360;
    const lj = Math.floor(lonNorm / 360 * nlon) % nlon;
    return data[li * nlon + lj];
  }

  $: elevation = sampleElevation(lat, lon);
  $: tempC = (monthProgress, sampleT2m(lat, lon));
  $: ocean = (monthProgress, sampleOceanInfo(lat, lon));
  $: wind = (monthProgress, sampleWind(lat, lon));
</script>

<div class="inspect-popup" style="left: {screenX + 12}px; top: {screenY - 40}px;">
  <div class="inspect-coords">{Math.abs(lat).toFixed(1)}°{lat >= 0 ? 'N' : 'S'}, {Math.abs(lon).toFixed(1)}°{lon >= 0 ? 'E' : 'W'}</div>
  <div class="inspect-stat">{tempC.toFixed(1)}°C</div>
  {#if ocean.isOcean && ocean.sstC !== null}
    <div class="inspect-stat inspect-sst">SST: {ocean.sstC.toFixed(1)}°C</div>
  {/if}
  {#if elevation !== null}
    <div class="inspect-stat inspect-elev">{ocean.isOcean ? 'Depth' : 'Elev'}: {ocean.isOcean ? Math.abs(Math.round(elevation)).toLocaleString() : Math.round(elevation).toLocaleString()} m</div>
  {/if}
  <div class="inspect-stat inspect-wind">
    <span class="wind-arrow" style="transform: rotate({wind.dir}deg)">↓</span>
    {(wind.speed * 3.6).toFixed(0)} km/h
  </div>
</div>

<style>
  .inspect-popup {
    position: fixed;
    background: rgba(0, 0, 0, 0.85);
    border: 1px solid #00e5ff;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    pointer-events: none;
    z-index: 100;
    font-size: 0.85rem;
    line-height: 1.4;
    white-space: nowrap;
  }

  .inspect-coords {
    color: #aaa;
  }

  .inspect-stat {
    color: #fff;
  }

  .inspect-sst {
    color: #6ec6ff;
  }

  .inspect-elev {
    color: #aaa;
  }

  .inspect-wind {
    color: #ccc;
    display: flex;
    align-items: center;
    gap: 0.3rem;
  }

  .wind-arrow {
    display: inline-block;
    font-size: 1rem;
    line-height: 1;
  }
</style>
