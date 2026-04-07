<script lang="ts">
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import { fly } from 'svelte/transition';
  import type { ClimateLayerData } from './loadBinaryData';
  import { useImperial } from './stores';
  import { computeSuggestions, streamChat } from './chatUtils';
  import type { ChatMessage, MsgPart } from './chatUtils';
  import { renderMarkdown } from './renderMarkdown';
  import { temperatureToColor } from './globe/colormap';
  import { precipMmdayToColor } from './globe/precipitationColormap';

  const dispatch = createEventDispatcher();

  export let lat: number;
  export let lon: number;
  export let monthProgress: number;
  export let temperatureData: number[][][] | null;
  export let layerData: ClimateLayerData | null;
  export let stage: number = 5;

  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
  const MAX_MESSAGES = 50;

  function tempCss(c: number): string { const [r,g,b] = temperatureToColor(c); return `rgb(${r},${g},${b})`; }
  function precipCss(mmMonth: number): string { const [r,g,b] = precipMmdayToColor(mmMonth / 30.44); return `rgb(${r},${g},${b})`; }

  // Chat state
  let messages: ChatMessage[] = [];
  let inputText = '';
  let streaming = false;
  let limitReached = false;
  let errorOccurred = false;
  let lastFailedMessage = '';
  let chatContainer: HTMLDivElement;
  let currentParts: MsgPart[] = [];
  let sentLat: number | null = null;
  let sentLon: number | null = null;
  let sentMonth: number | null = null;

  $: suggestions = obsLoaded
    ? computeSuggestions(lat, ocean, elevation, cycleTemps, cyclePrecip, wind, currentMonthIdx, tempC, stage, obsTemps, obsPrecips)
    : [];

  let activeTab: 'cycle' | 'ask' = 'ask';
  let hoveredMonthIdx: number | null = null;

  // ── Grid index helpers ──

  function latIndex(lat: number, nlat: number): number {
    return Math.max(0, Math.min(nlat - 1, Math.round((lat + 90) / 180 * nlat - 0.5)));
  }

  function lonIndex(lon: number, nlon: number): number {
    const lonNorm = ((lon % 360) + 360) % 360;
    return Math.floor(lonNorm / 360 * nlon) % nlon;
  }

  function monthLerp(): { m0: number; m1: number; t: number } {
    const m0 = Math.floor(monthProgress) % 12;
    return { m0, m1: (m0 + 1) % 12, t: monthProgress - Math.floor(monthProgress) };
  }

  // ── Sampling functions ──

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
    const li = latIndex(lat, nlat);
    const lj = lonIndex(lon, nlon);
    if (lmData[li * nlon + lj] !== 0) return { isOcean: false, sstC: null };
    const surfData = layerData.surface.data as Float32Array;
    const sNlat = layerData.surface.shape[1];
    const sNlon = layerData.surface.shape[2];
    const { m0, m1, t } = monthLerp();
    const si = latIndex(lat, sNlat);
    const sj = lonIndex(lon, sNlon);
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
    const li = latIndex(lat, nlat);
    const lj = lonIndex(lon, nlon);
    const { m0, m1, t } = monthLerp();
    const u = (1 - t) * uData[m0 * n + li * nlon + lj] + t * uData[m1 * n + li * nlon + lj];
    const v = (1 - t) * vData[m0 * n + li * nlon + lj] + t * vData[m1 * n + li * nlon + lj];
    const speed = Math.sqrt(u * u + v * v);
    const dir = (Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360;
    return { speed, dir };
  }

  function sampleElevation(lat: number, lon: number): number | null {
    if (!layerData?.elevation) return null;
    const data = layerData.elevation.data as Float32Array;
    const nlat = layerData.elevation.shape[0];
    const nlon = layerData.elevation.shape[1];
    return data[latIndex(lat, nlat) * nlon + lonIndex(lon, nlon)];
  }

  /** High-res land check (720×1440) — matches the elevation grid. */
  function isLandHighRes(lat: number, lon: number): boolean {
    if (!layerData?.land_mask) return true;
    const data = layerData.land_mask.data as Uint8Array;
    const nlat = layerData.land_mask.shape[0];
    const nlon = layerData.land_mask.shape[1];
    return data[latIndex(lat, nlat) * nlon + lonIndex(lon, nlon)] !== 0;
  }

  // ── Annual cycle data ──

  function sampleT2mMonth(lat: number, lon: number, month: number): number {
    if (!temperatureData) return 0;
    const nlat = temperatureData[0].length;
    const nlon = temperatureData[0][0].length;
    const latIdx = (lat + 90) / 180 * nlat - 0.5;
    const lonNorm = ((lon % 360) + 360) % 360;
    const lonIdx = lonNorm / 360 * nlon;
    const li = Math.max(0, Math.min(nlat - 1, Math.floor(latIdx)));
    const li2 = Math.min(nlat - 1, li + 1);
    const lj = Math.floor(((lonIdx % nlon) + nlon) % nlon);
    const lj2 = (lj + 1) % nlon;
    const ft = latIdx - li;
    const fl = lonIdx - lj;
    const v00 = temperatureData[month][li][lj];
    const v01 = temperatureData[month][li][lj2];
    const v10 = temperatureData[month][li2][lj];
    const v11 = temperatureData[month][li2][lj2];
    return (1 - ft) * ((1 - fl) * v00 + fl * v01) + ft * ((1 - fl) * v10 + fl * v11);
  }

  function samplePrecipMonth(lat: number, lon: number, month: number): number {
    if (!layerData?.precipitation) return 0;
    const data = layerData.precipitation.data as Float32Array;
    const nlat = layerData.precipitation.shape[1];
    const nlon = layerData.precipitation.shape[2];
    const n = nlat * nlon;
    const li = latIndex(lat, nlat);
    const lj = lonIndex(lon, nlon);
    // Convert kg/m²/s to mm/month (approx 30.44 days/month)
    return data[month * n + li * nlon + lj] * 86400 * 30.44;
  }

  $: cycleTemps = Array.from({ length: 12 }, (_, m) => sampleT2mMonth(lat, lon, m));
  $: cyclePrecip = Array.from({ length: 12 }, (_, m) => samplePrecipMonth(lat, lon, m));

  // Obs data (fetched from backend on location change)
  let obsTemps: (number | null)[] = Array(12).fill(null);
  let obsPrecips: (number | null)[] = Array(12).fill(null); // mm/month (already converted by backend)
  let obsLoaded = false;

  $: fetchObs(lat, lon);

  async function fetchObs(lat: number, lon: number) {
    obsLoaded = false;
    try {
      const res = await fetch(`${API_BASE}/api/obs?lat=${lat}&lon=${lon}`);
      const data = await res.json();
      obsTemps = data.temps;
      // Backend returns mm/month already (converted via _to_display_units)
      obsPrecips = data.precips;
    } catch { /* leave previous values */ }
    obsLoaded = true;
  }

  $: obsTempMin = Math.min(...(obsTemps.filter(v => v !== null) as number[]));
  $: obsTempMax = Math.max(...(obsTemps.filter(v => v !== null) as number[]));
  $: obsPrecipPeak = Math.max(...(obsPrecips.filter(v => v !== null) as number[]), 0);

  $: tempMin = Math.min(Math.min(...cycleTemps), isFinite(obsTempMin) ? obsTempMin : Infinity);
  $: tempMax = Math.max(Math.max(...cycleTemps), isFinite(obsTempMax) ? obsTempMax : -Infinity);
  $: precipMax = Math.max(Math.max(...cyclePrecip, 1), obsPrecipPeak, 20) * 1.25; // min 20mm/mo

  $: currentMonthIdx = Math.floor(monthProgress) % 12;
  $: displayMonthIdx = hoveredMonthIdx !== null ? hoveredMonthIdx : currentMonthIdx;

  const CHART_W = 580;
  const CHART_H = 180;
  const PAD_L = 36;
  const PAD_R = 12;
  const PAD_T = 38;
  const PAD_B = 28;
  const INNER_W = CHART_W - PAD_L - PAD_R;
  const INNER_H = CHART_H - PAD_T - PAD_B;
  const TEMP_PAD = 0.15; // fraction of range to pad top/bottom
  const BAR_W = INNER_W / 12 * 0.6;
  const MONTH_SHORT = ['J','F','M','A','M','J','J','A','S','O','N','D'];

  $: tempSpan = Math.max(tempMax - tempMin, 10); // minimum 10°C range
  $: tempPadded = tempSpan * TEMP_PAD;
  $: tempMid = (tempMax + tempMin) / 2;
  $: tempPlotMin = tempMid - tempSpan / 2 - tempPadded;
  $: tempPlotMax = tempMid + tempSpan / 2 + tempPadded;
  $: tempRange = tempPlotMax - tempPlotMin;
  $: chartBars = cyclePrecip.map((p, i) => ({
    x: PAD_L + (i + 0.2) * (INNER_W / 12),
    y: PAD_T + INNER_H - (p / precipMax) * INNER_H,
    h: (p / precipMax) * INNER_H,
    p,
    active: i === displayMonthIdx,
  }));
  $: chartLines = cycleTemps.slice(0, 11).map((_, i) => ({
    x1: PAD_L + (i + 0.5) * (INNER_W / 12),
    y1: PAD_T + INNER_H - ((cycleTemps[i] - tempPlotMin) / tempRange) * INNER_H,
    x2: PAD_L + (i + 1.5) * (INNER_W / 12),
    y2: PAD_T + INNER_H - ((cycleTemps[i + 1] - tempPlotMin) / tempRange) * INNER_H,
    tAvg: (cycleTemps[i] + cycleTemps[i + 1]) / 2,
  }));
  $: chartDots = cycleTemps.map((t, i) => ({
    cx: PAD_L + (i + 0.5) * (INNER_W / 12),
    cy: PAD_T + INNER_H - ((t - tempPlotMin) / tempRange) * INNER_H,
    t,
    active: i === displayMonthIdx,
    isCurrent: i === currentMonthIdx,
  }));
  $: chartMonths = MONTH_SHORT.map((label, i) => ({
    x: PAD_L + (i + 0.5) * (INNER_W / 12),
    label,
    active: i === displayMonthIdx,
    isCurrent: i === currentMonthIdx,
  }));
  $: hitAreas = Array.from({ length: 12 }, (_, i) => ({
    x: PAD_L + i * (INNER_W / 12),
    i,
  }));

  // Obs chart geometry (same scale as sim chart)
  $: obsChartBars = obsPrecips.map((p, i) => ({
    x: PAD_L + (i + 0.2) * (INNER_W / 12),
    y: PAD_T + INNER_H - ((p ?? 0) / precipMax) * INNER_H,
    h: ((p ?? 0) / precipMax) * INNER_H,
    valid: p !== null,
  }));
  $: obsChartLines = obsTemps.slice(0, 11).map((_, i) => ({
    x1: PAD_L + (i + 0.5) * (INNER_W / 12),
    y1: PAD_T + INNER_H - ((obsTemps[i]! - tempPlotMin) / tempRange) * INNER_H,
    x2: PAD_L + (i + 1.5) * (INNER_W / 12),
    y2: PAD_T + INNER_H - ((obsTemps[i + 1]! - tempPlotMin) / tempRange) * INNER_H,
    valid: obsTemps[i] !== null && obsTemps[i + 1] !== null,
  }));
  $: obsChartDots = obsTemps.map((t, i) => ({
    cx: PAD_L + (i + 0.5) * (INNER_W / 12),
    cy: t !== null ? PAD_T + INNER_H - ((t - tempPlotMin) / tempRange) * INNER_H : 0,
    t,
    valid: t !== null,
  }));

  function toggleUnits() { useImperial.update(v => !v); }
  function cToF(c: number): number { return c * 9 / 5 + 32; }
  function mToFt(m: number): number { return m * 3.28084; }
  function kphToMph(kph: number): number { return kph * 0.621371; }

  $: imp = $useImperial;
  $: elevation = sampleElevation(lat, lon);
  $: isLand = isLandHighRes(lat, lon);
  $: tempC = (monthProgress, sampleT2m(lat, lon));
  $: ocean = (monthProgress, sampleOceanInfo(lat, lon));
  $: wind = (monthProgress, sampleWind(lat, lon));

  // ── Chat functions ──

  let abortController: AbortController | null = null;

  function scrollToBottom() {
    if (chatContainer) {
      requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
      });
    }
  }

  async function sendMessage(text: string) {
    if (streaming || limitReached || !text.trim()) return;

    messages = [...messages, { role: 'user' as const, content: text.trim() }];
    inputText = '';
    streaming = true;
    errorOccurred = false;

    if (messages.filter(m => m.role === 'user').length >= MAX_MESSAGES) {
      limitReached = true;
    }

    messages = [...messages, { role: 'assistant' as const, content: '' }];
    currentParts = [];
    scrollToBottom();

    const month = Math.floor(monthProgress) % 12;
    try {
      abortController = new AbortController();
      await streamChat(
        API_BASE,
        {
          lat, lon, month,
          prevLat: sentLat, prevLon: sentLon, prevMonth: sentMonth,
          imperial: $useImperial,
          stage,
          messages: messages.filter(m => m.content !== '').map(m => ({ role: m.role, content: m.content })),
        },
        abortController.signal,
        {
          onPart(parts) {
            currentParts = parts;
            messages[messages.length - 1].parts = currentParts;
            messages = [...messages];
            scrollToBottom();
          },
          onContent(chunk) {
            messages[messages.length - 1].content += chunk;
          },
          onError(msg) {
            messages[messages.length - 1].content += msg;
          },
          onDone() {},
        }
      );
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        const last = messages[messages.length - 1];
        if (last.role === 'assistant' && !last.content) {
          last.content = 'Failed to connect to the explanation server.';
          messages = [...messages];
        }
        errorOccurred = true;
        lastFailedMessage = messages.length >= 2 ? messages[messages.length - 2].content : '';
      }
    } finally {
      streaming = false;
      abortController = null;
      sentLat = lat;
      sentLon = lon;
      sentMonth = month;
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputText);
    }
  }

  function retryLastMessage() {
    messages = messages.slice(0, -2);
    errorOccurred = false;
    sendMessage(lastFailedMessage);
  }

  function close() {
    abortController?.abort();
    dispatch('close');
  }

  onDestroy(() => {
    abortController?.abort();
  });

  const MONTH_NAMES = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
  ];
  $: displayMonth = MONTH_NAMES[Math.floor(monthProgress) % 12];

  let isMobile = typeof window !== 'undefined' && window.matchMedia('(max-width: 800px)').matches;

  // ── Drag-to-dismiss (mobile) ──
  let panelEl: HTMLDivElement;
  let dragHandleEl: HTMLDivElement;
  let dragStartY = 0;
  let dragOffsetY = 0;
  let dragging = false;

  function onDragStart(e: TouchEvent | MouseEvent) {
    if (!isMobile) return;
    // Allow taps on interactive children (e.g. the close button) to bubble
    // through normally instead of being captured as a drag.
    const target = e.target as HTMLElement | null;
    if (target?.closest('[data-no-drag]')) return;
    dragging = true;
    dragOffsetY = 0;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragStartY = clientY;
    // Prevent text selection during drag
    e.preventDefault();
  }

  function onDragMove(e: TouchEvent | MouseEvent) {
    if (!dragging) return;
    // Block iOS scroll so the panel can follow the finger
    if ('touches' in e && e.cancelable) e.preventDefault();
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragOffsetY = Math.max(0, clientY - dragStartY); // only allow dragging down
    // iOS Safari doesn't visually update `transform` on this fixed element
    // (verified empirically — the inline style updates and computed style
    // shows it, but no repaint happens). Animating `bottom` instead works.
    if (panelEl) panelEl.style.bottom = `${-dragOffsetY}px`;
  }

  function onDragEnd() {
    if (!dragging) return;
    dragging = false;
    if (dragOffsetY > 100) {
      // Dismiss threshold reached
      close();
    } else {
      // Snap back
      if (panelEl) panelEl.style.bottom = '';
      dragOffsetY = 0;
    }
  }

  // ── Click edge to close (desktop) ──
  function onPanelClick(e: MouseEvent) {
    if (isMobile) return;
    // Close if clicking within 8px of the left edge
    const rect = panelEl?.getBoundingClientRect();
    if (rect && e.clientX - rect.left < 8) {
      close();
    }
  }

  function addDragListeners() {
    // touchstart on the drag handle MUST be {passive: false} so we can
    // call e.preventDefault() and stop iOS from interpreting the gesture
    // as a scroll. Svelte's on:touchstart attaches as passive, so we
    // bypass it with a manual addEventListener on the bound handle.
    if (dragHandleEl) {
      dragHandleEl.addEventListener('touchstart', onDragStart, { passive: false });
    }
    window.addEventListener('touchmove', onDragMove, { passive: false });
    window.addEventListener('touchend', onDragEnd);
    window.addEventListener('touchcancel', onDragEnd);
    window.addEventListener('mousemove', onDragMove);
    window.addEventListener('mouseup', onDragEnd);
  }

  function removeDragListeners() {
    if (dragHandleEl) {
      dragHandleEl.removeEventListener('touchstart', onDragStart);
    }
    window.removeEventListener('touchmove', onDragMove);
    window.removeEventListener('touchend', onDragEnd);
    window.removeEventListener('touchcancel', onDragEnd);
    window.removeEventListener('mousemove', onDragMove);
    window.removeEventListener('mouseup', onDragEnd);
  }

  let prevMobile = isMobile;
  $: {
    if (typeof window !== 'undefined' && prevMobile !== isMobile) {
      if (isMobile) addDragListeners();
      else removeDragListeners();
      prevMobile = isMobile;
    }
  }

  onMount(() => {
    if (isMobile) addDragListeners();

    const mql = window.matchMedia('(max-width: 800px)');
    const onMediaChange = (e: MediaQueryListEvent) => { isMobile = e.matches; };
    mql.addEventListener('change', onMediaChange);

    return () => {
      removeDragListeners();
      mql.removeEventListener('change', onMediaChange);
    };
  });
</script>

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div
  class="inspect-panel"
  class:dragging
  bind:this={panelEl}
  transition:fly={{ x: isMobile ? 0 : 700, y: isMobile ? 400 : 0, duration: 200 }}
  on:click={onPanelClick}
>
  {#if isMobile}
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="drag-area" bind:this={dragHandleEl} on:mousedown={onDragStart}>
      <div class="drag-handle">
        <div class="drag-handle-pill"></div>
      </div>
      <div class="panel-header">
        <div class="coords">
          {Math.abs(lat).toFixed(1)}°{lat >= 0 ? 'N' : 'S'}, {Math.abs(lon).toFixed(1)}°{lon >= 0 ? 'E' : 'W'}
          <span class="month-tag">{displayMonth}</span>
        </div>
        <button class="close-btn" data-no-drag on:click|stopPropagation={close}>×</button>
      </div>
    </div>
  {:else}
    <div class="panel-header">
      <div class="coords">
        {Math.abs(lat).toFixed(1)}°{lat >= 0 ? 'N' : 'S'}, {Math.abs(lon).toFixed(1)}°{lon >= 0 ? 'E' : 'W'}
        <span class="month-tag">{displayMonth}</span>
      </div>
      <button class="close-btn" on:click|stopPropagation={close}>×</button>
    </div>
  {/if}

  <div class="stats">
    <div class="stat">
      <span class="unit-toggle" role="button" tabindex="0" on:click|stopPropagation={toggleUnits}>
        {imp ? cToF(tempC).toFixed(1) + '°F' : tempC.toFixed(1) + '°C'}
      </span>
    </div>
    {#if ocean.isOcean && ocean.sstC !== null}
      <div class="stat sst">
        Sea: <span class="unit-toggle" role="button" tabindex="0" on:click|stopPropagation={toggleUnits}>
          {imp ? cToF(ocean.sstC).toFixed(1) + '°F' : ocean.sstC.toFixed(1) + '°C'}
        </span>
      </div>
    {/if}
    {#if elevation !== null}
      <div class="stat elev">
        {isLand ? 'Elev' : 'Depth'}: <span class="unit-toggle" role="button" tabindex="0" on:click|stopPropagation={toggleUnits}>
          {#if imp}
            {isLand ? Math.round(mToFt(elevation)).toLocaleString() : Math.abs(Math.round(mToFt(elevation))).toLocaleString()} ft
          {:else}
            {isLand ? Math.round(elevation).toLocaleString() : Math.abs(Math.round(elevation)).toLocaleString()} m
          {/if}
        </span>
      </div>
    {/if}
    <div class="stat wind-stat">
      <span class="wind-arrow" style="transform: rotate({wind.dir}deg)">↓</span>
      <span class="unit-toggle" role="button" tabindex="0" on:click|stopPropagation={toggleUnits}>
        {#if imp}
          {kphToMph(wind.speed * 3.6).toFixed(0)} mph
        {:else}
          {(wind.speed * 3.6).toFixed(0)} km/h
        {/if}
      </span>
    </div>
  </div>

  <div class="tab-bar">
    <div class="tab-group">
      <button
        class="tab-btn"
        class:active={activeTab === 'ask'}
        on:click|stopPropagation={() => activeTab = 'ask'}
      >Ask</button><button
        class="tab-btn"
        class:active={activeTab === 'cycle'}
        on:click|stopPropagation={() => activeTab = 'cycle'}
      >Charts</button>
    </div>
  </div>

  {#if activeTab === 'cycle'}
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="cycle-area" on:mouseleave={() => hoveredMonthIdx = null}>
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <svg
        viewBox="0 0 {CHART_W} {CHART_H}" width="100%" preserveAspectRatio="xMidYMid meet" class="cycle-chart"
      >
        <!-- Precip bars -->
        {#each chartBars as bar}
          {@const minH = 2}
          {@const displayH = Math.max(bar.h, minH)}
          {@const displayY = PAD_T + INNER_H - displayH}
          {@const c = precipCss(bar.p)}
          <rect
            x={bar.x} y={displayY} width={BAR_W} height={displayH}
            fill={c}
            opacity={bar.active ? 0.6 : 0.4}
            stroke={bar.active ? 'rgba(255,255,255,0.6)' : 'none'}
            stroke-width="1"
          />
        {/each}

        <!-- Temp line segment gradients -->
        <defs>
          {#each chartLines as seg, i}
            <linearGradient id="tgrad-sim-{i}" x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2} gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color={tempCss(cycleTemps[i])} />
              <stop offset="100%" stop-color={tempCss(cycleTemps[i + 1])} />
            </linearGradient>
          {/each}
        </defs>
        <!-- Temp line segments -->
        {#each chartLines as seg, i}
          <line x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2} stroke="url(#tgrad-sim-{i})" stroke-width="2" />
        {/each}

        <!-- Temp dots + label for active month -->
        {#each chartDots as dot}
          {@const c = tempCss(dot.t)}
          <circle
            cx={dot.cx} cy={dot.cy} r={dot.active ? 5 : 4}
            fill={c}
            stroke={dot.active ? '#fff' : 'rgba(255,255,255,0.25)'}
            stroke-width={dot.active ? 1.5 : 1}
          />
          {#if dot.active}
            {@const barY = chartBars[displayMonthIdx].y}
            {@const precipLabelY = Math.min(barY - 4, dot.cy - 20)}
            <text x={dot.cx} y={precipLabelY} text-anchor="middle" fill="rgba(42,158,158,0.9)" font-size="11">
              {$useImperial ? (cyclePrecip[displayMonthIdx] / 25.4).toFixed(1) + '"' : cyclePrecip[displayMonthIdx].toFixed(0) + 'mm'}
            </text>
            <text x={dot.cx} y={dot.cy - 9} text-anchor="middle" fill="#f4a460" font-size="12">
              {$useImperial ? cToF(dot.t).toFixed(0) + '°F' : dot.t.toFixed(0) + '°C'}
            </text>
          {/if}
        {/each}

        <!-- Month labels -->
        {#each chartMonths as m}
          <text
            x={m.x} y={CHART_H - 6}
            text-anchor="middle"
            fill={m.active ? '#fff' : m.isCurrent ? 'rgba(255,255,255,0.5)' : '#888'}
            font-size="12"
            font-weight={m.active ? '600' : '400'}
          >{m.label}</text>
        {/each}

        <!-- Y axis labels (clickable to toggle units) -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <text x={PAD_L - 4} y={PAD_T + 4} text-anchor="end" fill="#f4a460" font-size="11" style="cursor:pointer" on:click|stopPropagation={toggleUnits}>
          {imp ? cToF(tempPlotMax).toFixed(0) + '°' : tempPlotMax.toFixed(0) + '°'}
        </text>
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <text x={PAD_L - 4} y={PAD_T + INNER_H} text-anchor="end" fill="#f4a460" font-size="11" style="cursor:pointer" on:click|stopPropagation={toggleUnits}>
          {imp ? cToF(tempPlotMin).toFixed(0) + '°' : tempPlotMin.toFixed(0) + '°'}
        </text>
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <text x={CHART_W - PAD_R} y={PAD_T + 4} text-anchor="end" fill="rgba(42,158,158,0.9)" font-size="11" style="cursor:pointer" on:click|stopPropagation={toggleUnits}>
          {$useImperial ? (precipMax / 25.4).toFixed(1) + 'in' : precipMax.toFixed(0) + 'mm'}/mo
        </text>

        <!-- Chart label -->
        <text x={PAD_L + 4} y="14" fill="rgba(255,255,255,0.8)" font-size="11">Simulated</text>

        <!-- Invisible hit areas per month column -->
        {#each hitAreas as hit}
          <!-- svelte-ignore a11y-no-static-element-interactions -->
          <rect
            x={hit.x} y={PAD_T} width={INNER_W / 12} height={INNER_H + PAD_B}
            fill="rgba(0,0,0,0.001)"
            style="cursor: pointer; pointer-events: all;"
            on:mousemove={() => hoveredMonthIdx = hit.i}
            on:click|stopPropagation={() => { dispatch('setMonth', hit.i); hoveredMonthIdx = null; }}
          />
        {/each}
      </svg>

      <!-- Observed chart -->
      <svg
        viewBox="0 0 {CHART_W} {CHART_H}" width="100%" preserveAspectRatio="xMidYMid meet" class="cycle-chart"
      >
        <!-- Obs precip bars -->
        {#each obsChartBars as bar, i}
          {#if bar.valid}
            {@const minH = 2}
            {@const displayH = Math.max(bar.h, minH)}
            {@const displayY = PAD_T + INNER_H - displayH}
            {@const active = i === displayMonthIdx}
            {@const c = precipCss(obsPrecips[i] ?? 0)}
            <rect
              x={bar.x} y={displayY} width={BAR_W} height={displayH}
              fill={c}
              opacity={active ? 0.7 : 0.3}
              stroke={active ? 'rgba(255,255,255,0.6)' : c} stroke-width={active ? 1 : 0.75} stroke-dasharray={active ? 'none' : '3 2'}
            />
          {/if}
        {/each}

        <!-- Obs temp line segment gradients -->
        <defs>
          {#each obsChartLines as seg, i}
            {#if seg.valid}
              <linearGradient id="tgrad-obs-{i}" x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2} gradientUnits="userSpaceOnUse">
                <stop offset="0%" stop-color={tempCss(obsTemps[i] ?? 0)} />
                <stop offset="100%" stop-color={tempCss(obsTemps[i + 1] ?? 0)} />
              </linearGradient>
            {/if}
          {/each}
        </defs>
        <!-- Obs temp line segments -->
        {#each obsChartLines as seg, i}
          {#if seg.valid}
            <line x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2}
              stroke="url(#tgrad-obs-{i})" stroke-width="1.5" stroke-dasharray="4 3" />
          {/if}
        {/each}

        <!-- Obs temp dots + hover labels -->
        {#each obsChartDots as dot, i}
          {#if dot.valid}
            {@const active = i === displayMonthIdx}
            {@const c = tempCss(dot.t ?? 0)}
            <circle cx={dot.cx} cy={dot.cy} r={active ? 5 : 4}
              fill={c}
              stroke={active ? '#fff' : 'rgba(255,255,255,0.25)'}
              stroke-width={active ? 1.5 : 1}
            />
            {#if active}
              {@const obsBar = obsChartBars[i]}
              {@const precipLabelY = obsBar.valid ? Math.min(obsBar.y - 4, dot.cy - 20) : dot.cy - 20}
              {#if obsBar.valid}
                <text x={dot.cx} y={precipLabelY} text-anchor="middle" fill="rgba(42,158,158,0.9)" font-size="11">
                  {$useImperial ? (obsPrecips[i]! / 25.4).toFixed(1) + '"' : obsPrecips[i]!.toFixed(0) + 'mm'}
                </text>
              {/if}
              <text x={dot.cx} y={dot.cy - 9} text-anchor="middle" fill="#f4a460" font-size="12">
                {$useImperial ? cToF(dot.t!).toFixed(0) + '°F' : dot.t!.toFixed(0) + '°C'}
              </text>
            {/if}
          {/if}
        {/each}

        <!-- Month labels -->
        {#each chartMonths as m}
          <text
            x={m.x} y={CHART_H - 6}
            text-anchor="middle"
            fill={m.active ? '#fff' : m.isCurrent ? 'rgba(255,255,255,0.5)' : '#888'}
            font-size="12"
            font-weight={m.active ? '600' : '400'}
          >{m.label}</text>
        {/each}

        <!-- Y axis labels (clickable to toggle units) -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <text x={PAD_L - 4} y={PAD_T + 4} text-anchor="end" fill="#f4a460" font-size="11" style="cursor:pointer" on:click|stopPropagation={toggleUnits}>
          {imp ? cToF(tempPlotMax).toFixed(0) + '°' : tempPlotMax.toFixed(0) + '°'}
        </text>
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <text x={PAD_L - 4} y={PAD_T + INNER_H} text-anchor="end" fill="#f4a460" font-size="11" style="cursor:pointer" on:click|stopPropagation={toggleUnits}>
          {imp ? cToF(tempPlotMin).toFixed(0) + '°' : tempPlotMin.toFixed(0) + '°'}
        </text>
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <text x={CHART_W - PAD_R} y={PAD_T + 4} text-anchor="end" fill="rgba(42,158,158,0.9)" font-size="11" style="cursor:pointer" on:click|stopPropagation={toggleUnits}>
          {$useImperial ? (precipMax / 25.4).toFixed(1) + 'in' : precipMax.toFixed(0) + 'mm'}/mo
        </text>

        <!-- Chart label -->
        <text x={PAD_L + 4} y="14" fill="rgba(255,255,255,0.8)" font-size="11">Observed (1981–2010)</text>

        <!-- Invisible hit areas per month column -->
        {#each hitAreas as hit}
          <!-- svelte-ignore a11y-no-static-element-interactions -->
          <rect
            x={hit.x} y={PAD_T} width={INNER_W / 12} height={INNER_H + PAD_B}
            fill="rgba(0,0,0,0.001)"
            style="cursor: pointer; pointer-events: all;"
            on:mousemove={() => hoveredMonthIdx = hit.i}
            on:click|stopPropagation={() => { dispatch('setMonth', hit.i); hoveredMonthIdx = null; }}
          />
        {/each}
      </svg>

      <div class="cycle-footer">
        <span class="legend-temp">— Temperature</span>
        <span class="legend-precip">▪ Precipitation</span>
        <span class="unit-toggle" role="button" tabindex="0" on:click|stopPropagation={toggleUnits}>
          {imp ? '°F / in' : '°C / mm'}
        </span>
      </div>
    </div>
  {:else}
    <div class="chat-area" bind:this={chatContainer}>
      {#if messages.length === 0}
        <div class="suggestions">
          {#each suggestions as suggestion}
            <button class="suggestion-btn" on:click={() => sendMessage(suggestion)}>
              {suggestion}
            </button>
          {/each}
        </div>
      {:else}
        {#each messages as msg, mi}
          <div class="chat-msg {msg.role}">
            {#if msg.role === 'assistant' && mi === messages.length - 1 && streaming && (!msg.parts || msg.parts.length === 0)}
              <div class="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            {:else if msg.parts && msg.parts.length > 0}
              {#each msg.parts as part}
                {#if part.type === 'tools'}
                  <div class="tool-progress">
                    {#each part.fields as field}
                      <span class="tool-chip">{field}</span>
                    {/each}
                  </div>
                {:else if msg.role === 'assistant'}
                  {@html renderMarkdown(part.content)}
                {:else}
                  {part.content}
                {/if}
              {/each}
            {:else if msg.role === 'assistant'}
              {@html renderMarkdown(msg.content)}
            {:else}
              {msg.content}
            {/if}
            {#if errorOccurred && msg.role === 'assistant' && mi === messages.length - 1}
              <button class="retry-btn" on:click={retryLastMessage}>Retry</button>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

    {#if !limitReached}
      <div class="chat-input-area">
        <input
          type="text"
          class="chat-input"
          placeholder="Ask about this location..."
          bind:value={inputText}
          on:keydown={handleKeydown}
          disabled={streaming}
        />
      </div>
    {:else}
      <div class="limit-msg">Session limit reached</div>
    {/if}
  {/if}
</div>

<style>
  .inspect-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: min(700px, 90vw);
    height: 100vh;
    height: 100dvh;
    background: rgba(0, 0, 0, 0.92);
    border-left: 1px solid #1a6b6b;
    z-index: 200;
    display: flex;
    flex-direction: column;
    font-size: 1rem;
  }

  .inspect-panel.dragging {
    transition: none;
  }

  .drag-area {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    touch-action: none;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #333;
  }

  .coords {
    color: #aaa;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .month-tag {
    background: #1a6b6b;
    color: #fff;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.875rem;
  }

  .close-btn {
    background: none;
    border: none;
    color: #999;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0;
    min-width: auto;
    line-height: 1;
  }
  .close-btn:hover { color: #ccc; }

  .stats {
    padding: 0.5rem 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 1rem;
  }

  .stat { color: #fff; }
  .sst { color: #6ec6ff; }
  .elev { color: #aaa; }
  .wind-stat {
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
  .unit-toggle {
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .unit-toggle:hover { opacity: 0.65; }

  .tab-bar {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #333;
  }

  .tab-group {
    display: flex;
  }

  .tab-btn {
    padding: 0.4rem 0.8rem;
    background: rgba(14, 74, 74, 0.3);
    color: #fff;
    border: 1px solid rgba(26, 107, 107, 0.5);
    cursor: pointer;
    font-size: 0.875rem;
    min-width: auto;
    transition: background 0.15s, color 0.15s;
  }

  .tab-btn:first-child {
    border-radius: 4px 0 0 4px;
  }

  .tab-btn:last-child {
    border-radius: 0 4px 4px 0;
    margin-left: -1px;
  }

  .tab-btn.active {
    background: #1a6b6b;
    color: #fff;
    border-color: rgba(255, 255, 255, 0.3);
    z-index: 1;
    position: relative;
  }

  .tab-btn:hover:not(.active) {
    background: rgba(14, 74, 74, 0.5);
  }

  .cycle-area {
    flex: 1;
    padding: 1rem 1rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    overflow-y: auto;
  }

  .cycle-chart {
    display: block;
    width: 100%;
    flex-shrink: 0;
  }

  .cycle-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.875rem;
    padding-left: 2px;
  }

  .cycle-month-stat {
    display: flex;
    gap: 0.75rem;
    align-items: center;
  }

  .stat-month {
    color: #fff;
    font-weight: 500;
    min-width: 3.5rem;
  }

  .cycle-legend-keys {
    display: flex;
    gap: 0.75rem;
  }

  .legend-temp { color: #f4a460; }
  .legend-precip { color: rgba(42, 158, 158, 0.9); }

  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .suggestions {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .suggestion-btn {
    text-align: left;
    background: rgba(26, 107, 107, 0.2);
    border: 1px solid #1a6b6b;
    color: #aadede;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    min-width: auto;
  }
  .suggestion-btn:hover {
    background: rgba(26, 107, 107, 0.35);
  }

  .chat-msg {
    line-height: 1.5;
    word-wrap: break-word;
  }

  .chat-msg.user {
    color: #aadede;
    font-weight: 500;
    white-space: pre-wrap;
  }

  .chat-msg.assistant {
    color: #ddd;
  }

  /* Markdown content inside assistant messages */
  .chat-msg.assistant :global(p) {
    margin: 0.4em 0;
  }
  .chat-msg.assistant :global(p:first-child) {
    margin-top: 0;
  }
  .chat-msg.assistant :global(p:last-child) {
    margin-bottom: 0;
  }
  .chat-msg.assistant :global(h2),
  .chat-msg.assistant :global(h3) {
    margin: 0.8em 0 0.3em;
    font-size: 1em;
    font-weight: 600;
    color: #aadede;
  }
  .chat-msg.assistant :global(h2:first-child),
  .chat-msg.assistant :global(h3:first-child) {
    margin-top: 0;
  }
  .chat-msg.assistant :global(table) {
    border-collapse: collapse;
    font-size: 0.85em;
    margin: 0.5em 0;
    width: 100%;
  }
  .chat-msg.assistant :global(th),
  .chat-msg.assistant :global(td) {
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 0.25em 0.5em;
    text-align: left;
  }
  .chat-msg.assistant :global(th) {
    background: rgba(26, 107, 107, 0.2);
    color: #aadede;
    font-weight: 500;
  }
  .chat-msg.assistant :global(blockquote) {
    border-left: 3px solid rgba(26, 107, 107, 0.5);
    margin: 0.5em 0;
    padding: 0.25em 0.75em;
    color: rgba(255, 255, 255, 0.7);
  }
  .chat-msg.assistant :global(ol),
  .chat-msg.assistant :global(ul) {
    margin: 0.4em 0;
    padding-left: 1.5em;
  }
  .chat-msg.assistant :global(code) {
    background: rgba(255, 255, 255, 0.08);
    padding: 0.1em 0.3em;
    border-radius: 3px;
    font-size: 0.9em;
  }
  .chat-msg.assistant :global(strong) {
    color: #fff;
  }
  .chat-msg.assistant :global(.katex-display) {
    margin: 0.5em 0;
    overflow-x: auto;
  }

  .retry-btn {
    margin-top: 0.5rem;
    background: rgba(26, 107, 107, 0.2);
    border: 1px solid #1a6b6b;
    color: #aadede;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    min-width: auto;
  }
  .retry-btn:hover {
    background: rgba(26, 107, 107, 0.35);
  }

  .tool-progress {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
  }

  .tool-chip {
    background: rgba(26, 107, 107, 0.25);
    border: 1px solid rgba(26, 107, 107, 0.4);
    color: #aadede;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.875rem;
  }

  .typing-indicator {
    display: flex;
    gap: 4px;
    padding: 4px 0;
  }
  .typing-indicator span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #1a6b6b;
    animation: blink 1.2s infinite;
  }
  .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
  .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes blink {
    0%, 60%, 100% { opacity: 0.3; }
    30% { opacity: 1; }
  }

  .chat-input-area {
    padding: 0.5rem 1rem;
    border-top: 1px solid #333;
  }

  .chat-input {
    width: 100%;
    background: rgba(14, 74, 74, 0.15);
    border: 1px solid rgba(26, 107, 107, 0.35);
    color: #fff;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    /* iOS Safari auto-zooms inputs with font-size below 16px on focus.
       Keep at 16px+ to prevent the page from being yanked around. */
    font-size: 16px;
    outline: none;
    box-sizing: border-box;
  }
  .chat-input:focus {
    border-color: #1a6b6b;
  }
  .chat-input:disabled {
    opacity: 0.5;
  }

  .limit-msg {
    padding: 0.75rem 1rem;
    text-align: center;
    color: #888;
    border-top: 1px solid #333;
    font-size: 0.9rem;
  }

  /* Desktop: left-edge close affordance */
  .inspect-panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 8px;
    height: 100%;
    cursor: e-resize;
    z-index: 1;
  }

  .drag-handle {
    display: none;
  }

  @media (max-width: 800px), (max-height: 500px) {
    .inspect-panel {
      top: auto;
      bottom: 0;
      left: 0;
      right: 0;
      width: 100%;
      height: auto;
      max-height: 55vh;
      max-height: 55dvh;
      border-left: none;
      border-top: 1px solid #1a6b6b;
      border-radius: 12px 12px 0 0;
    }

    .inspect-panel::before {
      display: none;
    }

    .drag-handle {
      display: flex;
      justify-content: center;
      padding: 0.5rem 0 0.25rem;
      cursor: grab;
      touch-action: none;
    }

    .drag-handle:active {
      cursor: grabbing;
    }

    .drag-handle-pill {
      width: 32px;
      height: 4px;
      background: rgba(255, 255, 255, 0.3);
      border-radius: 2px;
    }
  }
</style>
