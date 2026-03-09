<script lang="ts">
  import { createEventDispatcher, onDestroy, onMount } from 'svelte';
  import { fly } from 'svelte/transition';
  import type { ClimateLayerData } from './loadBinaryData';
  import { useImperial } from './stores';

  const dispatch = createEventDispatcher();

  export let lat: number;
  export let lon: number;
  export let monthProgress: number;
  export let temperatureData: number[][][] | null;
  export let layerData: ClimateLayerData | null;

  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
  const MAX_MESSAGES = 50;

  // Chat state
  type MsgPart = { type: 'text'; content: string } | { type: 'tools'; content: string; fields: string[] };
  let messages: { role: 'user' | 'assistant'; content: string; parts?: MsgPart[] }[] = [];
  let inputText = '';
  let streaming = false;
  let limitReached = false;
  let chatContainer: HTMLDivElement;
  let currentParts: MsgPart[] = [];
  let sentLat: number | null = null;
  let sentLon: number | null = null;
  let sentMonth: number | null = null;

  const SUGGESTIONS = [
    'Why is it this temperature?',
    'What affects the climate here?',
    'Compare to nearby coast',
  ];

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

  function toggleUnits() { useImperial.update(v => !v); }
  function cToF(c: number): number { return c * 9 / 5 + 32; }
  function mToFt(m: number): number { return m * 3.28084; }
  function kphToMph(kph: number): number { return kph * 0.621371; }

  $: imp = $useImperial;
  $: elevation = sampleElevation(lat, lon);
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

    const userMsg = { role: 'user' as const, content: text.trim() };
    messages = [...messages, userMsg];
    inputText = '';
    streaming = true;

    if (messages.filter(m => m.role === 'user').length >= MAX_MESSAGES) {
      limitReached = true;
    }

    // Add placeholder for assistant response
    messages = [...messages, { role: 'assistant' as const, content: '' }];
    currentParts = [];
    scrollToBottom();

    try {
      abortController = new AbortController();
      const month = Math.floor(monthProgress) % 12;
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lat, lon, month,
          prevLat: sentLat,
          prevLon: sentLon,
          prevMonth: sentMonth,
          imperial: $useImperial,
          messages: messages.filter(m => m.content !== '').map(m => ({
            role: m.role, content: m.content,
          })),
        }),
        signal: abortController.signal,
      });

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No response body');

      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              const last = currentParts[currentParts.length - 1];
              if (last && last.type === 'text') {
                last.content += parsed.error;
              } else {
                currentParts.push({ type: 'text', content: parsed.error });
              }
              messages[messages.length - 1].content += parsed.error;
            } else if (parsed.tool) {
              const last = currentParts[currentParts.length - 1];
              if (last && last.type === 'tools') {
                last.fields = [...last.fields, parsed.tool];
              } else {
                currentParts.push({ type: 'tools', content: '', fields: [parsed.tool] });
              }
            } else if (parsed.text) {
              const last = currentParts[currentParts.length - 1];
              if (last && last.type === 'text') {
                last.content += parsed.text;
              } else {
                currentParts.push({ type: 'text', content: parsed.text });
              }
              messages[messages.length - 1].content += parsed.text;
            } else {
              continue;
            }
            currentParts = [...currentParts];
            messages[messages.length - 1].parts = currentParts;
            messages = [...messages];
            scrollToBottom();
          } catch { /* skip malformed chunks */ }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        const last = messages[messages.length - 1];
        if (last.role === 'assistant' && !last.content) {
          last.content = 'Failed to connect to the explanation server. Make sure the backend is running on port 8000.';
          messages = [...messages];
        }
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

  let isMobile = typeof window !== 'undefined' && window.innerWidth <= 640;

  // ── Drag-to-dismiss (mobile) ──
  let panelEl: HTMLDivElement;
  let dragStartY = 0;
  let dragOffsetY = 0;
  let dragging = false;

  function onDragStart(e: TouchEvent | MouseEvent) {
    if (!isMobile) return;
    dragging = true;
    dragOffsetY = 0;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragStartY = clientY;
    // Prevent text selection during drag
    e.preventDefault();
  }

  function onDragMove(e: TouchEvent | MouseEvent) {
    if (!dragging) return;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    dragOffsetY = Math.max(0, clientY - dragStartY); // only allow dragging down
    if (panelEl) panelEl.style.transform = `translateY(${dragOffsetY}px)`;
  }

  function onDragEnd() {
    if (!dragging) return;
    dragging = false;
    if (dragOffsetY > 100) {
      // Dismiss threshold reached
      close();
    } else {
      // Snap back
      if (panelEl) panelEl.style.transform = '';
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

  onMount(() => {
    if (isMobile) {
      window.addEventListener('touchmove', onDragMove, { passive: false });
      window.addEventListener('touchend', onDragEnd);
      window.addEventListener('mousemove', onDragMove);
      window.addEventListener('mouseup', onDragEnd);
    }
    return () => {
      window.removeEventListener('touchmove', onDragMove);
      window.removeEventListener('touchend', onDragEnd);
      window.removeEventListener('mousemove', onDragMove);
      window.removeEventListener('mouseup', onDragEnd);
    };
  });
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div
  class="inspect-panel"
  class:dragging
  bind:this={panelEl}
  transition:fly={{ x: isMobile ? 0 : 700, y: isMobile ? 400 : 0, duration: 200 }}
  on:click={onPanelClick}
>
  {#if isMobile}
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="drag-handle" on:touchstart={onDragStart} on:mousedown={onDragStart}>
      <div class="drag-handle-pill"></div>
    </div>
  {/if}
  <div class="panel-header">
    <div class="coords">
      {Math.abs(lat).toFixed(1)}°{lat >= 0 ? 'N' : 'S'}, {Math.abs(lon).toFixed(1)}°{lon >= 0 ? 'E' : 'W'}
      <span class="month-tag">{displayMonth}</span>
    </div>
    <button class="close-btn" on:click|stopPropagation={close}>×</button>
  </div>

  <div class="stats">
    <div class="stat">
      <span class="unit-toggle" on:click|stopPropagation={toggleUnits}>
        {imp ? cToF(tempC).toFixed(1) + '°F' : tempC.toFixed(1) + '°C'}
      </span>
    </div>
    {#if ocean.isOcean && ocean.sstC !== null}
      <div class="stat sst">
        Sea: <span class="unit-toggle" on:click|stopPropagation={toggleUnits}>
          {imp ? cToF(ocean.sstC).toFixed(1) + '°F' : ocean.sstC.toFixed(1) + '°C'}
        </span>
      </div>
    {/if}
    {#if elevation !== null}
      <div class="stat elev">
        {ocean.isOcean ? 'Depth' : 'Elev'}: <span class="unit-toggle" on:click|stopPropagation={toggleUnits}>
          {#if imp}
            {ocean.isOcean ? Math.abs(Math.round(mToFt(elevation))).toLocaleString() : Math.round(mToFt(elevation)).toLocaleString()} ft
          {:else}
            {ocean.isOcean ? Math.abs(Math.round(elevation)).toLocaleString() : Math.round(elevation).toLocaleString()} m
          {/if}
        </span>
      </div>
    {/if}
    <div class="stat wind-stat">
      <span class="wind-arrow" style="transform: rotate({wind.dir}deg)">↓</span>
      <span class="unit-toggle" on:click|stopPropagation={toggleUnits}>
        {#if imp}
          {kphToMph(wind.speed * 3.6).toFixed(0)} mph
        {:else}
          {(wind.speed * 3.6).toFixed(0)} km/h
        {/if}
      </span>
    </div>
  </div>

  <div class="divider"></div>

  <div class="chat-area" bind:this={chatContainer}>
    {#if messages.length === 0}
      <div class="suggestions">
        <div class="suggestions-label">Ask about this location:</div>
        {#each SUGGESTIONS as suggestion}
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
              {:else}
                {part.content}
              {/if}
            {/each}
          {:else}
            {msg.content}
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
</div>

<style>
  .inspect-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: min(700px, 90vw);
    height: 100vh;
    background: rgba(0, 0, 0, 0.92);
    border-left: 1px solid #1a6b6b;
    z-index: 100;
    display: flex;
    flex-direction: column;
    font-size: 1rem;
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
    font-size: 0.85rem;
  }

  .close-btn {
    background: none;
    border: none;
    color: #666;
    font-size: 1.2rem;
    cursor: pointer;
    padding: 0;
    min-width: auto;
    line-height: 1;
  }
  .close-btn:hover { color: #aaa; }

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

  .divider {
    height: 1px;
    background: #333;
    margin: 0 1rem;
  }

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

  .suggestions-label {
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
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
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .chat-msg.user {
    color: #aadede;
    font-weight: 500;
  }

  .chat-msg.assistant {
    color: #ddd;
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
    font-size: 0.75rem;
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
    background: #1a1a1a;
    border: 1px solid #444;
    color: #fff;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.9rem;
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

  .inspect-panel.dragging {
    transition: none;
  }

  .drag-handle {
    display: none;
  }

  @media (max-width: 640px) {
    .inspect-panel {
      top: auto;
      bottom: 0;
      left: 0;
      right: 0;
      width: 100%;
      max-height: 65vh;
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
