<script lang="ts">
  import { onMount } from 'svelte';
  import GIF from 'gif.js-upgrade';
  import { Thermometer, Globe as GlobeIcon, SunMoon, Sun, Play, Pause, Home } from 'lucide-svelte';
  import Globe from './lib/globe/Globe.svelte';
  import InspectPanel from './lib/InspectPanel.svelte';
  import { loadBinaryDataInWorker, loadLandMask1deg } from './lib/globe/loadBinaryData';
  import type { ClimateLayerData } from './lib/globe/loadBinaryData';

  let temperatureData: number[][][] | null = null;
  let layerData: ClimateLayerData | null = null;
  let activeLayer: 'temperature' | 'blue-marble' = 'blue-marble';
  let monthProgress = 0; // Continuous 0-12 value
  let loading = true; // true until main data is loaded
  let error: string | null = null;
  let playing = true;
  let animationFrameId: number | null = null;
  let lastTime: number | null = null;
  let globeComponent: Globe;
  let recording = false;
  let recordingProgress = '';
  let uniformLighting = false;
  let pickLoc: { lat: number; lon: number } | null = null;

  // Two-phase state
  let primordialLandMask: { data: Uint8Array; nlat: number; nlon: number } | null = null;
  let landMaskReady = false;
  let revealed = false;
  let controlsVisible = false;
  let revealClicked = false;
  let controlsCheckInterval: ReturnType<typeof setInterval> | null = null;

  // Derive discrete month for UI display
  $: displayMonth = Math.round(monthProgress) % 12;

  const MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  function animateMonth(time: number) {
    if (lastTime !== null) {
      const dt = (time - lastTime) / 1000;
      const isAutoRotating = globeComponent?.isAutoRotating() ?? true;
      const speed = isAutoRotating ? 1 : (1 / 15);
      monthProgress = (monthProgress + dt * speed) % 12;
    }
    lastTime = time;
    animationFrameId = requestAnimationFrame(animateMonth);
  }

  function startPlaying() {
    if (animationFrameId) return;
    playing = true;
    lastTime = null;
    animationFrameId = requestAnimationFrame(animateMonth);
  }

  function stopPlaying() {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    lastTime = null;
    playing = false;
  }

  function stopAutoRotate() {
    if (globeComponent) {
      globeComponent.setAutoRotate(false);
    }
  }

  function togglePlay() {
    if (playing) {
      stopPlaying();
    } else {
      startPlaying();
    }
  }

  function handlePick(e: CustomEvent<{ lat: number; lon: number; screenX: number; screenY: number } | null>) {
    if (!e.detail) {
      pickLoc = null;
      return;
    }
    const { lat, lon } = e.detail;
    pickLoc = { lat, lon };
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && pickLoc) {
      pickLoc = null;
      globeComponent?.dismissMarker();
    }
  }

  function resetView() {
    if (globeComponent) {
      globeComponent.resetView();
    }
    monthProgress = 0;
    if (!playing) {
      startPlaying();
    }
  }

  function handleReveal() {
    revealClicked = true;
    // Start the bloom-in effect
    globeComponent?.triggerFlash();

    // Wait for bloom to reach full intensity (~600ms), then build the globe behind it
    setTimeout(() => {
      revealed = true;

      // After the build, let the globe render then start the fade-out
      requestAnimationFrame(() => requestAnimationFrame(() => {
        globeComponent?.startFlashFade();
      }));

      // Fade in controls after flash fades
      const showControls = () => setTimeout(() => { controlsVisible = true; }, 1500);
      if (!loading && temperatureData) {
        showControls();
      } else {
        controlsCheckInterval = setInterval(() => {
          if (!loading && temperatureData) {
            clearInterval(controlsCheckInterval!);
            controlsCheckInterval = null;
            showControls();
          }
        }, 100);
      }
    }, 650); // slightly after bloom completes
  }

  async function recordGif() {
    if (recording || !globeComponent) return;

    stopPlaying();
    globeComponent.resetView();
    recording = true;
    recordingProgress = 'Initializing...';

    const canvas = globeComponent.getCanvas();
    if (!canvas) {
      recording = false;
      return;
    }

    const gif = new GIF({
      workers: 2,
      quality: 10,
      width: canvas.width,
      height: canvas.height,
      workerScript: '/gif.worker.js',
    });

    const framesPerMonth = 10;
    const totalFrames = 12 * framesPerMonth;
    const rotationPerFrame = (2 * Math.PI) / totalFrames;

    for (let i = 0; i < totalFrames; i++) {
      monthProgress = i / framesPerMonth;
      const currentMonth = Math.floor(monthProgress);
      recordingProgress = `Capturing ${MONTH_NAMES[currentMonth]}... (${i + 1}/${totalFrames})`;

      globeComponent.rotateGlobe(rotationPerFrame);
      globeComponent.renderFrame();
      await new Promise(r => requestAnimationFrame(r));

      gif.addFrame(canvas, { copy: true, delay: 80 });
    }

    recordingProgress = 'Encoding GIF...';

    gif.on('finished', (blob: Blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'climate-visualization.gif';
      a.click();
      URL.revokeObjectURL(url);
      recording = false;
      recordingProgress = '';
    });

    gif.render();
  }

  onMount(() => {
    // Start animation immediately so auto-rotate and controls work
    startPlaying();

    // Fetch land mask and main data in parallel
    loadLandMask1deg('').then(lm => {
      primordialLandMask = lm;
      landMaskReady = true;
    }).catch(e => {
      console.error('Failed to load land mask:', e);
    });

    loadBinaryDataInWorker(
      '',
      (ld) => { layerData = ld; },
      (td) => { temperatureData = td; loading = false; },
      (e) => { error = e.message; loading = false; },
    );

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (controlsCheckInterval) clearInterval(controlsCheckInterval);
    };
  });
</script>

<svelte:window on:keydown={handleKeydown} />
<main>
  {#if error}
    <div class="globe-wrapper">
      {#if landMaskReady}
        <Globe
          bind:this={globeComponent}
          data={null}
          {monthProgress}
          {activeLayer}
          layerData={null}
          {uniformLighting}
          {primordialLandMask}
          revealed={false}
        />
      {/if}
    </div>
    <div class="error-overlay">Error: {error}</div>
  {:else}
    <div class="globe-wrapper">
      <Globe
        bind:this={globeComponent}
        data={temperatureData}
        {monthProgress}
        {activeLayer}
        {layerData}
        {uniformLighting}
        {primordialLandMask}
        {revealed}
        on:interact={stopAutoRotate}
        on:pick={handlePick}
      />
    </div>
    {#if revealed && pickLoc}
      <InspectPanel
        lat={pickLoc.lat}
        lon={pickLoc.lon}
        {monthProgress}
        {temperatureData}
        {layerData}
        on:close={() => { pickLoc = null; globeComponent?.dismissMarker(); }}
      />
    {/if}
    {#if !revealed && !revealClicked}
      <button class="reveal-btn" on:click={handleReveal}>
        Let there be light
      </button>
    {/if}
    {#if revealed}
      <div class="controls" class:visible={controlsVisible}>
        <div class="layer-tabs">
          <button
            class="layer-tab"
            class:active={activeLayer === 'temperature'}
            on:click={() => activeLayer = 'temperature'}
            disabled={recording}
            data-tooltip="Temperature"
          ><Thermometer size={16} /></button>
          <button
            class="layer-tab"
            class:active={activeLayer === 'blue-marble'}
            on:click={() => activeLayer = 'blue-marble'}
            disabled={recording || !layerData}
            data-tooltip="Blue Marble"
          ><GlobeIcon size={16} /></button>
        </div>
        <div class="layer-tabs">
          <button
            class="layer-tab"
            class:active={!uniformLighting}
            on:click={() => uniformLighting = false}
            disabled={recording}
            data-tooltip="Day / Night"
          ><SunMoon size={16} /></button>
          <button
            class="layer-tab"
            class:active={uniformLighting}
            on:click={() => uniformLighting = true}
            disabled={recording}
            data-tooltip="Always Day"
          ><Sun size={16} /></button>
        </div>
        <div class="separator"></div>
        <label>
          <span class="month-label">{MONTH_NAMES[displayMonth]}</span>
          <input
            type="range"
            min="0"
            max="11.99"
            step="0.01"
            bind:value={monthProgress}
            on:input={stopPlaying}
          />
        </label>
        <button class="action-btn" on:click={togglePlay} disabled={recording} data-tooltip={playing ? 'Pause' : 'Play'}>
          {#if playing}
            <Pause size={16} />
          {:else}
            <Play size={16} />
          {/if}
        </button>
        <button class="action-btn" on:click={resetView} disabled={recording} data-tooltip="Reset View">
          <Home size={16} />
        </button>
        <button class="action-btn" on:click={recordGif} disabled={recording}>
          {#if recording}
            {recordingProgress}
          {:else}
            Record GIF
          {/if}
        </button>
      </div>
    {/if}
  {/if}
</main>

<style>
  :global(html, body) {
    margin: 0;
    padding: 0;
    background: #000;
    color: #fff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    overflow: hidden;
  }

  :global(#app) {
    width: 100%;
    height: 100%;
  }

  main {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    position: relative;
  }

  .globe-wrapper {
    flex: 1;
    min-height: 0;
  }

  .reveal-btn {
    position: absolute;
    bottom: 3rem;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.8rem 2rem;
    font-size: 1.1rem;
    color: #fff;
    background: linear-gradient(to bottom, #1a6b6b, #0e4a4a);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 6px;
    cursor: pointer;
    letter-spacing: 0.05em;
    z-index: 10;
    transition: background 0.2s, border-color 0.2s;
  }

  .reveal-btn:hover {
    background: linear-gradient(to bottom, #155a5a, #0a3838);
    border-color: rgba(255, 255, 255, 0.4);
  }

  .error-overlay {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #ff4444;
    font-size: 1.2rem;
    z-index: 10;
  }

  .controls {
    padding: 0.75rem 1rem;
    padding-bottom: 1.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
    background: rgba(0, 0, 0, 0.8);
    opacity: 0;
    transition: opacity 0.6s ease;
  }

  .controls.visible {
    opacity: 1;
  }

  .layer-tabs {
    display: flex;
    gap: 0;
  }

  .layer-tab {
    padding: 0.4rem 0.8rem;
    background: rgba(14, 74, 74, 0.3);
    color: #fff;
    border: 1px solid rgba(26, 107, 107, 0.5);
    cursor: pointer;
    font-size: 0.85rem;
    min-width: auto;
    transition: background 0.15s, color 0.15s;
  }

  .layer-tab:first-child {
    border-radius: 4px 0 0 4px;
  }

  .layer-tab:last-child {
    border-radius: 0 4px 4px 0;
    margin-left: -1px;
  }

  .layer-tab.active {
    background: #1a6b6b;
    color: #fff;
    border-color: rgba(255, 255, 255, 0.3);
    z-index: 1;
    position: relative;
  }

  .layer-tab:hover:not(:disabled):not(.active) {
    background: rgba(14, 74, 74, 0.5);
  }

  .layer-tab.active:hover:not(:disabled) {
    background: #156060 !important;
  }

  .separator {
    width: 1px;
    height: 24px;
    background: rgba(26, 107, 107, 0.5);
  }

  .controls label {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .month-label {
    min-width: 100px;
    text-align: right;
    color: #fff;
  }

  input[type="range"] {
    width: 300px;
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
    height: 4px;
    background: #2a9e9e;
    border-radius: 2px;
    outline: none;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #2a9e9e;
    border: 2px solid rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: background 0.15s;
  }

  input[type="range"]::-webkit-slider-thumb:hover {
    background: #3fc0c0;
  }

  input[type="range"]::-webkit-slider-thumb:active {
    background: #1a8080;
  }

  input[type="range"]::-moz-range-thumb {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #2a9e9e;
    border: 2px solid rgba(255, 255, 255, 0.4);
    cursor: pointer;
  }

  input[type="range"]::-moz-range-thumb:active {
    background: #1a8080;
  }

  input[type="range"]::-moz-range-track {
    height: 4px;
    background: #2a9e9e;
    border-radius: 2px;
  }

  [data-tooltip] {
    position: relative;
  }

  [data-tooltip]::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    padding: 0.35rem 0.6rem;
    background: rgba(14, 74, 74, 0.9);
    color: #fff;
    font-size: 0.85rem;
    white-space: nowrap;
    border-radius: 4px;
    border: 1px solid rgba(26, 107, 107, 0.6);
    pointer-events: none;
    opacity: 0;
    transition: opacity 0.15s ease;
  }

  [data-tooltip]:hover::after {
    opacity: 1;
  }

  button {
    padding: 0.5rem 1rem;
    background: rgba(14, 74, 74, 0.3);
    color: #fff;
    border: 1px solid rgba(26, 107, 107, 0.5);
    border-radius: 4px;
    cursor: pointer;
    min-width: auto;
    transition: background 0.15s;
  }

  .action-btn {
    background: #1a6b6b;
    border-color: rgba(255, 255, 255, 0.3);
  }

  .action-btn:hover:not(:disabled) {
    background: #156060;
  }

  button:hover:not(:disabled):not(.reveal-btn):not(.action-btn) {
    background: rgba(14, 74, 74, 0.5);
  }

  button:disabled {
    cursor: wait;
    opacity: 0.7;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
  }

  .error {
    color: #ff4444;
  }
</style>
