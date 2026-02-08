<script lang="ts">
  import { onMount } from 'svelte';
  import GIF from 'gif.js-upgrade';
  import Globe from './lib/globe/Globe.svelte';
  import { loadBinaryData, fieldToNestedArray } from './lib/globe/loadBinaryData';
  import type { ClimateLayerData } from './lib/globe/loadBinaryData';

  let temperatureData: number[][][] | null = null;
  let layerData: ClimateLayerData | null = null;
  let activeLayer: 'temperature' | 'blue-marble' = 'temperature';
  let monthProgress = 0; // Continuous 0-12 value
  let loading = true;
  let error: string | null = null;
  let playing = true;
  let animationFrameId: number | null = null;
  let lastTime: number | null = null;
  let globeComponent: Globe;
  let recording = false;
  let recordingProgress = '';
  let uniformLighting = false;

  // Derive discrete month for UI display
  $: displayMonth = Math.round(monthProgress) % 12;

  const MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  function animateMonth(time: number) {
    if (lastTime !== null) {
      const dt = (time - lastTime) / 1000;
      // When auto-rotating: 1 month per second
      // When not auto-rotating: 1 year per 3 minutes (12 days per year, 1 day per 15 sec)
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

  function resetView() {
    if (globeComponent) {
      globeComponent.resetView();
    }
    monthProgress = 0;
    if (!playing) {
      startPlaying();
    }
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

    // Capture 12 months, rotating full 360 degrees over the year
    const framesPerMonth = 10;
    const totalFrames = 12 * framesPerMonth;
    const rotationPerFrame = (2 * Math.PI) / totalFrames; // Full rotation over all frames

    for (let i = 0; i < totalFrames; i++) {
      monthProgress = i / framesPerMonth;
      const currentMonth = Math.floor(monthProgress);
      recordingProgress = `Capturing ${MONTH_NAMES[currentMonth]}... (${i + 1}/${totalFrames})`;

      // Rotate globe
      globeComponent.rotateGlobe(rotationPerFrame);

      // Render and wait a frame
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

  onMount(async () => {
    try {
      const binData = await loadBinaryData('');
      layerData = binData;
      temperatureData = fieldToNestedArray(binData.temperature_2m);
      loading = false;
      startPlaying();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
      loading = false;
    }

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  });
</script>

<main>
  {#if loading}
    <div class="loading">Loading climate data...</div>
  {:else if error}
    <div class="error">Error: {error}</div>
  {:else if temperatureData}
    <div class="globe-wrapper">
      <Globe
        bind:this={globeComponent}
        data={temperatureData}
        {monthProgress}
        {activeLayer}
        {layerData}
        {uniformLighting}
        on:interact={stopAutoRotate}
      />
    </div>
    <div class="controls">
      <div class="layer-tabs">
        <button
          class="layer-tab"
          class:active={activeLayer === 'temperature'}
          on:click={() => activeLayer = 'temperature'}
          disabled={recording}
        >Temperature</button>
        <button
          class="layer-tab"
          class:active={activeLayer === 'blue-marble'}
          on:click={() => activeLayer = 'blue-marble'}
          disabled={recording || !layerData}
        >Blue Marble</button>
      </div>
      <div class="layer-tabs">
        <button
          class="layer-tab"
          class:active={!uniformLighting}
          on:click={() => uniformLighting = false}
          disabled={recording}
        >Day/Night</button>
        <button
          class="layer-tab"
          class:active={uniformLighting}
          on:click={() => uniformLighting = true}
          disabled={recording}
        >Always Day</button>
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
      <button on:click={togglePlay} disabled={recording}>
        {#if playing}
          Pause
        {:else}
          Play
        {/if}
      </button>
      <button on:click={resetView} disabled={recording}>
        Reset
      </button>
      <button on:click={recordGif} disabled={recording}>
        {#if recording}
          {recordingProgress}
        {:else}
          Record GIF
        {/if}
      </button>
    </div>
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
  }

  .globe-wrapper {
    flex: 1;
    min-height: 0;
  }

  .controls {
    padding: 0.75rem 1rem;
    padding-bottom: 1.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 1.5rem;
    background: rgba(0, 0, 0, 0.8);
  }

  .layer-tabs {
    display: flex;
    gap: 0;
  }

  .layer-tab {
    padding: 0.4rem 0.8rem;
    background: #222;
    color: #999;
    border: 1px solid #444;
    cursor: pointer;
    font-size: 0.85rem;
    min-width: auto;
  }

  .layer-tab:first-child {
    border-radius: 4px 0 0 4px;
  }

  .layer-tab:last-child {
    border-radius: 0 4px 4px 0;
    border-left: none;
  }

  .layer-tab.active {
    background: #444;
    color: #fff;
    border-color: #666;
  }

  .layer-tab:hover:not(:disabled):not(.active) {
    background: #333;
    color: #ccc;
  }

  .separator {
    width: 1px;
    height: 24px;
    background: #444;
  }

  .controls label {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .month-label {
    min-width: 100px;
    text-align: right;
  }

  input[type="range"] {
    width: 300px;
    cursor: pointer;
  }

  button {
    padding: 0.5rem 1rem;
    background: #333;
    color: #fff;
    border: 1px solid #555;
    border-radius: 4px;
    cursor: pointer;
    min-width: 80px;
  }

  button:hover:not(:disabled) {
    background: #444;
  }

  button:disabled {
    cursor: wait;
    opacity: 0.7;
  }

  .loading, .error {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
  }

  .error {
    color: #ff4444;
  }
</style>
