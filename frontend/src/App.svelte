<script lang="ts">
  import { onMount } from 'svelte';
  import GIF from 'gif.js-upgrade';
  import Globe from './lib/globe/Globe.svelte';
  import InspectPanel from './lib/InspectPanel.svelte';
  import ControlBar from './lib/ControlBar.svelte';
  import Legend from './lib/Legend.svelte';
  import { loadBinaryDataInWorker, loadLandMask1deg } from './lib/globe/loadBinaryData';
  import type { ClimateLayerData } from './lib/globe/loadBinaryData';
  import { useImperial } from './lib/stores';

  let temperatureData: number[][][] | null = null;
  let layerData: ClimateLayerData | null = null;
  let activeLayer: 'temperature' | 'precipitation' | 'blue-marble' = 'blue-marble';

  function cToF(c: number): number { return c * 9 / 5 + 32; }
  function mmToIn(mm: number): number { return mm / 25.4; }
  function toggleUnits() { useImperial.update(v => !v); }

  $: tempLegendStops = $useImperial ? [
    { value: cToF(-30).toFixed(0) + '', color: 'rgb(59,30,109)' },
    { value: cToF(0).toFixed(0) + '', color: 'rgb(30,136,229)' },
    { value: cToF(10).toFixed(0) + '', color: 'rgb(102,187,106)' },
    { value: cToF(25).toFixed(0) + '', color: 'rgb(251,140,0)' },
    { value: cToF(40).toFixed(0) + '', color: 'rgb(138,0,0)' },
  ] : [
    { value: '-30', color: 'rgb(59,30,109)' },
    { value: '0', color: 'rgb(30,136,229)' },
    { value: '10', color: 'rgb(102,187,106)' },
    { value: '25', color: 'rgb(251,140,0)' },
    { value: '40', color: 'rgb(138,0,0)' },
  ];
  $: tempLegendLabel = $useImperial ? '°F' : '°C';

  $: precipLegendStops = $useImperial ? [
    { value: '0', color: 'rgb(210,200,180)' },
    { value: mmToIn(30).toFixed(1), color: 'rgb(180,210,170)' },
    { value: mmToIn(90).toFixed(0), color: 'rgb(100,190,120)' },
    { value: mmToIn(180).toFixed(0), color: 'rgb(40,150,100)' },
    { value: mmToIn(450).toFixed(0), color: 'rgb(20,50,120)' },
  ] : [
    { value: '0', color: 'rgb(210,200,180)' },
    { value: '30', color: 'rgb(180,210,170)' },
    { value: '90', color: 'rgb(100,190,120)' },
    { value: '180', color: 'rgb(40,150,100)' },
    { value: '450', color: 'rgb(20,50,120)' },
  ];
  $: precipLegendLabel = $useImperial ? 'in/mo' : 'mm/mo';
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
    {#if revealed && activeLayer === 'temperature'}
      <Legend
        stops={tempLegendStops}
        label={tempLegendLabel}
        visible={controlsVisible}
        on:toggleUnits={toggleUnits}
      />
    {/if}
    {#if revealed && activeLayer === 'precipitation'}
      <Legend
        stops={precipLegendStops}
        label={precipLegendLabel}
        visible={controlsVisible}
        on:toggleUnits={toggleUnits}
      />
    {/if}
    {#if revealed}
      <ControlBar
        bind:activeLayer
        bind:uniformLighting
        bind:monthProgress
        {playing}
        {recording}
        {recordingProgress}
        layerDataLoaded={!!layerData}
        {displayMonth}
        visible={controlsVisible}
        on:togglePlay={togglePlay}
        on:stopPlaying={stopPlaying}
        on:resetView={resetView}
        on:recordGif={recordGif}
      />
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
