<script lang="ts">
  import { onMount } from 'svelte';
  import GIF from 'gif.js-upgrade';
  import Globe from './lib/globe/Globe.svelte';
  import InspectPanel from './lib/InspectPanel.svelte';
  import ControlBar from './lib/ControlBar.svelte';
  import Legend from './lib/Legend.svelte';
  import { loadLandMask1deg, snapToLand } from './lib/globe/loadBinaryData';
  import type { ClimateLayerData } from './lib/globe/loadBinaryData';
  import { useImperial } from './lib/stores';
  import OnboardingOverlay from './lib/OnboardingOverlay.svelte';
  import About from './lib/About.svelte';
  let aboutOpen = false;
  import { currentStage, stageLoading, STAGES, STAGE_NAMES, type Stage } from './lib/onboardingState';
  import { loadStageData, preloadStageFile } from './lib/globe/loadBinaryData';

  let temperatureData: number[][][] | null = null;
  let layerData: ClimateLayerData | null = null;
  let activeLayer: 'temperature' | 'precipitation' | 'blue-marble' = 'temperature';

  function cToF(c: number): number { return c * 9 / 5 + 32; }
  function mmToIn(mm: number): number { return mm / 25.4; }
  function toggleUnits() { useImperial.update(v => !v); }

  $: tempLegendStops = $useImperial ? [
    { value: cToF(-30).toFixed(0) + '', color: 'rgb(59,30,109)' },
    { value: cToF(0).toFixed(0) + '', color: 'rgb(30,136,229)', discontinuity: true },
    { value: '', color: 'rgb(100,181,246)' },
    { value: cToF(10).toFixed(0) + '', color: 'rgb(102,187,106)' },
    { value: cToF(25).toFixed(0) + '', color: 'rgb(251,140,0)' },
    { value: cToF(40).toFixed(0) + '', color: 'rgb(138,0,0)' },
  ] : [
    { value: '-30', color: 'rgb(59,30,109)' },
    { value: '0', color: 'rgb(30,136,229)', discontinuity: true },
    { value: '', color: 'rgb(100,181,246)' },
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
  let controlsVisible = false;
  let revealClicked = false;

  // Post-onboarding location prompt
  let locationDismissed = false;

  function dismissLocationPrompt() {
    locationDismissed = true;
  }

  function requestLocation() {
    locationDismissed = true;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        let { latitude: lat, longitude: lon } = pos.coords;
        // Snap to nearest land cell using high-res land mask
        if (layerData?.land_mask) {
          const snapped = snapToLand(layerData.land_mask, lat, lon);
          lat = snapped.lat;
          lon = snapped.lon;
        }
        pickLoc = { lat, lon };
        globeComponent?.flyTo(lat, lon);
      },
      () => {
        // Permission denied or error — just dismiss
      },
    );
  }

  // Stage-derived state
  $: stage = $currentStage;
  $: revealed = stage >= 1;

  // Switch to blue-marble once when first reaching stage 5
  let didAutoSwitchLayer = false;
  $: if (stage === 5 && !didAutoSwitchLayer) {
    didAutoSwitchLayer = true;
    activeLayer = 'blue-marble';
  }
  // Force temperature layer if current layer isn't available at this stage
  $: if (!layerData?.surface && activeLayer === 'blue-marble') {
    activeLayer = 'temperature';
  }
  $: if (!layerData?.precipitation && activeLayer === 'precipitation') {
    activeLayer = 'temperature';
  }

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
      const speed = isAutoRotating ? 1 : (1 / 5);
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
    if (!locationDismissed && stage === 5) dismissLocationPrompt();
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

  async function advanceStage() {
    const nextStage = (stage + 1) as Stage;
    if (nextStage > 5) return;

    if (nextStage === 1) {
      // Stage 0 → 1: "Let there be light" — flash effect then load stage 1
      revealClicked = true;
      stageLoading.set(true);
      globeComponent?.triggerFlash();

      // Load stage 1 data in parallel with flash bloom
      const loadPromise = loadStageData(
        1,
        (ld) => { layerData = ld; },
        (td) => { temperatureData = td; },
        (e) => { error = e.message; },
      );
      await Promise.all([loadPromise, new Promise(r => setTimeout(r, 650))]);
      currentStage.set(1);
      stageLoading.set(false);

      requestAnimationFrame(() => requestAnimationFrame(() => {
        globeComponent?.startFlashFade();
      }));

      // Show controls after flash fades
      setTimeout(() => { controlsVisible = true; }, 1500);

      // Preload stage 2
      preloadStageFile(2);
    } else {
      // Stage N → N+1: same pattern as "let there be light"
      // 1. Flash blooms in (hides everything)
      // 2. Data loads behind flash (instant if preloaded)
      // 3. Dispose old + swap new while flash at peak
      // 4. Fade reveals new globe
      stageLoading.set(true);
      globeComponent?.triggerFlash();

      // Load next stage data (instant if preloaded, parallel with bloom)
      let nextLayerData: typeof layerData = null;
      let nextTempData: typeof temperatureData = null;
      const loadPromise = loadStageData(
        nextStage,
        (ld) => { nextLayerData = ld; },
        (td) => { nextTempData = td; },
        (e) => { error = e.message; },
      );

      // Wait for both flash bloom (650ms) and data load to finish
      await Promise.all([loadPromise, new Promise(r => setTimeout(r, 650))]);

      // Flash is at peak — swap data behind it
      globeComponent?.disposeStageMeshes();
      layerData = nextLayerData;
      temperatureData = nextTempData;
      currentStage.set(nextStage);
      stageLoading.set(false);

      // Let meshes build for a couple frames, then start fade
      requestAnimationFrame(() => requestAnimationFrame(() => {
        globeComponent?.startFlashFade();
      }));

      // Preload next stage
      if (nextStage < 5) {
        preloadStageFile(nextStage + 1);
      }
    }
  }

  async function skipToFullModel() {
    stageLoading.set(true);
    globeComponent?.triggerFlash();

    let nextLayerData: typeof layerData = null;
    let nextTempData: typeof temperatureData = null;
    const loadPromise = loadStageData(
      5,
      (ld) => { nextLayerData = ld; },
      (td) => { nextTempData = td; },
      (e) => { error = e.message; },
    );

    await Promise.all([loadPromise, new Promise(r => setTimeout(r, 650))]);

    globeComponent?.disposeStageMeshes();
    layerData = nextLayerData;
    temperatureData = nextTempData;

    currentStage.set(5);
    stageLoading.set(false);
    if (!controlsVisible) controlsVisible = true;

    requestAnimationFrame(() => requestAnimationFrame(() => {
      globeComponent?.startFlashFade();
    }));
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

    const GIF_MAX = 800;
    const gifScale = Math.min(1, GIF_MAX / Math.max(canvas.width, canvas.height));
    const gifWidth = Math.round(canvas.width * gifScale);
    const gifHeight = Math.round(canvas.height * gifScale);

    // Offscreen canvas to downscale frames if needed
    const offscreen = document.createElement('canvas');
    offscreen.width = gifWidth;
    offscreen.height = gifHeight;
    const offCtx = offscreen.getContext('2d')!;

    const gif = new GIF({
      workers: 2,
      quality: 10,
      width: gifWidth,
      height: gifHeight,
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

      offCtx.drawImage(canvas, 0, 0, gifWidth, gifHeight);
      gif.addFrame(offscreen, { copy: true, delay: 80 });
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
    startPlaying();

    // Only fetch land mask for primordial globe — data loads on demand per stage
    loadLandMask1deg().then(lm => {
      primordialLandMask = lm;
    }).catch(e => {
      console.error('Failed to load land mask:', e);
    });

    // Preload stage 1 data in background while user sees primordial globe
    preloadStageFile(1);

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  });
</script>

<svelte:window on:keydown={handleKeydown} />
<main>
  {#if error}
    <div class="globe-wrapper">
      {#if primordialLandMask}
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
        {stage}
        on:interact={stopAutoRotate}
        on:pick={handlePick}
      />
    </div>
    {#if stage >= 1 && pickLoc}
      <InspectPanel
        lat={pickLoc.lat}
        lon={pickLoc.lon}
        {monthProgress}
        {temperatureData}
        {layerData}
        {stage}
        on:close={() => { pickLoc = null; globeComponent?.dismissMarker(); }}
        on:setMonth={(e) => { monthProgress = e.detail; playing = false; }}
      />
    {/if}
    {#if stage === 0 && !revealClicked}
      <div class="title-card">
        <h1 class="title">Building Earth</h1>
        <p class="subtitle">A first-principles climate simulation</p>
        <button class="credits-link" on:click={() => aboutOpen = true}>About</button>
      </div>
      <button class="reveal-btn" on:click={advanceStage}>
        Let there be light
      </button>
    {/if}
    {#if stage >= 1 && stage <= 4}
      <OnboardingOverlay
        {stage}
        loading={$stageLoading}
        buttonLabel={STAGES[stage].button ? `${STAGES[stage].button} (${stage}/4)` : null}
        on:advance={advanceStage}
        on:skip={skipToFullModel}
      />
    {/if}
    {#if stage === 5}
      <OnboardingOverlay
        {stage}
        loading={false}
        buttonLabel={null}
        {locationDismissed}
        on:locate={requestLocation}
        on:dismissLocation={dismissLocationPrompt}
      />
    {/if}
    {#if stage >= 1 && activeLayer === 'temperature'}
      <Legend
        stops={tempLegendStops}
        label={tempLegendLabel}
        visible={controlsVisible && !$stageLoading}
        on:toggleUnits={toggleUnits}
      />
    {/if}
    {#if stage >= 1 && activeLayer === 'precipitation'}
      <Legend
        stops={precipLegendStops}
        label={precipLegendLabel}
        visible={controlsVisible && !$stageLoading}
        on:toggleUnits={toggleUnits}
      />
    {/if}
    {#if stage >= 1}
      <ControlBar
        bind:activeLayer
        bind:uniformLighting
        bind:monthProgress
        {playing}
        {recording}
        {recordingProgress}
        layerDataLoaded={!!layerData}
        hasPrecipitation={!!layerData?.precipitation}
        hasSurface={!!layerData?.surface}
        {displayMonth}
        visible={controlsVisible && !$stageLoading}
        {stage}
        on:togglePlay={togglePlay}
        on:stopPlaying={stopPlaying}
        on:resetView={resetView}
        on:recordGif={recordGif}
      />
    {/if}
  {/if}
</main>
<About bind:open={aboutOpen} />

<style>
  @font-face {
    font-family: 'Space Grotesk';
    font-style: normal;
    font-weight: 300 700;
    font-display: swap;
    src: url('/fonts/SpaceGrotesk-Latin.woff2') format('woff2');
  }

  :global(html, body) {
    margin: 0;
    padding: 0;
    background: #000;
    color: #fff;
    font-family: 'Space Grotesk', sans-serif;
    overflow: hidden;
  }

  :global(button, input, select, textarea) {
    font-family: inherit;
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

  .title-card {
    position: absolute;
    top: 3rem;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    z-index: 10;
  }

  .title {
    font-size: 4rem;
    font-weight: 300;
    letter-spacing: 0.08em;
    color: #fff;
    text-shadow: 0 2px 30px rgba(0, 0, 0, 0.8);
    margin: 0;
  }

  .subtitle {
    font-size: 1.15rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    color: rgba(255, 255, 255, 0.85);
    text-shadow: 0 2px 12px rgba(0, 0, 0, 0.9);
    margin: 0;
  }

  .credits-link {
    margin-top: 1.25rem;
    background: none;
    border: none;
    padding: 0;
    color: rgba(255, 255, 255, 0.55);
    font-size: 0.8rem;
    font-family: inherit;
    cursor: pointer;
    letter-spacing: 0.05em;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.8);
    text-decoration: underline;
    text-underline-offset: 3px;
    transition: color 0.15s;
    display: block;
  }

  .credits-link:hover {
    color: rgba(255, 255, 255, 0.85);
  }

  @media (max-width: 640px), (max-height: 500px) {
    .title-card {
      top: 2rem;
      left: 1.5rem;
      transform: none;
    }

    .title {
      font-size: 2.5rem;
    }

    .subtitle {
      font-size: 0.95rem;
    }
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

</style>
