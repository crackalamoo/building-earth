<script lang="ts">
  import { onMount } from 'svelte';
  import GIF from 'gif.js-upgrade';
  import Globe from './lib/Globe.svelte';

  interface ClimateData {
    surface: number[][][]; // [month][lat][lon]
    [key: string]: number[][][];
  }

  let data: ClimateData | null = null;
  let month = 0;
  let loading = true;
  let error: string | null = null;
  let playing = true;
  let autoPlayInterval: number | null = null;
  let globeComponent: Globe;
  let recording = false;
  let recordingProgress = '';

  const MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  function startPlaying() {
    if (autoPlayInterval) return;
    playing = true;
    autoPlayInterval = setInterval(() => {
      month = (month + 1) % 12;
    }, 1000);
    if (globeComponent) {
      globeComponent.setAutoRotate(true);
    }
  }

  function stopPlaying() {
    if (autoPlayInterval) {
      clearInterval(autoPlayInterval);
      autoPlayInterval = null;
    }
    playing = false;
    if (globeComponent) {
      globeComponent.setAutoRotate(false);
    }
  }

  function togglePlay() {
    if (playing) {
      stopPlaying();
    } else {
      // Reset view when resuming play
      if (globeComponent) {
        globeComponent.resetView();
      }
      month = 0;
      startPlaying();
    }
  }

  async function recordGif() {
    if (recording || !globeComponent) return;

    stopPlaying();
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
      month = Math.floor(i / framesPerMonth);
      recordingProgress = `Capturing ${MONTH_NAMES[month]}... (${i + 1}/${totalFrames})`;

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
      const response = await fetch('/main.json');
      if (!response.ok) {
        throw new Error(`Failed to load data: ${response.status}`);
      }
      data = await response.json();
      loading = false;
      startPlaying();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
      loading = false;
    }

    return () => {
      if (autoPlayInterval) clearInterval(autoPlayInterval);
    };
  });
</script>

<main>
  {#if loading}
    <div class="loading">Loading climate data...</div>
  {:else if error}
    <div class="error">Error: {error}</div>
  {:else if data}
    <div class="globe-wrapper">
      <Globe bind:this={globeComponent} data={data.temperature_2m} {month} on:interact={stopPlaying} />
    </div>
    <div class="controls">
      <label>
        <span class="month-label">{MONTH_NAMES[month]}</span>
        <input
          type="range"
          min="0"
          max="11"
          bind:value={month}
          on:input={stopPlaying}
        />
      </label>
      <button class="play-button" on:click={togglePlay} disabled={recording}>
        {#if playing}
          Pause
        {:else}
          Play
        {/if}
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
    padding: 1rem;
    padding-bottom: 2rem;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    background: rgba(0, 0, 0, 0.8);
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
    min-width: 120px;
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
