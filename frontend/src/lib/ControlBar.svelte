<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Thermometer, Globe as GlobeIcon, CloudRain, SunMoon, Sun, Play, Pause, Home, Clapperboard } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  const MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  export let activeLayer: 'temperature' | 'precipitation' | 'blue-marble' = 'blue-marble';
  export let uniformLighting = false;
  export let monthProgress = 0;
  export let playing = false;
  export let recording = false;
  export let recordingProgress = '';
  export let layerDataLoaded = false;
  export let displayMonth = 0;
  export let visible = false;
  export let stage = 5;
  export let hasPrecipitation = false;
  export let hasSurface = false;
</script>

<div class="controls" class:visible>
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
      class:active={activeLayer === 'precipitation'}
      on:click={() => activeLayer = 'precipitation'}
      disabled={recording || !layerDataLoaded || !hasPrecipitation}
      data-tooltip="Precipitation"
      style="border-radius: 0; margin-left: -1px;"
    ><CloudRain size={16} /></button>
    <button
      class="layer-tab"
      class:active={activeLayer === 'blue-marble'}
      on:click={() => activeLayer = 'blue-marble'}
      disabled={recording || !layerDataLoaded || !hasSurface}
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
      on:input={() => dispatch('stopPlaying')}
    />
  </label>
  <button class="action-btn" on:click={() => dispatch('togglePlay')} disabled={recording} data-tooltip={playing ? 'Pause' : 'Play'}>
    {#if playing}
      <Pause size={16} />
    {:else}
      <Play size={16} />
    {/if}
  </button>
  <button class="action-btn" on:click={() => dispatch('resetView')} disabled={recording} data-tooltip="Reset View">
    <Home size={16} />
  </button>
  <button class="action-btn record-btn" on:click={() => dispatch('recordGif')} disabled={recording} class:hidden={stage < 5} data-tooltip={recording ? recordingProgress : 'Record GIF'}>
    <Clapperboard size={16} />
  </button>
</div>

<style>
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
    font-size: 0.875rem;
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
    height: 18px;
    background: transparent;
    outline: none;
    margin: 0;
    padding: 0;
  }

  /* WebKit (Safari, Chrome) needs the track styled via this pseudo-element.
     Setting background on the input itself does not draw a track in Safari. */
  input[type="range"]::-webkit-slider-runnable-track {
    height: 4px;
    background: #2a9e9e;
    border-radius: 2px;
    border: none;
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
    /* Center the thumb on the 4px track: (4 - 14) / 2 = -5 */
    margin-top: -5px;
  }

  input[type="range"]::-webkit-slider-thumb:hover {
    background: #3fc0c0;
  }

  input[type="range"]::-webkit-slider-thumb:active {
    background: #1a8080;
  }

  input[type="range"]::-moz-range-track {
    height: 4px;
    background: #2a9e9e;
    border-radius: 2px;
    border: none;
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
    font-size: 0.875rem;
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

  button:hover:not(:disabled):not(.action-btn) {
    background: rgba(14, 74, 74, 0.5);
  }

  button:disabled {
    cursor: not-allowed;
    opacity: 0.7;
  }

  @media (max-width: 640px), (max-height: 500px) {
    .controls {
      flex-wrap: wrap;
      gap: 0.75rem;
      padding: 0.5rem 0.75rem 1rem;
    }
    .separator { display: none; }
    .controls label {
      order: 10;
      width: 100%;
      gap: 0.5rem;
    }
    .month-label { min-width: 70px; font-size: 0.9rem; }
    input[type="range"] { width: auto; flex: 1; }
    .record-btn { display: none; }
  }

  .hidden {
    display: none;
  }
</style>
