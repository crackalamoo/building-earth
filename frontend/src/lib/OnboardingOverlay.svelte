<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { useImperial } from './stores';
  import type { Stage } from './onboardingState';

  export let stage: Stage;
  export let loading = false;
  export let buttonLabel: string | null = null;
  export let locationDismissed = false;

  const dispatch = createEventDispatcher();

  function cToF(c: number): string { return (c * 9 / 5 + 32).toFixed(0); }
  function deltaC(d: number): string { return d.toFixed(0); }
  function deltaF(d: number): string { return (d * 9 / 5).toFixed(0); }

  interface TemperatureValue {
    celsius: number;
    isDelta: boolean;
  }

  // Parse template markers like {70_C} and {40_delta} from writeup text
  const WRITEUPS: Record<number, { text: string; values: Map<string, TemperatureValue> }> = {
    1: (() => {
      const values = new Map<string, TemperatureValue>([
        ['10_C', { celsius: 10, isDelta: false }],
        ['-168_C', { celsius: -168, isDelta: false }],
        ['-21_C', { celsius: -21, isDelta: false }],
      ]);
      return {
        text: "This is Earth with no air — just bare rock in space. The only thing setting the temperature is sunlight hitting the surface and heat radiating back out.\n\nThe equator gets the most direct sunlight, but even there the surface only averages {10_C} — because half the time it's night, and all the heat escapes immediately. The poles spend months in darkness and drop to {-168_C}. The global average is {-21_C}, well below freezing.\n\nNothing could live here. The planet needs something to hold onto its heat.",
        values,
      };
    })(),
    2: (() => {
      const values = new Map<string, TemperatureValue>([
        ['-21_C', { celsius: -21, isDelta: false }],
        ['19_C', { celsius: 19, isDelta: false }],
        ['40_delta', { celsius: 40, isDelta: true }],
        ['70_C', { celsius: 70, isDelta: false }],
        ['15_C', { celsius: 15, isDelta: false }],
        ['4_delta', { celsius: 4, isDelta: true }],
      ]);
      return {
        text: "Now there's air. Gases like CO₂ and water vapor let sunlight through but trap the heat trying to escape — like a blanket around the planet. This is the greenhouse effect.\n\nThe global average jumps {40_delta}, from {-21_C} to {19_C} — just {4_delta} above the real Earth's average of {15_C}. But look at the tropics: they hit {70_C} because there are no clouds to block sunlight and no wind to carry the heat away. Meanwhile the poles are still frozen.\n\nThe atmosphere traps heat, but it can't move it. Every spot on the planet is on its own.",
        values,
      };
    })(),
    3: (() => {
      const values = new Map<string, TemperatureValue>([
        ['29_C', { celsius: 29, isDelta: false }],
        ['-7_C', { celsius: -7, isDelta: false }],
      ]);
      return {
        text: "Hot air is lighter than cold air. It rises at the equator, and cold air sinks at the poles — this creates wind.\n\nBut because the Earth is spinning, the wind doesn't blow straight from equator to pole. It curves, creating the trade winds near the tropics and the westerly winds that bring weather across Europe and North America.\n\nThis circulation carries heat from the tropics toward the poles. The equator cools to {29_C}. The poles warm to {-7_C}. The extreme temperature gap shrinks dramatically, and the map starts to look like the planet you know.",
        values,
      };
    })(),
    4: (() => {
      return {
        text: "Wind over warm ocean picks up water vapor. That moist air rises, cools, and forms clouds. Rain falls. This cycle moves huge amounts of energy — every bit of evaporation absorbs heat from the surface and releases it high in the atmosphere.\n\nOcean currents join in. The Gulf Stream pushes warm tropical water toward Northern Europe, keeping London milder than Montreal despite being farther north. Snow and ice reflect sunlight back to space, keeping the poles cold.\n\nThe machine is complete. Sunlight, air, water, and ice — all connected, all responding to each other, creating the climate you live inside.",
        values: new Map(),
      };
    })(),
  };

  function formatValue(key: string, tv: TemperatureValue, imperial: boolean): string {
    if (tv.isDelta) {
      return imperial ? `${deltaF(tv.celsius)}°F` : `${deltaC(tv.celsius)}°C`;
    }
    return imperial ? `${cToF(tv.celsius)}°F` : `${tv.celsius}°C`;
  }

  // Split writeup text into segments: plain text + temperature tokens
  type Segment = { type: 'text'; content: string } | { type: 'value'; key: string };

  function parseWriteup(stage: number): Segment[] {
    const info = WRITEUPS[stage];
    if (!info) return [];
    const parts: Segment[] = [];
    const regex = /\{([^}]+)\}/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(info.text)) !== null) {
      if (match.index > lastIndex) {
        parts.push({ type: 'text', content: info.text.slice(lastIndex, match.index) });
      }
      parts.push({ type: 'value', key: match[1] });
      lastIndex = regex.lastIndex;
    }
    if (lastIndex < info.text.length) {
      parts.push({ type: 'text', content: info.text.slice(lastIndex) });
    }
    return parts;
  }

  $: segments = parseWriteup(stage);
  $: writeupInfo = WRITEUPS[stage];
  $: imperial = $useImperial;

  function toggleUnits() {
    useImperial.update(v => !v);
  }

  $: visible = stage >= 1 && stage <= 4 && !loading && !(stage === 4 && locationDismissed);
</script>

{#if visible && writeupInfo}
  <div class="overlay" class:fade-in={true}>
    <div class="writeup">
      {#each segments as seg}
        {#if seg.type === 'text'}
          {#each seg.content.split('\n\n') as para, i}
            {#if i > 0}<br /><br />{/if}{para}
          {/each}
        {:else if writeupInfo.values.has(seg.key)}
          <button class="value-toggle" on:click={toggleUnits}>
            {formatValue(seg.key, writeupInfo.values.get(seg.key), imperial)}
          </button>
        {:else}
          {seg.key}
        {/if}
      {/each}
    </div>
    {#if buttonLabel}
      <button class="next-btn" on:click={() => dispatch('advance')} disabled={loading}>
        {#if loading}
          Loading...
        {:else}
          {buttonLabel}
        {/if}
      </button>
    {/if}
    {#if stage === 1}
      <span class="explore-hint">Try clicking the globe</span>
    {/if}
    {#if stage < 4}
      <button class="skip-link" on:click={() => dispatch('skip')}>
        Skip to full model
      </button>
    {/if}
    {#if stage === 4}
      <button class="next-btn" on:click={() => dispatch('locate')}>
        Where am I?
      </button>
    {/if}
  </div>
{/if}

<style>
  .overlay {
    position: absolute;
    top: 0;
    right: 0;
    width: min(400px, 90vw);
    max-height: calc(100vh - 120px);
    padding: 1.5rem;
    padding-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    z-index: 5;
    animation: fadeIn 0.6s ease;
    overflow-y: auto;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

.writeup {
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.88);
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.9);
  }

  .value-toggle {
    display: inline;
    background: none;
    border: none;
    border-bottom: 1px dashed rgba(42, 158, 158, 0.6);
    color: #3fc0c0;
    font-size: inherit;
    font-family: inherit;
    padding: 0;
    margin: 0;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }

  .value-toggle:hover {
    color: #5ddcdc;
    border-color: rgba(42, 158, 158, 0.9);
  }

  .next-btn {
    align-self: flex-start;
    padding: 0.7rem 1.6rem;
    font-size: 1rem;
    color: #fff;
    background: linear-gradient(to bottom, #1a6b6b, #0e4a4a);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 6px;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: background 0.2s, border-color 0.2s;
  }

  .next-btn:hover:not(:disabled) {
    background: linear-gradient(to bottom, #155a5a, #0a3838);
    border-color: rgba(255, 255, 255, 0.4);
  }

  .next-btn:disabled {
    opacity: 0.7;
    cursor: wait;
  }

  .skip-link {
    align-self: flex-start;
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.75);
    font-size: 0.875rem;
    cursor: pointer;
    padding: 0;
    text-decoration: underline;
    text-underline-offset: 2px;
    transition: color 0.15s;
  }

  .skip-link:hover {
    color: rgba(255, 255, 255, 0.9);
  }

  .explore-hint {
    font-size: 0.875rem;
    color: rgba(255, 255, 255, 0.45);
    font-style: italic;
  }

  @media (max-width: 640px), (max-height: 500px) {
    .overlay {
      width: 100vw;
      top: 0;
      right: 0;
      padding: 1rem;
      max-height: 50vh;
      overflow-y: hidden;
    }

    .writeup {
      font-size: 0.9rem;
      overflow-y: auto;
      flex-shrink: 1;
      min-height: 0;
    }
  }
</style>
