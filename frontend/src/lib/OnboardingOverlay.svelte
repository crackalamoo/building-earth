<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { useImperial } from './stores';
  import type { Stage } from './onboardingState';

  export let stage: Stage;
  export let loading = false;
  export let buttonLabel: string | null = null;
  export let locationDismissed = false;
  // Mobile-only: when true, hide the writeup and show only a compact bar
  // with the next/skip controls plus a "Read again" button. App.svelte
  // toggles this on globe interaction; it resets each stage change.
  export let collapsed = false;

  const dispatch = createEventDispatcher();

  function cToF(c: number): string { return (c * 9 / 5 + 32).toFixed(0); }
  function deltaC(d: number): string { return d.toFixed(0); }
  function deltaF(d: number): string { return (d * 9 / 5).toFixed(0); }

  interface TemperatureValue {
    celsius: number;
    isDelta: boolean;
  }

  // Parse template markers like {70_C} and {40_delta} from writeup text.
  // Each writeup has a `teaser` (one sentence shown by default on mobile)
  // and a full `text` that the user can opt into reading.
  const WRITEUPS: Record<number, { teaser: string; text: string; values: Map<string, TemperatureValue> }> = {
    1: (() => {
      const values = new Map<string, TemperatureValue>([
        ['8_C', { celsius: 8, isDelta: false }],
        ['-170_C', { celsius: -170, isDelta: false }],
        ['-9_C', { celsius: -9, isDelta: false }],
      ]);
      return {
        teaser: "This is Earth as a bare rock in space with no air — sunlight in, heat out. The poles freeze in months of darkness; even at the equator, 12 hours of night is enough to chill the air.",
        text: "This is Earth with no air — just bare water and rock in space. The only thing setting the temperature is sunlight hitting the surface and heat radiating back out.\n\nThe equator gets the most direct sunlight, but even there the surface only averages {8_C} — because half the time it's night, and all the heat escapes immediately. The poles spend months in complete darkness and plunge to {-170_C} in winter. The global average is {-9_C}, well below freezing.\n\nNothing could live here. The planet needs something to hold onto its heat.",
        values,
      };
    })(),
    2: (() => {
      const values = new Map<string, TemperatureValue>([
        ['-9_C', { celsius: -9, isDelta: false }],
        ['10_C', { celsius: 10, isDelta: false }],
        ['19_delta', { celsius: 19, isDelta: true }],
        ['40_C', { celsius: 40, isDelta: false }],
        ['-50_C', { celsius: -50, isDelta: false }],
        ['17_C', { celsius: 17, isDelta: false }],
      ]);
      return {
        teaser: "Air traps the heat trying to escape — the greenhouse effect. But there's no wind to spread it around.",
        text: "Now there's air. Gases like CO₂ and water vapor are transparent to sunlight, but they absorb the heat that the surface radiates back. The air warms up, and radiates heat back down. This is the **greenhouse effect**.\n\nThe global average jumps {19_delta}, from {-9_C} to {10_C} — already most of the way to the real Earth's average of {17_C}. But the tropics hit {40_C} because there are no clouds to block sunlight and no wind to carry the excess heat away. Meanwhile the poles plunge to {-50_C}.\n\nThe atmosphere traps heat, but it can't move it. Every spot on the planet is on its own.",
        values,
      };
    })(),
    3: (() => {
      const values = new Map<string, TemperatureValue>([
        ['19_C', { celsius: 19, isDelta: false }],
        ['-27_C', { celsius: -27, isDelta: false }],
        ['7_C', { celsius: 7, isDelta: false }],
      ]);
      return {
        teaser: "Air spills from hot to cold up high, and the reverse down low, creating wind. Eddies carry heat from the tropics toward the poles.",
        text: "Hot air expands, and cold air shrinks. Where a column of air is hot, it expands so much that air starts spilling out to adjacent areas. Conversely, air from elsewhere flows into cold areas, where the air has shrunk. This is how temperature differences create pressure differences, and pressure differences create wind.\n\nWhere extreme heat over land meets warm oceans, this contributes to summer monsoons, most famously the Indian monsoon. Warm, moist ocean air rushes into the land.\n\nCold air sinks at the poles and flows toward the warm equator, where air is rising. The spinning Earth deflects this flow sideways — this is the **Coriolis effect** — so surface winds across most of the globe blow from the east.\n\nWind and turbulent eddies carry heat from the tropics toward the poles. Water evaporates over warm ocean, forms clouds, and falls as rain. Snow piles up in the cold regions and reflects sunlight back to space, keeping them cold.\n\nThe equator cools to {19_C} as heat is carried poleward. The poles warm to {-27_C}. But the global average actually drops to {7_C} — clouds now block incoming sunlight, and widespread snow reflects it. The familiar deserts are missing — the Sahara gets more rain than the Amazon.",
        values,
      };
    })(),
    4: (() => {
      const values = new Map<string, TemperatureValue>([
        ['13_C', { celsius: 13, isDelta: false }],
      ]);
      return {
        teaser: "Hot air rises at the equator and sinks at 30°. The desert belt is born.",
        text: "Hot air rises near the equator, creating a belt of low pressure around the hottest parts of the globe. It flows poleward, and by around 30° latitude it piles up and sinks back to the surface. This giant loop is the **Hadley cell**.\n\nAs the air sinks, it's squeezed by the atmosphere above and warms up. Warm air can hold much more moisture before saturating, so clouds can't form and rain doesn't fall. This is why the Sahara, the Sonoran, and the Australian outback all sit near 30°. At the surface, the return flow blows back toward the equator where pressure is lower — the **trade winds**.\n\nWhere trade winds from each hemisphere collide, air rises and produces an intense rain belt that feeds the world's tropical rainforests and monsoon regions. The global average warms back to {13_C}.",
        values,
      };
    })(),
    5: (() => {
      const values = new Map<string, TemperatureValue>([
        ['16_C', { celsius: 16, isDelta: false }],
        ['17_C', { celsius: 17, isDelta: false }],
        ['1_delta', { celsius: 1, isDelta: true }],
      ]);
      return {
        teaser: "Ocean currents move heat around the planet. The machine is complete.",
        text: "Ocean currents complete the picture. The Gulf Stream carries warm tropical water toward Northern Europe, keeping London and Paris mild despite sitting at the same latitude as Labrador. Cold currents along the west coasts of continents cool the air and bring fog to San Francisco and the Atacama.\n\nBiological life isn't just a passive responder to the climate. Trees and plants pump enormous amounts of water into the air through their leaves, far more than bare soil alone would evaporate, especially in rainforests. And over just the past century, humans have begun to change the climate faster than anything in millions of years.\n\nOur final global average of {16_C} is within {1_delta} of the observed {17_C} — not bad for a model built from first principles.\n\nThe machine is complete. Sunlight, air, water, and ice — all connected, all responding to each other. This is the climate system you live inside.",
        values,
      };
    })(),
  };

  function formatValue(key: string, tv: TemperatureValue, imperial: boolean): string {
    if (tv.isDelta) {
      return imperial ? `${deltaF(tv.celsius)}°F` : `${deltaC(tv.celsius)}°C`;
    }
    return imperial ? `${cToF(tv.celsius)}°F` : `${tv.celsius}°C`;
  }

  // Split writeup text into segments: plain text, temperature tokens, or bold terms
  type Segment =
    | { type: 'text'; content: string }
    | { type: 'value'; key: string }
    | { type: 'bold'; content: string };

  function parseWriteup(stage: number): Segment[] {
    const info = WRITEUPS[stage];
    if (!info) return [];
    const parts: Segment[] = [];
    // Match either {value_key} or **bold text**
    const regex = /\{([^}]+)\}|\*\*([^*]+)\*\*/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(info.text)) !== null) {
      if (match.index > lastIndex) {
        parts.push({ type: 'text', content: info.text.slice(lastIndex, match.index) });
      }
      if (match[1] !== undefined) {
        parts.push({ type: 'value', key: match[1] });
      } else {
        parts.push({ type: 'bold', content: match[2] });
      }
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

  $: visible = stage >= 1 && stage <= 5 && !loading && !(stage === 5 && locationDismissed);
</script>

{#if visible && writeupInfo}
  <div
    class="backdrop"
    class:visible={!collapsed}
    on:click={() => dispatch('collapse')}
    role="presentation"
  />
  <div
    class="overlay"
    class:fade-in={true}
    class:collapsed
    on:click|stopPropagation
    role="dialog"
    aria-modal={!collapsed}
  >
    {#if collapsed}
      <p class="teaser">
        {writeupInfo.teaser}
        <button class="read-link" on:click={() => dispatch('expand')}>Read more</button>
      </p>
    {:else}
      <div class="header">
        <button
          class="close-btn"
          aria-label="Close"
          on:click={() => dispatch('collapse')}
        >
          ×
        </button>
      </div>
      <div class="writeup">
        {#each segments as seg}
          {#if seg.type === 'text'}
            {#each seg.content.split('\n\n') as para, i}
              {#if i > 0}<br /><br />{/if}{para}
            {/each}
          {:else if seg.type === 'bold'}
            <strong>{seg.content}</strong>
          {:else if writeupInfo.values.has(seg.key)}
            <button class="value-toggle" on:click={toggleUnits}>
              {formatValue(seg.key, writeupInfo.values.get(seg.key), imperial)}
            </button>
          {:else}
            {seg.key}
          {/if}
        {/each}
      </div>
    {/if}
    {#if buttonLabel}
      <button class="next-btn" on:click={() => dispatch('advance')} disabled={loading}>
        {#if loading}
          Loading...
        {:else}
          {buttonLabel}
        {/if}
      </button>
    {/if}
    {#if stage === 1 && !collapsed}
      <span class="explore-hint">Try clicking the globe</span>
    {/if}
    {#if stage < 4}
      <button class="skip-link" on:click={() => dispatch('skip')}>
        Skip to full model
      </button>
    {/if}
    {#if stage === 5}
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
    width: min(600px, 90vw);
    max-height: calc(100vh - 120px);
    max-height: calc(100dvh - 120px);
    padding: 1.5rem;
    padding-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    z-index: 5;
    background: rgba(0, 0, 0, 0.35);
    border-left: 1px solid rgba(26, 107, 107, 0.35);
    border-bottom: 1px solid rgba(26, 107, 107, 0.35);
    border-bottom-left-radius: 6px;
    animation: fadeIn 0.6s ease;
    overflow-y: scroll;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
    scrollbar-color: #2a9e9e rgba(0, 0, 0, 0.3);
  }

  .overlay::-webkit-scrollbar,
  .writeup::-webkit-scrollbar {
    width: 8px;
  }

  .overlay::-webkit-scrollbar-track,
  .writeup::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
  }

  .overlay::-webkit-scrollbar-thumb,
  .writeup::-webkit-scrollbar-thumb {
    background: #2a9e9e;
    border-radius: 4px;
  }

  .overlay::-webkit-scrollbar-thumb:hover,
  .writeup::-webkit-scrollbar-thumb:hover {
    background: #3fc0c0;
  }

  /* Desktop: slide in from the right edge where the overlay sits */
  @keyframes fadeIn {
    from { opacity: 0; transform: translateX(24px); }
    to { opacity: 1; transform: translateX(0); }
  }

  /* Mobile collapsed teaser: slide down from the top edge */
  @keyframes fadeInMobile {
    from { opacity: 0; transform: translateY(-24px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* Mobile expanded modal: just fade in (the modal uses translate(-50%, -50%)
     for centering, so animating transform here would fight that). */
  @keyframes modalFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
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

  .read-link {
    align-self: flex-start;
    background: none;
    border: none;
    color: #3fc0c0;
    font-size: 0.95rem;
    cursor: pointer;
    padding: 0;
    text-decoration: underline;
    text-underline-offset: 2px;
    transition: color 0.15s;
  }

  .read-link:hover {
    color: #5ddcdc;
  }

  /* Backdrop sits behind the modal-style overlay on mobile. Hidden on
     desktop and when the overlay is in collapsed teaser mode. */
  .backdrop {
    display: none;
  }

  /* Header row (mobile-only) holding the close button so it sits above the
     writeup text instead of overlapping it. The header is hidden on desktop. */
  .header {
    display: none;
  }

  .teaser {
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.4;
    color: rgba(255, 255, 255, 0.9);
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.9);
    flex: 1 1 auto;
    min-width: 0;
  }

  .close-btn {
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.8);
    font-size: 1.6rem;
    line-height: 1;
    cursor: pointer;
    padding: 0;
    transition: color 0.15s;
  }

  .close-btn:hover {
    color: rgba(255, 255, 255, 0.95);
  }

  @media (max-width: 800px), (max-height: 500px) {
    /* Mobile expanded: modal with backdrop, centered. Matches About modal. */
    .backdrop.visible {
      display: block;
      position: fixed;
      inset: 0;
      z-index: 99;
      background: rgba(0, 0, 0, 0.6);
    }

    .overlay {
      /* Default modal-style positioning (used when expanded on mobile) */
      position: fixed;
      top: 50%;
      left: 50%;
      right: auto;
      transform: translate(-50%, -50%);
      width: min(480px, 90vw);
      max-height: calc(100vh - 4rem);
      max-height: calc(100dvh - 4rem);
      padding: 1rem;
      background: rgba(0, 0, 0, 0.92);
      border: 1px solid #1a6b6b;
      border-radius: 0;
      z-index: 100;
      overflow: hidden;
    }

    /* Mobile expanded modal: cancel the desktop right-slide animation
       (it conflicts with the center transform) and just fade in. */
    .overlay {
      animation: modalFadeIn 0.3s ease;
    }

    /* Collapsed: small teaser bar pinned to the top of the screen */
    .overlay.collapsed {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      transform: none;
      width: 100vw;
      max-height: none;
      animation: fadeInMobile 0.4s ease;
      padding: 0.75rem 1rem;
      flex-direction: row;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.6rem 0.75rem;
      background: rgba(0, 0, 0, 0.55);
      border: none;
      border-bottom: 1px solid rgba(26, 107, 107, 0.35);
      overflow: visible;
    }

    .overlay.collapsed .next-btn {
      align-self: center;
      padding: 0.5rem 1rem;
      font-size: 0.9rem;
    }

    .overlay.collapsed .skip-link {
      display: none;
    }

    .overlay.collapsed .read-link {
      align-self: baseline;
      white-space: nowrap;
    }

    .header {
      display: flex;
      justify-content: flex-end;
      align-items: center;
      flex-shrink: 0;
      margin: -0.5rem -0.25rem 0 0;
    }

    .overlay.collapsed .header {
      display: none;
    }

    .writeup {
      font-size: 0.95rem;
      flex: 1 1 auto;
      min-height: 0;
      overflow-y: auto;
      padding-right: 0.5rem;
      text-shadow: none;
      color: rgba(255, 255, 255, 0.95);
      scrollbar-width: thin;
      scrollbar-color: #2a9e9e rgba(0, 0, 0, 0.3);
    }

    .next-btn,
    .skip-link,
    .header {
      flex-shrink: 0;
    }
  }
</style>
