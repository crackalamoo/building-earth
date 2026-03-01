<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  export let stops: { value: string; color: string }[] = [];
  export let label: string = '';
  export let visible: boolean = false;

  const dispatch = createEventDispatcher();
</script>

{#if visible}
  <div class="legend" class:visible>
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <span class="unit-toggle" on:click={() => dispatch('toggleUnits')}>{label}</span>
    <div class="legend-bar-container">
      <div
        class="legend-bar"
        style="background: linear-gradient(to top, {stops.map(s => s.color).join(', ')});"
      ></div>
      <div class="legend-ticks">
        {#each stops as stop}
          <span class="tick">{stop.value}</span>
        {/each}
      </div>
    </div>
  </div>
{/if}

<style>
  .legend {
    position: absolute;
    bottom: 5rem;
    right: 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    padding: 0.5rem 0.6rem;
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(26, 107, 107, 0.4);
    min-width: 5rem;
    border-radius: 6px;
    opacity: 0;
    transition: opacity 0.4s ease;
    z-index: 10;
  }

  .legend.visible {
    opacity: 1;
  }

  .unit-toggle {
    font-size: 0.8rem;
    color: rgba(255, 255, 255, 0.7);
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: opacity 0.15s;
  }

  .unit-toggle:hover {
    opacity: 0.65;
  }

  .legend-bar-container {
    display: flex;
    gap: 0.4rem;
    align-items: stretch;
  }

  .legend-bar {
    width: 14px;
    height: 120px;
    border-radius: 2px;
  }

  .legend-ticks {
    display: flex;
    flex-direction: column-reverse;
    justify-content: space-between;
    height: 120px;
  }

  .tick {
    font-size: 0.75rem;
    color: rgba(255, 255, 255, 0.6);
    line-height: 1;
  }
</style>
