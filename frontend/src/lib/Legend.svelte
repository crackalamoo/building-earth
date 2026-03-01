<script lang="ts">
  export let stops: { value: number; color: string }[] = [];
  export let label: string = '';
  export let visible: boolean = false;
</script>

{#if visible}
  <div class="legend" class:visible>
    <div class="legend-label">{label}</div>
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
    border-radius: 6px;
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
    z-index: 5;
  }

  .legend.visible {
    opacity: 1;
  }

  .legend-label {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.7);
    letter-spacing: 0.03em;
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
    border: 1px solid rgba(255, 255, 255, 0.15);
  }

  .legend-ticks {
    display: flex;
    flex-direction: column-reverse;
    justify-content: space-between;
    height: 120px;
  }

  .tick {
    font-size: 0.65rem;
    color: rgba(255, 255, 255, 0.6);
    line-height: 1;
  }
</style>
