<script lang="ts">
  import { X } from 'lucide-svelte';
  import { createEventDispatcher } from 'svelte';

  export let open = false;
  const dispatch = createEventDispatcher();

  function close() {
    open = false;
    dispatch('close');
  }
</script>

{#if open}
  <div class="backdrop" on:click={close} role="presentation">
    <div class="modal" on:click|stopPropagation role="dialog" aria-modal="true">
      <div class="modal-header">
        <span class="modal-title">About</span>
        <button class="close-btn" on:click={close}><X size={16} /></button>
      </div>
      <div class="modal-body">
        <p>A first-principles climate simulation built from scratch with a beautiful visualizer and an AI explainer.</p>
        <a class="github-link" href="https://github.com/crackalamoo/2025-proj0" target="_blank" rel="noopener">View source on GitHub</a>
        <div class="section-heading">My Inspirations</div>
        <p>A heartfelt thanks to all of the following:</p>
        <ul>
          <li><a href="https://earth.nullschool.net" target="_blank" rel="noopener">earth.nullschool.net</a> — real-time global weather conditions with mesmerizing wind particles and a wonderfully interactive UX</li>
          <li><a href="https://weatherspark.com" target="_blank" rel="noopener">WeatherSpark</a> — climate comparison charts full of interesting data</li>
          <li><a href="https://apps.apple.com/us/app/weather/id1069513131" target="_blank" rel="noopener">The iOS Weather app</a> — an absolutely stunning, unnecessarily beautiful display of the weather forecast</li>
          <li><a href="https://earth.google.com" target="_blank" rel="noopener">Google Earth</a> — the original beautiful globe, with insane detail</li>
          <li>California and its microclimates</li>
        </ul>
      </div>
    </div>
  </div>
{/if}

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    z-index: 100;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .modal {
    position: relative;
    background: rgba(0, 0, 0, 0.92);
    border: 1px solid #1a6b6b;
    width: min(480px, 90vw);
    color: rgba(255, 255, 255, 0.88);
    font-size: 1rem;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #333;
  }

  .modal-title {
    font-size: 1rem;
    font-weight: 600;
    color: #fff;
  }

  .modal-body {
    padding: 1rem;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.95);
  }

  p {
    margin: 0 0 1rem;
  }

  .github-link {
    display: inline-block;
    margin-bottom: 1.5rem;
    color: #3fc0c0;
    font-size: 0.9rem;
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  .github-link:hover {
    color: #5ddcdc;
  }

  .section-heading {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.75);
    margin-bottom: 0.5rem;
  }

  ul {
    margin: 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    font-size: 0.9rem;
  }

  li {
    color: rgba(255, 255, 255, 0.9);
    padding-left: 1.25rem;
    position: relative;
  }

  li::before {
    content: '-';
    position: absolute;
    left: 0;
    color: rgba(255, 255, 255, 0.35);
  }

  li a {
    color: #3fc0c0;
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  li a:hover {
    color: #5ddcdc;
  }

  .close-btn {
    width: 28px;
    height: 28px;
    padding: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    border-radius: 4px;
    transition: color 0.15s, background 0.15s;
  }

  .close-btn:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.1);
  }
</style>
