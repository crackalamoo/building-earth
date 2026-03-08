# Frontend

Svelte + Three.js globe visualization.

## Structure

- `src/App.svelte`: main app with month animation controls
- `src/lib/globe/Globe.svelte`: Three.js globe component
- `src/lib/globe/`: also contains individual visual layer modules (atmosphere, sun, stars, clouds, trees, wind particles, city lights), geographic data helpers (elevation, borders), and data loading
- `src/lib/globe/colormap.ts`: temperature-to-color mapping
- `src/lib/globe/InspectPopup.svelte`: location info popup on click
