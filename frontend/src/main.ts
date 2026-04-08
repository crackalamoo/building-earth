import { mount } from 'svelte'
import 'katex/dist/katex.min.css'
import './app.css'
import App from './App.svelte'

// Block iOS Safari double-tap-zoom and pinch-zoom that CSS touch-action and
// the viewport meta refuse to suppress on iOS 17+. iOS fires non-standard
// `gesturestart` / `gesturechange` / `gestureend` events for any multi-touch
// or smart-zoom gesture; preventDefault on gesturestart kills the whole
// gesture before it can scale the page.
document.addEventListener('gesturestart', (e) => e.preventDefault());
document.addEventListener('gesturechange', (e) => e.preventDefault());

// Defense in depth: also intercept the synthetic `dblclick` that iOS fires
// in tandem with double-tap-zoom on text. preventDefault here cancels the
// associated zoom on most iOS versions.
document.addEventListener('dblclick', (e) => e.preventDefault());

const app = mount(App, {
  target: document.getElementById('app')!,
})

export default app
