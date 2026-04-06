import { writable } from 'svelte/store';

/** Countries/regions that use imperial units (primarily US). */
const IMPERIAL_REGIONS = new Set(['US', 'LR', 'MM']);

function detectImperial(): boolean {
  try {
    const locale = navigator.language || '';
    const region = new Intl.Locale(locale).region;
    if (region && IMPERIAL_REGIONS.has(region)) return true;
  } catch {
    // Intl.Locale not supported or invalid locale — fall through to false
  }
  return false;
}

/** When true, display imperial units (°F, ft, mph). When false, metric (°C, m, km/h). */
export const useImperial = writable(detectImperial());
