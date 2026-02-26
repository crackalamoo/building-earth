import { writable } from 'svelte/store';

/** When true, display imperial units (°F, ft, mph). When false, metric (°C, m, km/h). */
export const useImperial = writable(false);
