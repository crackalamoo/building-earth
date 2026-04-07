/**
 * Device-aware rendering settings. Mobile devices get aggressive downscaling
 * because their GPUs and memory budgets are much tighter than desktops.
 *
 * Mobile detection is via user agent (matches loadBinaryData.ts).
 */
const IS_MOBILE: boolean =
  typeof navigator !== 'undefined' &&
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

export const rendering = {
  particleCount: IS_MOBILE ? 6000 : 15000,
  trailLength: 30,
  pixelRatioCap: IS_MOBILE ? 1 : 2,
};
