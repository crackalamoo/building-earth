/**
 * Device-aware rendering settings. Mobile devices get aggressive downscaling
 * because their GPUs and memory budgets are much tighter than desktops.
 * Desktop Safari also gets a lower pixel-ratio cap because its compositor
 * can't keep up with retina rendering at large canvas sizes the way Chrome's
 * Angle backend can.
 *
 * Mobile detection is via user agent (matches loadBinaryData.ts).
 */
const IS_MOBILE: boolean =
  typeof navigator !== 'undefined' &&
  /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

const IS_SAFARI: boolean =
  typeof navigator !== 'undefined' &&
  /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

export const rendering = {
  particleCount: IS_MOBILE ? 6000 : 15000,
  trailLength: 30,
  pixelRatioCap: IS_MOBILE ? 1 : (IS_SAFARI ? 1 : 2),
};
