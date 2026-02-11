/**
 * Temperature colormap matching the matplotlib visualization.
 * Maps temperature in Celsius to RGB colors.
 */

interface ColorStop {
  temp: number;
  color: [number, number, number]; // RGB 0-255
}

// Matches build_temperature_cmap() in plotting.py
const COLOR_STOPS: ColorStop[] = [
  { temp: -30, color: [59, 30, 109] },   // #3B1E6D deep purple
  { temp: -15, color: [11, 30, 109] },   // #0B1E6D deep cold blue
  { temp: 0, color: [30, 136, 229] },    // #1E88E5 cooler blue (freezing)
  { temp: 0.01, color: [100, 181, 246] }, // #64B5F6 light blue (just above freezing)
  { temp: 10, color: [102, 187, 106] },  // #66BB6A green
  { temp: 21, color: [255, 235, 59] },   // #FFEB3B yellow
  { temp: 25, color: [251, 140, 0] },    // #FB8C00 orange
  { temp: 30, color: [211, 47, 47] },    // #D32F2F red
  { temp: 35, color: [181, 56, 42] },    // #B5382A deep red
  { temp: 40, color: [138, 0, 0] },      // #8A0000 darker red
];

const MIN_TEMP = COLOR_STOPS[0].temp;
const MAX_TEMP = COLOR_STOPS[COLOR_STOPS.length - 1].temp;

/**
 * Interpolate between two colors.
 */
function lerpColor(
  c1: [number, number, number],
  c2: [number, number, number],
  t: number
): [number, number, number] {
  return [
    Math.round(c1[0] + (c2[0] - c1[0]) * t),
    Math.round(c1[1] + (c2[1] - c1[1]) * t),
    Math.round(c1[2] + (c2[2] - c1[2]) * t),
  ];
}

/**
 * Get RGB color for a temperature value in Celsius.
 * Returns [r, g, b] with values 0-255.
 */
export function temperatureToColor(tempC: number): [number, number, number] {
  // Clamp to range
  const t = Math.max(MIN_TEMP, Math.min(MAX_TEMP, tempC));

  // Find surrounding color stops
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const stop1 = COLOR_STOPS[i];
    const stop2 = COLOR_STOPS[i + 1];

    if (t >= stop1.temp && t <= stop2.temp) {
      const range = stop2.temp - stop1.temp;
      const frac = range > 0 ? (t - stop1.temp) / range : 0;
      return lerpColor(stop1.color, stop2.color, frac);
    }
  }

  // Fallback (shouldn't reach here due to clamping)
  return COLOR_STOPS[COLOR_STOPS.length - 1].color;
}

/**
 * Get RGB color normalized to 0-1 range (for Three.js).
 */
export function temperatureToColorNormalized(tempC: number): [number, number, number] {
  const [r, g, b] = temperatureToColor(tempC);
  return [r / 255, g / 255, b / 255];
}
