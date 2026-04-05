/**
 * Precipitation colormap.
 * Maps precipitation in kg/m²/s to RGB colors via mm/day conversion.
 */

interface ColorStop {
  mmday: number;
  color: [number, number, number]; // RGB 0-255
}

const COLOR_STOPS: ColorStop[] = [
  { mmday: 0,  color: [210, 200, 180] }, // dry tan
  { mmday: 1,  color: [180, 210, 170] }, // pale green
  { mmday: 3,  color: [100, 190, 120] }, // green
  { mmday: 6,  color: [40, 150, 100] },  // dark green
  { mmday: 10, color: [30, 100, 140] },  // teal-blue
  { mmday: 15, color: [20, 50, 120] },   // deep blue
];

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
 * Get RGB [0-255] for a precipitation value in mm/day.
 */
export function precipMmdayToColor(mmday: number): [number, number, number] {
  const clamped = Math.min(Math.max(0, mmday), COLOR_STOPS[COLOR_STOPS.length - 1].mmday);
  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const s0 = COLOR_STOPS[i];
    const s1 = COLOR_STOPS[i + 1];
    if (clamped >= s0.mmday && clamped <= s1.mmday) {
      const range = s1.mmday - s0.mmday;
      const frac = range > 0 ? (clamped - s0.mmday) / range : 0;
      return lerpColor(s0.color, s1.color, frac);
    }
  }
  return COLOR_STOPS[COLOR_STOPS.length - 1].color;
}

/**
 * Get normalized RGB [0-1] for a precipitation value in kg/m²/s.
 */
export function precipitationToColorNormalized(kgPerM2PerS: number): [number, number, number] {
  const mmday = Math.max(0, kgPerM2PerS * 86400);
  const clamped = Math.min(mmday, COLOR_STOPS[COLOR_STOPS.length - 1].mmday);

  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const s0 = COLOR_STOPS[i];
    const s1 = COLOR_STOPS[i + 1];
    if (clamped >= s0.mmday && clamped <= s1.mmday) {
      const range = s1.mmday - s0.mmday;
      const frac = range > 0 ? (clamped - s0.mmday) / range : 0;
      const [r, g, b] = lerpColor(s0.color, s1.color, frac);
      return [r / 255, g / 255, b / 255];
    }
  }

  const last = COLOR_STOPS[COLOR_STOPS.length - 1].color;
  return [last[0] / 255, last[1] / 255, last[2] / 255];
}
