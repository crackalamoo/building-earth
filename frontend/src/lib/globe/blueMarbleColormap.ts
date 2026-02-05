/**
 * Blue Marble colormap — stylized Pixar-style earth with vegetation,
 * ocean colors, and temperature-driven snow/ice.
 *
 * Snow/ice logic replicates backend/climate_sim/physics/snow_albedo.py
 * so no albedo field is needed.
 */

// ── Color palette ──────────────────────────────────────────────────

// Ocean base (deep saturated blue, slightly brighter in tropics)
const OCEAN_DEEP: [number, number, number] = [0.04, 0.10, 0.30];
const OCEAN_SHALLOW: [number, number, number] = [0.06, 0.18, 0.42];

// Land soil-moisture gradient (dry → wet)
const ARID_SAND: [number, number, number] = [0.82, 0.72, 0.50];
const DRY_SCRUB: [number, number, number] = [0.62, 0.58, 0.32];
const TEMPERATE_GREEN: [number, number, number] = [0.28, 0.50, 0.20];
const LUSH_GREEN: [number, number, number] = [0.10, 0.40, 0.12];

// Snow/ice colors
const SEASONAL_SNOW: [number, number, number] = [0.88, 0.90, 0.92];
const ICE_SHEET: [number, number, number] = [0.95, 0.97, 1.00];
const SEA_ICE: [number, number, number] = [0.80, 0.88, 0.95];

// ── Snow/ice parameters (matching backend) ─────────────────────────

// Land snow fraction: Hermite smoothstep
const SNOW_FREEZE_C = -5.0;
const SNOW_MELT_C = 1.0;

// Sea ice fraction: Hermite smoothstep
const ICE_FREEZE_C = -1.8; // seawater freezing point
const ICE_FULL_C = -8.0;
const ICE_MAX_FRACTION = 0.70;

// Ice sheet vs seasonal snow discrimination
const ICE_SHEET_TEMP_C = -35.0;
const SEASONAL_SNOW_TEMP_C = -15.0;

// ── Helpers ────────────────────────────────────────────────────────

function hermite(u: number): number {
  return u * u * (3 - 2 * u);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function lerp3(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
): [number, number, number] {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

// ── Snow/ice computation ───────────────────────────────────────────

/** Land snow fraction from surface temp (matches backend). */
function landSnowFraction(tempC: number): number {
  const u = clamp01((SNOW_MELT_C - tempC) / (SNOW_MELT_C - SNOW_FREEZE_C));
  return hermite(u);
}

/** Sea ice fraction from surface temp (matches backend). */
function seaIceFraction(tempC: number): number {
  const u = clamp01((ICE_FREEZE_C - tempC) / (ICE_FREEZE_C - ICE_FULL_C));
  return hermite(u) * ICE_MAX_FRACTION;
}

/** Blend fraction between ice sheet (1) and seasonal snow (0). */
function iceSheetFraction(tempC: number): number {
  const u = clamp01((SEASONAL_SNOW_TEMP_C - tempC) / (SEASONAL_SNOW_TEMP_C - ICE_SHEET_TEMP_C));
  return hermite(u);
}

// ── Public colormap ────────────────────────────────────────────────

/**
 * Compute Blue Marble color for a single vertex.
 *
 * @param isLand       - true if land cell
 * @param surfaceTempC - surface temperature in Celsius (native 5deg)
 * @param soilMoisture - soil moisture 0-1 fraction of field capacity (native 5deg)
 * @returns [r, g, b] normalized 0-1
 */
export function blueMarbleColor(
  isLand: boolean,
  surfaceTempC: number,
  soilMoisture: number,
): [number, number, number] {
  if (!isLand) {
    // ── Ocean ──
    // Warmer water is slightly lighter/shallower looking
    const warmth = clamp01((surfaceTempC + 2) / 30);
    let color = lerp3(OCEAN_DEEP, OCEAN_SHALLOW, warmth * 0.4);

    // Sea ice overlay
    const iceFrac = seaIceFraction(surfaceTempC);
    if (iceFrac > 0.001) {
      color = lerp3(color, SEA_ICE, iceFrac);
    }

    return color;
  }

  // ── Land ──
  // Base color from soil moisture (0 = bone dry → 1 = saturated)
  let baseColor: [number, number, number];

  if (soilMoisture < 0.1) {
    // Arid desert (sandy tan)
    baseColor = lerp3(ARID_SAND, DRY_SCRUB, soilMoisture / 0.1);
  } else if (soilMoisture < 0.3) {
    // Dry scrubland (muted yellow-brown)
    const t = (soilMoisture - 0.1) / 0.2;
    baseColor = lerp3(DRY_SCRUB, TEMPERATE_GREEN, t);
  } else if (soilMoisture < 0.6) {
    // Temperate grassland/soil (olive → green)
    const t = (soilMoisture - 0.3) / 0.3;
    baseColor = lerp3(TEMPERATE_GREEN, LUSH_GREEN, t);
  } else {
    // Lush/wet (rich green)
    baseColor = LUSH_GREEN;
  }

  // Snow overlay
  const snowFrac = landSnowFraction(surfaceTempC);
  if (snowFrac > 0.001) {
    // Determine snow type: ice sheet vs seasonal
    const iceSheetFrac = iceSheetFraction(surfaceTempC);
    const snowColor = lerp3(SEASONAL_SNOW, ICE_SHEET, iceSheetFrac);
    baseColor = lerp3(baseColor, snowColor, snowFrac);
  }

  return baseColor;
}
