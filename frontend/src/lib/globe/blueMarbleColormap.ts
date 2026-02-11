/**
 * Blue Marble colormap — stylized Pixar-style earth with vegetation,
 * ocean colors, and temperature-driven snow/ice.
 *
 * Snow/ice logic replicates backend/climate_sim/physics/snow_albedo.py
 * so no albedo field is needed.
 */

// ── Color palette ──────────────────────────────────────────────────

// Ocean depth gradient (bathymetry-driven)
const OCEAN_ABYSS: [number, number, number] = [0.02, 0.06, 0.20];   // deep trenches
const OCEAN_DEEP: [number, number, number] = [0.04, 0.10, 0.28];    // deep ocean
const OCEAN_MID: [number, number, number] = [0.06, 0.16, 0.40];     // mid-ocean
const OCEAN_SHELF: [number, number, number] = [0.10, 0.28, 0.52];   // continental shelf
const OCEAN_COASTAL: [number, number, number] = [0.14, 0.38, 0.58]; // near-shore shallow

// Bare soil hues by annual mean temperature
const SOIL_TUNDRA: [number, number, number] = [0.58, 0.55, 0.50];     // pale grey-tan
const SOIL_PODZOL: [number, number, number] = [0.45, 0.40, 0.34];     // grey-brown boreal
const SOIL_TEMPERATE: [number, number, number] = [0.42, 0.36, 0.26];  // neutral brown
const SOIL_WARM: [number, number, number] = [0.48, 0.38, 0.26];       // warm brown (subtle warmth)
const SOIL_LATERITE: [number, number, number] = [0.55, 0.38, 0.24];   // muted red-brown tropical

// Ground cover (grass/moss) hues
const GRASS_DEAD: [number, number, number] = [0.62, 0.55, 0.30];      // yellow-brown dead grass
const GRASS_BOREAL: [number, number, number] = [0.22, 0.38, 0.18];    // boreal moss/lichen
const GRASS_TEMPERATE: [number, number, number] = [0.24, 0.50, 0.18]; // rich green grass
const GRASS_TROPICAL: [number, number, number] = [0.16, 0.52, 0.12];  // bright tropical green

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
 * @param isLand             - true if land cell
 * @param surfaceTempC       - surface temperature in Celsius (monthly)
 * @param soilMoisture       - soil moisture 0-1 fraction of field capacity (monthly)
 * @param elevationM         - elevation in meters (negative for ocean bathymetry)
 * @param vegetationFraction - vegetation cover 0-1 (monthly, controls bare soil vs ground cover)
 * @param annualMeanTempC    - annual mean surface temp in Celsius (for soil mineralogy hue)
 * @returns [r, g, b] normalized 0-1
 */
export function blueMarbleColor(
  isLand: boolean,
  surfaceTempC: number,
  soilMoisture: number,
  elevationM: number = 0,
  vegetationFraction: number = 0,
  annualMeanTempC: number = 15,
): [number, number, number] {
  if (!isLand) {
    // ── Ocean ── bathymetry-driven depth shading
    const depth = Math.max(0, -elevationM); // meters below sea level

    // Multi-stop depth gradient
    let color: [number, number, number];
    if (depth < 200) {
      // Continental shelf (0-200m)
      const t = depth / 200;
      color = lerp3(OCEAN_COASTAL, OCEAN_SHELF, t);
    } else if (depth < 2000) {
      // Continental slope (200-2000m)
      const t = (depth - 200) / 1800;
      color = lerp3(OCEAN_SHELF, OCEAN_MID, t);
    } else if (depth < 5000) {
      // Deep ocean (2000-5000m)
      const t = (depth - 2000) / 3000;
      color = lerp3(OCEAN_MID, OCEAN_DEEP, t);
    } else {
      // Abyssal/trench (5000m+)
      const t = clamp01((depth - 5000) / 4000);
      color = lerp3(OCEAN_DEEP, OCEAN_ABYSS, t);
    }

    // Sea ice overlay
    const iceFrac = seaIceFraction(surfaceTempC);
    if (iceFrac > 0.001) {
      color = lerp3(color, SEA_ICE, iceFrac);
    }

    return color;
  }

  // ── Land ──
  // 1. Bare soil color from annual mean temperature + moisture darkening
  let bareSoil: [number, number, number];
  if (annualMeanTempC < -5) {
    bareSoil = SOIL_TUNDRA;
  } else if (annualMeanTempC < 5) {
    const t = clamp01((annualMeanTempC + 5) / 10);
    bareSoil = lerp3(SOIL_TUNDRA, SOIL_PODZOL, t);
  } else if (annualMeanTempC < 15) {
    const t = clamp01((annualMeanTempC - 5) / 10);
    bareSoil = lerp3(SOIL_PODZOL, SOIL_TEMPERATE, t);
  } else if (annualMeanTempC < 28) {
    const t = clamp01((annualMeanTempC - 15) / 13);
    bareSoil = lerp3(SOIL_TEMPERATE, SOIL_WARM, t);
  } else {
    const t = clamp01((annualMeanTempC - 28) / 7);
    bareSoil = lerp3(SOIL_WARM, SOIL_LATERITE, t);
  }
  // Moisture darkens soil (wet organic soil is darker)
  const darken = 1.0 - 0.25 * soilMoisture;
  bareSoil = [bareSoil[0] * darken, bareSoil[1] * darken, bareSoil[2] * darken];

  // 2. Ground cover color from monthly temperature + moisture
  let groundCover: [number, number, number];
  const grassGreen = surfaceTempC < 0
    ? lerp3(GRASS_BOREAL, GRASS_TEMPERATE, clamp01((surfaceTempC + 10) / 10))
    : lerp3(GRASS_TEMPERATE, GRASS_TROPICAL, clamp01(surfaceTempC / 25));
  const grassAlive = clamp01(soilMoisture / 0.3);
  groundCover = lerp3(GRASS_DEAD, grassGreen, grassAlive);

  // 3. Blend bare soil ↔ ground cover by vegetation fraction
  let baseColor = lerp3(bareSoil, groundCover, clamp01(vegetationFraction));

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
