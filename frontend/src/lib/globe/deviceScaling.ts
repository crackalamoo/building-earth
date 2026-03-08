import { getGPUTier } from 'detect-gpu';

export const rendering = {
  particleCount: 15000,
  trailLength: 30,
  pixelRatioCap: 2,
};

getGPUTier().then((result) => {
  if (result.tier <= 1) {
    rendering.particleCount = 4000;
    rendering.pixelRatioCap = 1;
  }
});
