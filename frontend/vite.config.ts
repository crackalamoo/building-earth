import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [svelte()],
  server: {
    // Serve data files from the repo's data/ directory
    fs: {
      allow: ['..'],
    },
  },
  resolve: {
    alias: {
      '/data': path.resolve(__dirname, '../data'),
    },
  },
})
