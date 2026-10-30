import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Plotly is one ~4.6 MB chunk, loaded lazily the first time a chart renders.
  build: { chunkSizeWarningLimit: 5000 },
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
