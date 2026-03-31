import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    allowedHosts: ['scoreai.cloud', 'www.scoreai.cloud'],
    watch: {
      ignored: ['**/venv/**', '**/node_modules/**', '**/.git/**']
    }
  }
})
