import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  build: {
    // Output into the FastAPI daemon's static dir so the production daemon serves it
    outDir: path.resolve(__dirname, "../runtime/static"),
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // All REST + WS endpoints live under /api/ on the daemon (so they
      // can't collide with React Router paths). One proxy entry covers
      // both HTTP and the WebSocket upgrade.
      "/api": {
        target: "http://127.0.0.1:8000",
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
