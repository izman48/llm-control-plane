/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // 5273 avoids the very common 5173 (taken by other Vite apps); strictPort:false
    // lets Vite fall back to the next free port if 5273 is also busy, so `localhost`
    // always reaches our dev server.
    port: 5273,
    strictPort: false,
    proxy: {
      // dev-proxy to the FastAPI gateway (make dev)
      "/api": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/setupTests.ts"],
    css: false,
  },
});
