import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build goes straight into the Python package so `pip install` needs no Node.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/mltree_mcp/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: {
        manualChunks: {
          flow: ["@xyflow/react", "@dagrejs/dagre"],
        },
      },
    },
  },
  server: {
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
