import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `vite build` -> the live viewer, straight into the Python package so `pip install` needs no Node.
// `vite build --mode export` -> one JS + one CSS file that `mltree-mcp export` inlines into a single
// offline HTML file. That build swaps Monaco and Plotly for small built-in components.
export default defineConfig(({ mode }) => {
  const exp = mode === "export";
  return {
    plugins: [react()],
    base: "/",
    resolve: {
      alias: exp
        ? [
            { find: /^\.\/CodeView$/, replacement: "/src/CodeViewLite.tsx" },
            { find: /^\.\/PlotBar$/, replacement: "/src/PlotBarLite.tsx" },
          ]
        : [],
    },
    build: exp
      ? {
          outDir: "../src/mltree_mcp/static/export",
          emptyOutDir: true,
          cssCodeSplit: false,
          chunkSizeWarningLimit: 4000,
          rollupOptions: {
            input: "src/main.tsx",
            output: { inlineDynamicImports: true, entryFileNames: "export.js", assetFileNames: "export[extname]" },
          },
        }
      : {
          outDir: "../src/mltree_mcp/static",
          emptyOutDir: true,
          chunkSizeWarningLimit: 4000,
          rollupOptions: { output: { manualChunks: { flow: ["@xyflow/react", "@dagrejs/dagre"] } } },
        },
    server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  };
});
