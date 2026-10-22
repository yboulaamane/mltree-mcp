declare module "plotly.js-basic-dist-min" {
  const Plotly: {
    react: (el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
    purge: (el: HTMLElement) => void;
    Plots: { resize: (el: HTMLElement) => void };
  };
  export default Plotly;
}
declare module "monaco-editor/esm/vs/basic-languages/python/python.contribution";
