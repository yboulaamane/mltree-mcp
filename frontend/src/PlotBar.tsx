import { useEffect, useRef } from "react";
import Plotly from "plotly.js-basic-dist-min";
import type { Theme } from "./App";

export interface BarTrace {
  name: string;
  x: string[];
  y: (number | null)[];
}

export default function PlotBar({ traces, theme, title }: { traces: BarTrace[]; theme: Theme; title?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--fg").trim() || "#111";
    const grid = css.getPropertyValue("--border").trim() || "#ddd";
    Plotly.react(
      el,
      traces.map((t) => ({ type: "bar", name: t.name, x: t.x, y: t.y, hovertemplate: "%{x}: %{y}<extra>%{fullData.name}</extra>" })),
      {
        barmode: "group", title: title ? { text: title, font: { size: 13 } } : undefined,
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: fg, family: "Inter, system-ui, sans-serif", size: 12 },
        margin: { l: 48, r: 16, t: title ? 36 : 12, b: 40 },
        xaxis: { gridcolor: grid }, yaxis: { gridcolor: grid, zerolinecolor: grid },
        legend: { orientation: "h", y: -0.18 },
        colorway: ["#10b981", "#6366f1", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#84cc16", "#ec4899"],
      },
      { displayModeBar: false, responsive: true },
    );
  }, [traces, theme, title]);
  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el) Plotly.purge(el);
    };
  }, []);
  return <div ref={ref} className="plot" />;
}
