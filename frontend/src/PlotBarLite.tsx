/** Dependency-free grouped bar chart (SVG) used by exported HTML files instead of Plotly. */
import type { Theme } from "./App";
import type { BarTrace } from "./PlotBar";
import { fmtNum } from "./util";

const COLORS = ["#10b981", "#6366f1", "#f59e0b", "#ef4444", "#06b6d4", "#a855f7", "#84cc16", "#ec4899"];

export default function PlotBarLite({ traces, title }: { traces: BarTrace[]; theme: Theme; title?: string }) {
  const cats = traces[0]?.x ?? [];
  const vals = traces.flatMap((t) => t.y.filter((v): v is number => typeof v === "number"));
  const hi = Math.max(0, ...vals), lo = Math.min(0, ...vals);
  const span = hi - lo || 1;
  const W = 720, H = 260, L = 48, R = 12, T = title ? 28 : 10, B = 34;
  const gw = (W - L - R) / Math.max(1, cats.length);
  const bw = Math.min(36, (gw * 0.8) / Math.max(1, traces.length));
  const y = (v: number) => T + ((hi - v) / span) * (H - T - B);
  const ticks = [lo, lo + span / 2, hi];
  return (
    <div className="plot plot-lite">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="xMidYMid meet" role="img" aria-label={title ?? "metrics chart"}>
        {title && <text x={W / 2} y={16} textAnchor="middle" className="t">{title}</text>}
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} className="grid" />
            <text x={L - 6} y={y(t) + 4} textAnchor="end" className="ax">{fmtNum(t)}</text>
          </g>
        ))}
        {cats.map((c, ci) => (
          <g key={c}>
            {traces.map((t, ti) => {
              const v = t.y[ci];
              if (typeof v !== "number") return null;
              const x = L + ci * gw + (gw - bw * traces.length) / 2 + ti * bw;
              return (
                <rect key={ti} x={x} width={bw - 2} y={Math.min(y(v), y(0))} height={Math.max(1, Math.abs(y(v) - y(0)))}
                  fill={COLORS[ti % COLORS.length]} rx={2}><title>{`${t.name} · ${c}: ${fmtNum(v)}`}</title></rect>
              );
            })}
            <text x={L + ci * gw + gw / 2} y={H - B + 16} textAnchor="middle" className="ax">{c}</text>
          </g>
        ))}
      </svg>
      <div className="legend-lite">
        {traces.map((t, i) => <span key={t.name}><i style={{ background: COLORS[i % COLORS.length] }} />{t.name}</span>)}
      </div>
    </div>
  );
}
