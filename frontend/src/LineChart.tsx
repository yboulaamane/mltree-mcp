/** Small dependency-free SVG line chart for step-wise metric curves (live viewer and offline export). */
import { useEffect, useMemo, useRef, useState } from "react";
import { fmtNum } from "./util";

export const SERIES_COLORS = ["#10b981", "#6366f1", "#f59e0b", "#ef4444", "#0ea5e9", "#a855f7", "#84cc16", "#ec4899"];

export interface Series { name: string; points: { x: number; y: number }[] }

function ticks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

export default function LineChart(props: { series: Series[]; xLabel: string; yLabel: string; height?: number; logY?: boolean }) {
  // draw at the container's real pixel width so labels keep their size on wide screens
  const box = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(760);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = props.height ?? 280, P = { l: 56, r: 16, t: 14, b: 40 };
  const [hover, setHover] = useState<number | null>(null);
  const tf = (y: number) => (props.logY ? Math.log10(Math.max(y, 1e-12)) : y);
  const { xs, ys } = useMemo(() => {
    const pts = props.series.flatMap((s) => s.points);
    return { xs: pts.map((p) => p.x), ys: pts.map((p) => tf(p.y)) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.series, props.logY]);
  if (!xs.length) return <div ref={box}><p className="muted pad">No step-wise values for this metric.</p></div>;
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { y0 -= Math.abs(y0) * 0.1 || 1; y1 += Math.abs(y1) * 0.1 || 1; }
  const pad = (y1 - y0) * 0.05; y0 -= pad; y1 += pad;
  const sx = (x: number) => P.l + ((x - x0) / (x1 - x0)) * (W - P.l - P.r);
  const sy = (y: number) => P.t + (1 - (tf(y) - y0) / (y1 - y0)) * (H - P.t - P.b);
  const sya = (v: number) => P.t + (1 - (v - y0) / (y1 - y0)) * (H - P.t - P.b);
  const hx = hover === null ? null : hover;
  return (
    <div className="linechart" ref={box}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${props.yLabel} by ${props.xLabel}`}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const x = x0 + (((e.clientX - r.left) / r.width) * W - P.l) / (W - P.l - P.r) * (x1 - x0);
          const all = [...new Set(xs)].sort((a, b) => a - b);
          setHover(all.reduce((b, v) => (Math.abs(v - x) < Math.abs(b - x) ? v : b), all[0]));
        }}
        onMouseLeave={() => setHover(null)}>
        {ticks(y0, y1).map((v) => (
          <g key={`y${v}`} className="grid">
            <line x1={P.l} x2={W - P.r} y1={sya(v)} y2={sya(v)} />
            <text x={P.l - 6} y={sya(v) + 4} textAnchor="end">{props.logY ? fmtNum(10 ** v) : fmtNum(v)}</text>
          </g>
        ))}
        {ticks(x0, x1, 8).map((v) => (
          <g key={`x${v}`} className="grid">
            <text x={sx(v)} y={H - P.b + 16} textAnchor="middle">{fmtNum(v)}</text>
          </g>
        ))}
        <text className="axis-label" x={(P.l + W - P.r) / 2} y={H - 6} textAnchor="middle">{props.xLabel}</text>
        <text className="axis-label" x={12} y={(P.t + H - P.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(P.t + H - P.b) / 2})`}>{props.yLabel}</text>
        {props.series.map((s, i) => {
          const pts = [...s.points].sort((a, b) => a.x - b.x);
          const c = SERIES_COLORS[i % SERIES_COLORS.length];
          return (
            <g key={s.name}>
              <polyline fill="none" stroke={c} strokeWidth={2} points={pts.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
              {pts.length <= 40 && pts.map((p) => <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={2.5} fill={c} />)}
            </g>
          );
        })}
        {hx !== null && <line className="hover" x1={sx(hx)} x2={sx(hx)} y1={P.t} y2={H - P.b} />}
      </svg>
      <div className="legend-row">
        {props.series.map((s, i) => {
          const at = hx === null ? s.points[s.points.length - 1] : s.points.find((p) => p.x === hx);
          return (
            <span key={s.name}>
              <i style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }} /> {s.name}
              {at && <b className="mono"> {fmtNum(at.y)}</b>}
            </span>
          );
        })}
        {hx !== null && <span className="muted small">{props.xLabel} = {fmtNum(hx)}</span>}
      </div>
    </div>
  );
}
