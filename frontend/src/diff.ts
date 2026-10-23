/**
 * Line diff (Myers O(ND)) for comparing two script snapshots in the browser, so code diffs also work in the
 * offline HTML export. No dependencies.
 */
export type Op = { t: " " | "-" | "+"; text: string; a: number | null; b: number | null }; // 1-based line numbers

const MAX_D = 2000; // beyond this many edits, fall back to "everything replaced"

export function diffLines(aText: string, bText: string): Op[] {
  const a = aText.split("\n"), b = bText.split("\n");
  if (a.length && a[a.length - 1] === "") a.pop();
  if (b.length && b[b.length - 1] === "") b.pop();
  // trim common prefix/suffix (fast path for small edits)
  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
  let suf = 0;
  while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
  const A = a.slice(pre, a.length - suf), B = b.slice(pre, b.length - suf);
  const mid = myers(A, B);
  const ops: Op[] = [];
  for (let i = 0; i < pre; i++) ops.push({ t: " ", text: a[i], a: i + 1, b: i + 1 });
  let ia = pre, ib = pre;
  for (const t of mid) {
    if (t === " ") ops.push({ t, text: a[ia], a: ++ia, b: ++ib });
    else if (t === "-") ops.push({ t, text: a[ia], a: ++ia, b: null });
    else ops.push({ t, text: b[ib], a: null, b: ++ib });
  }
  for (let i = 0; i < suf; i++) ops.push({ t: " ", text: a[ia], a: ++ia, b: ++ib });
  return ops;
}

function myers(a: string[], b: string[]): (" " | "-" | "+")[] {
  const n = a.length, m = b.length, max = n + m;
  if (n === 0) return b.map(() => "+");
  if (m === 0) return a.map(() => "-");
  const off = max + 1;
  const v = new Array<number>(2 * max + 3).fill(0);
  const trace: number[][] = []; // trace[d][k + d + 1] = v[k] before step d (memory O(D^2))
  let found = false;
  for (let d = 0; d <= Math.min(max, MAX_D) && !found; d++) {
    trace.push(v.slice(off - d - 1, off + d + 2));
    for (let k = -d; k <= d; k += 2) {
      let x = k === -d || (k !== d && v[off + k - 1] < v[off + k + 1]) ? v[off + k + 1] : v[off + k - 1] + 1;
      let y = x - k;
      while (x < n && y < m && a[x] === b[y]) { x++; y++; }
      v[off + k] = x;
      if (x >= n && y >= m) { found = true; break; }
    }
  }
  if (!found) return [...a.map(() => "-" as const), ...b.map(() => "+" as const)];
  const out: (" " | "-" | "+")[] = [];
  let x = n, y = m;
  for (let d = trace.length - 1; d >= 0; d--) {
    const t = trace[d], at = (k: number) => t[k + d + 1];
    const k = x - y;
    const prevK = k === -d || (k !== d && at(k - 1) < at(k + 1)) ? k + 1 : k - 1;
    const px = d === 0 ? 0 : at(prevK), py = px - prevK;
    while (x > px && y > py) { out.push(" "); x--; y--; }
    if (d > 0) out.push(x === px ? "+" : "-");
    x = px; y = py;
  }
  return out.reverse();
}

export interface Hunk { header: string; ops: Op[] }

/** Group ops into unified-diff hunks with `context` unchanged lines around each change. */
export function hunks(ops: Op[], context = 3): Hunk[] {
  const out: Hunk[] = [];
  const changed = ops.map((o) => o.t !== " ");
  let i = 0;
  while (i < ops.length) {
    if (!changed[i]) { i++; continue; }
    const start = Math.max(0, i - context);
    let end = i;
    // extend while the next change is within 2*context
    for (;;) {
      while (end < ops.length && changed[end]) end++;
      let j = end;
      while (j < ops.length && !changed[j] && j - end < 2 * context) j++;
      if (j < ops.length && changed[j]) { end = j; continue; }
      break;
    }
    const stop = Math.min(ops.length, end + context);
    const slice = ops.slice(start, stop);
    const aStart = slice.find((o) => o.a !== null)?.a ?? 0, bStart = slice.find((o) => o.b !== null)?.b ?? 0;
    const aLen = slice.filter((o) => o.t !== "+").length, bLen = slice.filter((o) => o.t !== "-").length;
    out.push({ header: `@@ -${aStart},${aLen} +${bStart},${bLen} @@`, ops: slice });
    i = stop;
  }
  return out;
}

export type Row = { left: Op | null; right: Op | null };

/** Side-by-side rows: unchanged lines on both sides, removed/added blocks paired line by line. */
export function sideBySide(ops: Op[]): Row[] {
  const rows: Row[] = [];
  let i = 0;
  while (i < ops.length) {
    if (ops[i].t === " ") { rows.push({ left: ops[i], right: ops[i] }); i++; continue; }
    const del: Op[] = [], add: Op[] = [];
    while (i < ops.length && ops[i].t === "-") del.push(ops[i++]);
    while (i < ops.length && ops[i].t === "+") add.push(ops[i++]);
    for (let k = 0; k < Math.max(del.length, add.length); k++) rows.push({ left: del[k] ?? null, right: add[k] ?? null });
  }
  return rows;
}

export function stats(ops: Op[]): { added: number; removed: number } {
  let added = 0, removed = 0;
  for (const o of ops) { if (o.t === "+") added++; else if (o.t === "-") removed++; }
  return { added, removed };
}
