/** Lightweight read-only code view used by exported HTML files (Monaco is ~2 MB, this is ~2 KB). */
import { useMemo } from "react";
import type { Theme } from "./App";

const PY_KW = new Set(("False None True and as assert async await break class continue def del elif else except finally for from " +
  "global if import in is lambda nonlocal not or pass raise return try while with yield match case self").split(" "));
const PY_BUILTIN = new Set("print len range open str int float dict list set tuple enumerate zip isinstance super sorted min max sum abs".split(" "));
const TOKEN = /(#[^\n]*)|([rbfuRBFU]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(@[\w.]+)|(\b\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function highlight(src: string, lang: string): string {
  if (lang !== "python") return esc(src);
  let out = "";
  let last = 0;
  let prev = "";
  for (const m of src.matchAll(TOKEN)) {
    const i = m.index ?? 0;
    out += esc(src.slice(last, i));
    const [t, com, str, deco, num, word] = m;
    let cls = "";
    if (com) cls = "c";
    else if (str) cls = "s";
    else if (deco) cls = "d";
    else if (num) cls = "n";
    else if (word && PY_KW.has(word)) cls = "k";
    else if (word && (prev === "def" || prev === "class")) cls = "f";
    else if (word && PY_BUILTIN.has(word)) cls = "b";
    out += cls ? `<span class="tok-${cls}">${esc(t)}</span>` : esc(t);
    if (word) prev = word;
    last = i + t.length;
  }
  return out + esc(src.slice(last));
}

export default function CodeViewLite({ path, content }: { path: string; content: string; theme: Theme }) {
  const lang = path.endsWith(".py") || path.endsWith(".ipynb") ? "python" : "plain";
  const html = useMemo(() => highlight(content, lang), [content, lang]);
  const n = content.split("\n").length;
  return (
    <div className="code-lite">
      <pre className="gutter" aria-hidden>{Array.from({ length: n }, (_, i) => i + 1).join("\n")}</pre>
      <pre className="src"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>
    </div>
  );
}
