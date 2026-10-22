import Editor, { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/esm/vs/editor/editor.api";
import "monaco-editor/esm/vs/basic-languages/python/python.contribution";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import type { Theme } from "./App";

// Bundle Monaco locally (no CDN), so the viewer works offline and on air-gapped clusters.
(self as unknown as { MonacoEnvironment: unknown }).MonacoEnvironment = { getWorker: () => new EditorWorker() };
loader.config({ monaco });

function language(path: string): string {
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".json")) return "json";
  return "plaintext";
}

export default function CodeView({ path, content, theme }: { path: string; content: string; theme: Theme }) {
  return (
    <Editor
      height="100%"
      path={path}
      language={language(path)}
      value={content}
      theme={theme === "dark" ? "vs-dark" : "vs"}
      options={{
        readOnly: true, domReadOnly: true, minimap: { enabled: false }, fontSize: 12.5, lineNumbers: "on",
        scrollBeyondLastLine: false, renderLineHighlight: "none", wordWrap: "on", automaticLayout: true,
        padding: { top: 8 }, fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
      }}
    />
  );
}
