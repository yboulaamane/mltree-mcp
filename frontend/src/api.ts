import { useEffect, useState } from "react";
import { offlineApi, offlineData } from "./offline";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** Multi-project viewer: this page lives under /p/<slug>/ and its API under /p/<slug>/api/. */
export const BASE = typeof location !== "undefined" ? (location.pathname.match(/^\/p\/[^/]+/)?.[0] ?? "") : "";
export const apiUrl = (path: string) => (path.startsWith("/api") ? BASE + path : path);

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const offline = offlineData();
  if (offline) {
    try {
      return offlineApi(offline, path, init) as T;
    } catch (e) {
      throw new ApiError((e as { status?: number }).status ?? 400, (e as Error).message);
    }
  }
  const res = await fetch(apiUrl(path), {
    ...init,
    // X-MLTree marks requests from this app; the server requires it for anything that runs code
    headers: { "Content-Type": "application/json", "X-MLTree": "1", ...(init?.headers ?? {}) },
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, (body as { error?: string }).error ?? res.statusText);
  return body as T;
}

/** Live store revision: Server-Sent Events, with polling as a fallback. */
export function useRevision(): { revision: number; live: boolean } {
  const [revision, setRevision] = useState(0);
  const [live, setLive] = useState(false);
  useEffect(() => {
    if (offlineData()) return; // exported snapshot: nothing changes
    let timer: number | undefined;
    const poll = () => {
      timer = window.setInterval(async () => {
        try {
          const h = await api<{ revision: number }>("/api/health");
          setRevision(h.revision);
          setLive(true);
        } catch {
          setLive(false);
        }
      }, 3000);
    };
    if (!("EventSource" in window)) {
      poll();
      return () => window.clearInterval(timer);
    }
    const es = new EventSource(apiUrl("/api/events"));
    es.addEventListener("revision", (e) => {
      setRevision(JSON.parse((e as MessageEvent).data).revision);
      setLive(true);
    });
    es.onerror = () => setLive(false);
    es.onopen = () => setLive(true);
    return () => {
      es.close();
      window.clearInterval(timer);
    };
  }, []);
  return { revision, live };
}

/** Fetch `path` and refetch whenever one of `deps` changes. */
export function useApi<T>(path: string | null, deps: unknown[]): { data: T | null; error: string | null; loading: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!path) {
      setData(null);
      return;
    }
    let alive = true;
    setLoading(true);
    api<T>(path)
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  return { data, error, loading };
}
