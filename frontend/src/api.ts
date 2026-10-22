import { useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
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
    const es = new EventSource("/api/events");
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
