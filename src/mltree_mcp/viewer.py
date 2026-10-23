"""HTTP viewer for mltree: JSON API + the prebuilt React frontend (no Node needed at runtime).

    mltree-mcp view [--project DIR] [--host 127.0.0.1] [--port 8765]

Routes (all JSON unless noted):

    GET  /api/health                         {"ok": true, "project": ..., "revision": n}
    GET  /api/pipeline                       nodes, edges, latest run status per node
    PUT  /api/positions                      {"positions": {step: {x, y} | null}}
    GET  /api/steps/{step}                   one node
    GET  /api/steps/{step}/script[?run_id=]  current script, or a run's snapshot
    GET  /api/steps/{step}/snapshots         runs of the step that carry a script snapshot
    GET  /api/runs[?step=&status=&parent_run_id=&limit=]
    GET  /api/runs/{run_id}
    GET  /api/tree[?metric=&mode=min|max&step=]   run forest from parent_run_id + best run
    GET  /api/compare?ids=a,b[,c...]
    GET  /api/metrics                        numeric metric names seen in runs
    GET  /api/export[?scripts=0|1]          the shareable single-file HTML export (download)
    GET  /api/events                         text/event-stream; "revision" events on every store change

  Running steps (POSTs need `mltree-mcp view --allow-run`, a loopback bind and the X-MLTree header):

    GET  /api/steps/{step}/plan[?params=json&extra_args=]  command preview + form defaults
    POST /api/steps/{step}/run               {"params": {...}, "extra_args": "..."} -> {"run_id"}
    POST /api/steps/{step}/sweep             {"grid": {p: [v, ...]}, "params", "metric", "mode", "max_runs"}
    POST /api/runs/{run_id}/cancel           cancel a run or a whole sweep
    GET  /api/active                         runs and sweeps running in this viewer
    GET  /api/runs/{run_id}/log[?offset=&tail=]   log lines (JSON)
    GET  /api/runs/{run_id}/log/stream[?offset=]  text/event-stream of log lines, then an "end" event
    GET  /api/runs/{run_id}/history          step-wise metrics from MLTREE_METRIC lines
    GET  /*                                  the single-page app (src/mltree_mcp/static)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .store import MLTreeError, ProjectNotInitialized, Store, resolve_project_root

class RunForbidden(MLTreeError):
    """Raised when a run request is not allowed (no --allow-run, wrong host/origin)."""


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _err(exc: Exception) -> JSONResponse:
    msg = str(exc)
    if isinstance(exc, ProjectNotInitialized):
        code = 409
    elif msg.startswith("Unknown") or "does not exist" in msg or "has no script" in msg:
        code = 404
    else:
        code = 400
    return JSONResponse({"error": msg}, status_code=code)


LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def create_app(project: str | Path | None = None, poll_interval: float = 1.0, allow_run: bool = False,
               python: str | None = None) -> Starlette:
    """Build the ASGI app. `project` defaults to $MLTREE_PROJECT / $CLAUDE_PROJECT_DIR / cwd.

    With allow_run=True the viewer can launch step scripts (the only way it executes code).
    """
    from .runner import RunManager

    root = resolve_project_root(project)
    manager = RunManager(Store(root), python=python)

    def store() -> Store:
        return Store(root)

    def handle(fn: Callable[[Request], Any]):
        async def endpoint(request: Request) -> Response:
            try:
                result = fn(request)
                if asyncio.iscoroutine(result):
                    result = await result
            except RunForbidden as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            except MLTreeError as exc:
                return _err(exc)
            return result if isinstance(result, Response) else JSONResponse(result)

        return endpoint

    def health(_: Request) -> dict[str, Any]:
        s = store()
        return {"ok": True, "project": s.project_name() if s.is_initialized() else root.name,
                "root": str(root), "initialized": s.is_initialized(), "revision": s.revision(),
                "allow_run": allow_run}

    def guard(request: Request) -> None:
        """Running code is opt-in and same-origin only (blocks drive-by POSTs and DNS rebinding)."""
        if not allow_run:
            raise RunForbidden("Running steps is disabled. Restart the viewer with `mltree-mcp view --allow-run`.")
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].lower()
        if host not in LOOPBACK and host != "testserver":
            raise RunForbidden(f"Runs are only accepted on localhost (Host: {host!r}); use an SSH tunnel.")
        if request.headers.get("x-mltree") != "1":
            raise RunForbidden("Missing X-MLTree header.")
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0].lower() not in LOOPBACK | {"testserver"}:
            raise RunForbidden(f"Cross-origin run request from {origin} refused.")

    async def body_of(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise MLTreeError("Body must be JSON") from exc
        if not isinstance(body, dict):
            raise MLTreeError("Body must be a JSON object")
        return body

    def plan(request: Request):
        q = request.query_params
        params = None
        if q.get("params"):
            try:
                params = json.loads(q["params"])
            except json.JSONDecodeError as exc:
                raise MLTreeError("params must be JSON") from exc
        res = manager.plan(request.path_params["step"], params, q.get("extra_args") or None)
        res["allow_run"] = allow_run
        return res

    async def run_step(request: Request):
        guard(request)
        body = await body_of(request)
        live = await asyncio.to_thread(manager.start, request.path_params["step"], body.get("params"),
                                       body.get("extra_args") or None, None, body.get("parent_run_id") or None,
                                       str(body.get("notes") or ""))
        return {"run_id": live.run_id, "argv": live.argv, "status": live.status}

    async def sweep(request: Request):
        guard(request)
        from .runner import parse_grid_text

        body = await body_of(request)
        grid = body.get("grid") or {}
        if isinstance(grid, dict):
            grid = {k: (parse_grid_text(v) if isinstance(v, str) else v) for k, v in grid.items()}
            grid = {k: v for k, v in grid.items() if v != []}
        return await asyncio.to_thread(
            manager.start_sweep, request.path_params["step"], grid, body.get("params"), body.get("metric") or None,
            body.get("mode") or "max", body.get("extra_args") or None, body.get("name") or None,
            int(body.get("max_runs") or 64))

    async def cancel(request: Request):
        guard(request)
        return manager.cancel(request.path_params["run_id"])

    def active(_: Request):
        return {"runs": manager.active(),
                "sweeps": [p for p in (manager.sweep_progress(sid) for sid in list(manager._sweeps)) if p and not p["done"]]}

    def _int(q, key: str, default: int | None) -> int | None:
        try:
            return int(q[key]) if q.get(key) else default
        except ValueError as exc:
            raise MLTreeError(f"{key} must be an integer") from exc

    def run_log(request: Request):
        q = request.query_params
        return manager.log(request.path_params["run_id"], offset=_int(q, "offset", 0) or 0, tail=_int(q, "tail", None))

    def history(request: Request):
        return {"run_id": request.path_params["run_id"], "history": manager.history(request.path_params["run_id"])}

    async def log_stream(request: Request) -> Response:
        run_id = request.path_params["run_id"]
        try:
            offset = _int(request.query_params, "offset", 0) or 0
            first = manager.log(run_id, offset=offset)
        except MLTreeError as exc:
            return _err(exc)

        async def stream():
            nonlocal offset
            chunk = first
            yield "retry: 3000\n\n"
            while True:
                if chunk["lines"]:
                    yield f"event: lines\ndata: {json.dumps({'lines': chunk['lines'], 'next': chunk['next']})}\n\n"
                offset = chunk["next"]
                if chunk["done"]:
                    run = await asyncio.to_thread(store().get_run, run_id)
                    yield f"event: end\ndata: {json.dumps(run)}\n\n"
                    break
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.25)
                chunk = await asyncio.to_thread(manager.log, run_id, offset)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def pipeline(_: Request):
        return store().get_pipeline()

    async def positions(request: Request):
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise MLTreeError("Body must be JSON: {\"positions\": {step: {x, y}}}") from exc
        if not isinstance(body, dict):
            raise MLTreeError("Body must be a JSON object")
        return store().set_positions(body.get("positions"))

    def step(request: Request):
        return store().get_step(request.path_params["step"])

    def script(request: Request):
        return store().script(request.path_params["step"], request.query_params.get("run_id") or None)

    def snapshots(request: Request):
        return {"snapshots": store().snapshots(request.path_params["step"])}

    def runs(request: Request):
        q = request.query_params
        try:
            limit = int(q.get("limit", 200))
        except ValueError as exc:
            raise MLTreeError("limit must be an integer") from exc
        rs = store().list_runs(step=q.get("step") or None, status=q.get("status") or None,
                               parent_run_id=q.get("parent_run_id") or None, limit=limit)
        return {"count": len(rs), "runs": rs}

    def run(request: Request):
        return store().get_run(request.path_params["run_id"])

    def tree(request: Request):
        q = request.query_params
        mode = q.get("mode", "max")
        if mode not in ("min", "max"):
            raise MLTreeError("mode must be 'min' or 'max'")
        return store().run_tree(metric=q.get("metric") or None, mode=mode, step=q.get("step") or None)

    def compare(request: Request):
        ids = [i for i in request.query_params.get("ids", "").split(",") if i.strip()]
        return store().compare(ids)

    def metrics(_: Request):
        return {"metrics": store().metric_names()}

    def export(request: Request):
        from .export import build_snapshot, default_filename, render_html

        s = store()
        scripts = request.query_params.get("scripts", "1").lower() not in ("0", "false", "no")
        body = render_html(build_snapshot(s, include_scripts=scripts))
        return Response(body, media_type="text/html; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="{default_filename(s)}"', "Cache-Control": "no-store"})

    async def events(request: Request) -> StreamingResponse:
        async def stream():
            last = None
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    rev = await asyncio.to_thread(store().revision)
                except Exception:  # noqa: BLE001 - keep the stream alive while the db is being written
                    rev = last
                if rev != last:
                    last = rev
                    yield f"event: revision\ndata: {json.dumps({'revision': rev})}\n\n"
                else:
                    yield ": ping\n\n"
                await asyncio.sleep(poll_interval)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def spa(request: Request) -> Response:
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            return Response(
                "mltree viewer frontend is not built. Run `npm install && npm run build` in frontend/.",
                status_code=503, media_type="text/plain",
            )
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    routes = [
        Route("/api/health", handle(health)),
        Route("/api/pipeline", handle(pipeline)),
        Route("/api/positions", handle(positions), methods=["PUT", "POST"]),
        Route("/api/steps/{step}", handle(step)),
        Route("/api/steps/{step}/script", handle(script)),
        Route("/api/steps/{step}/snapshots", handle(snapshots)),
        Route("/api/steps/{step}/plan", handle(plan)),
        Route("/api/steps/{step}/run", handle(run_step), methods=["POST"]),
        Route("/api/steps/{step}/sweep", handle(sweep), methods=["POST"]),
        Route("/api/active", handle(active)),
        Route("/api/runs", handle(runs)),
        Route("/api/runs/{run_id}", handle(run)),
        Route("/api/runs/{run_id}/cancel", handle(cancel), methods=["POST"]),
        Route("/api/runs/{run_id}/log", handle(run_log)),
        Route("/api/runs/{run_id}/log/stream", log_stream),
        Route("/api/runs/{run_id}/history", handle(history)),
        Route("/api/tree", handle(tree)),
        Route("/api/compare", handle(compare)),
        Route("/api/metrics", handle(metrics)),
        Route("/api/export", handle(export)),
        Route("/api/events", events),
        Route("/api/{rest:path}", lambda r: JSONResponse({"error": "Not found"}, status_code=404)),
    ]
    if (STATIC_DIR / "assets").is_dir():
        routes.append(Mount("/assets", app=StaticFiles(directory=STATIC_DIR / "assets"), name="assets"))
    routes += [Route("/", spa), Route("/{rest:path}", spa)]
    @contextlib.asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            manager.shutdown()  # don't leave child processes behind when the viewer stops

    app = Starlette(routes=routes, middleware=[Middleware(GZipMiddleware, minimum_size=1024)], lifespan=lifespan)
    app.state.root = root
    app.state.manager = manager
    app.state.allow_run = allow_run
    return app


def view_url(host: str | None = None, port: int | None = None) -> str:
    import os

    host = host or os.environ.get("MLTREE_VIEW_HOST") or DEFAULT_HOST
    port = port or int(os.environ.get("MLTREE_VIEW_PORT") or DEFAULT_PORT)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "::", "localhost") else host
    return f"http://{shown}:{port}"


def serve(project: str | Path | None = None, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          allow_run: bool = False, python: str | None = None) -> None:
    import uvicorn

    if allow_run and host not in LOOPBACK:
        raise MLTreeError(f"--allow-run only binds to 127.0.0.1 (got --host {host}). On a server, keep the default "
                          "host and open an SSH tunnel: ssh -L 8765:localhost:8765 user@server")
    app = create_app(project, allow_run=allow_run, python=python)
    print(f"mltree viewer for {app.state.root} -> {view_url(host, port)}  (Ctrl+C to stop)", flush=True)
    if allow_run:
        print("Running steps is ENABLED (--allow-run): the Run buttons execute the project's scripts on this machine.",
              flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
