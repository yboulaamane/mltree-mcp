"""Run pipeline steps as subprocesses and record them as runs.

This is the ONLY place mltree executes user code, and only when asked (the viewer needs
``mltree-mcp view --allow-run``; the MCP tool run_step can be disabled with MLTREE_ALLOW_RUN=0).
The scanner never imports this module.

A run is recorded automatically: params, the script snapshot (via Store.log_run), the command,
exit status, duration, the log (.mltree/runs/<id>/log.txt) and metrics parsed from
``MLTREE_METRIC key=value`` stdout lines and/or a metrics JSON file the script writes
(``$MLTREE_METRICS_FILE`` or a declared ``*.json`` output modified during the run).
"""

from __future__ import annotations

import ast
import itertools
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import MLTreeError, Store

METRIC_LINE_RE = re.compile(r"^\s*MLTREE_METRIC\s+(.+?)\s*$")
PAIR_RE = re.compile(r"([A-Za-z_][\w./-]*)\s*=\s*([-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|nan|inf))(?=\s|,|$)")
MAX_LOG_LINES = 50_000
MAX_SWEEP_RUNS = 200
SKIP_DIRS = {".git", ".mltree", ".venv", "venv", "env", "node_modules", "__pycache__", ".ipynb_checkpoints", ".tox"}
RUNNABLE_SUFFIXES = {".py"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def run_allowed_by_env() -> bool:
    return os.environ.get("MLTREE_ALLOW_RUN", "1").strip().lower() not in ("0", "false", "no", "off")


def parse_metric_line(line: str) -> dict[str, float] | None:
    """``MLTREE_METRIC loss=0.31 acc=0.9 step=3`` or ``MLTREE_METRIC {"loss": 0.31}`` -> dict, else None."""
    m = METRIC_LINE_RE.match(line)
    if not m:
        return None
    body = m.group(1)
    out: dict[str, float] = {}
    if body.startswith("{"):
        try:
            obj = json.loads(body)
        except ValueError:
            return None
        out = {k: float(v) for k, v in flatten_metrics(obj).items()}
    else:
        for k, v in PAIR_RE.findall(body):
            out[k] = float(v)
    out = {k: v for k, v in out.items() if math.isfinite(v)}
    return out or None


def flatten_metrics(obj: Any, prefix: str = "", depth: int = 0) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(obj, dict) or depth > 1:
        return out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            out[key] = float(v)
        elif isinstance(v, dict):
            out.update(flatten_metrics(v, key + ".", depth + 1))
    return out


# ------------------------------------------------------------ CLI detection


@dataclass
class CliSpec:
    style: str = "env"  # flags | argv | hydra | env
    options: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {flag, store_true, positional}
    positionals: list[str] = field(default_factory=list)
    argv: dict[int, str] = field(default_factory=dict)  # sys.argv index -> variable


def detect_cli(source: str) -> CliSpec:
    """Statically work out how a script takes parameters (argparse/click flags, sys.argv, hydra)."""
    spec = CliSpec()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spec
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            last = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if last == "main" and isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "hydra":
                spec.style = "hydra"
            if last not in ("add_argument", "option", "argument"):
                continue
            flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if not flags:
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            longs = [f for f in flags if f.startswith("--")]
            shorts = [f for f in flags if f.startswith("-") and not f.startswith("--")]
            positional = not longs and not shorts
            if isinstance(kw.get("dest"), ast.Constant):
                name = str(kw["dest"].value)  # type: ignore[union-attr]
            else:
                name = (longs[0] if longs else flags[0]).lstrip("-").replace("-", "_")
            if last == "argument":  # click.argument("name")
                positional = True
            action = kw.get("action")
            store_false = isinstance(action, ast.Constant) and action.value == "store_false"
            store_true = (isinstance(action, ast.Constant) and action.value == "store_true") or (
                isinstance(kw.get("is_flag"), ast.Constant) and bool(kw["is_flag"].value)  # type: ignore[union-attr]
            )
            if positional:
                spec.positionals.append(name)
                spec.options[name] = {"flag": None, "positional": True, "store_true": False}
            else:
                spec.options[name] = {"flag": longs[0] if longs else shorts[0], "positional": False,
                                      "store_true": store_true, "store_false": store_false}
            if spec.style != "hydra":
                spec.style = "flags"
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            for sub in ast.walk(node.value):
                if (isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Attribute) and sub.value.attr == "argv"
                        and isinstance(sub.value.value, ast.Name) and sub.value.value.id == "sys"):
                    idx = sub.slice
                    if isinstance(idx, ast.Constant) and isinstance(idx.value, int) and idx.value >= 1:
                        spec.argv.setdefault(idx.value, node.targets[0].id)
    if spec.style == "env" and spec.argv:
        spec.style = "argv"
    return spec


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return str(v)


def build_args(spec: CliSpec, params: dict[str, Any]) -> tuple[list[str], list[str]]:
    """CLI arguments for params, plus the param names that could only be passed through $MLTREE_PARAMS."""
    args: list[str] = []
    env_only: list[str] = []
    if spec.style == "hydra":
        return [f"{k}={_fmt(v)}" for k, v in params.items()], []
    if spec.style == "flags":
        used = set()
        for name in spec.positionals:
            if name in params:
                args.append(_fmt(params[name]))
                used.add(name)
        for name, v in params.items():
            opt = spec.options.get(name)
            if name in used:
                continue
            if not opt or opt["positional"]:
                env_only.append(name)
                continue
            if opt["store_true"] or opt.get("store_false"):
                on = v is True or str(v).lower() in ("1", "true", "yes")
                if on == bool(opt["store_true"]):  # store_true: pass when on; store_false: pass when off
                    args.append(opt["flag"])
                continue
            if isinstance(v, list):
                args += [opt["flag"], *[_fmt(x) for x in v]]
            else:
                args += [opt["flag"], _fmt(v)]
        return args, env_only
    if spec.style == "argv":
        by_name = {var: i for i, var in spec.argv.items()}
        for i in range(1, max(spec.argv) + 1):
            var = spec.argv.get(i)
            if var is None or var not in params:
                break  # positional args must be contiguous
            args.append(_fmt(params[var]))
        env_only = [k for k in params if k not in by_name or by_name[k] > len(args)]
        return args, env_only
    return [], list(params)


def find_python(root: Path) -> str:
    """The interpreter to run steps with: $MLTREE_PYTHON, the project's venv, the active env, then PATH."""
    env = os.environ.get("MLTREE_PYTHON")
    if env:
        return env
    for cand in (root / ".venv" / "bin" / "python", root / "venv" / "bin" / "python",
                 root / ".venv" / "Scripts" / "python.exe", root / "venv" / "Scripts" / "python.exe"):
        if cand.exists():
            return str(cand)
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(var):
            cand = Path(os.environ[var]) / "bin" / "python"
            if cand.exists():
                return str(cand)
    return shutil.which("python3") or shutil.which("python") or sys.executable


# ------------------------------------------------------------------ runs


@dataclass
class LiveRun:
    run_id: str
    step: str
    argv: list[str]
    started: float
    proc: subprocess.Popen | None = None
    lines: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, float]] = field(default_factory=list)
    status: str = "running"
    exit_code: int | None = None
    cancelled: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    dropped: int = 0

    def summary(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "step": self.step, "status": self.status, "argv": self.argv,
                "exit_code": self.exit_code, "lines": len(self.lines) + self.dropped,
                "elapsed_s": round(time.monotonic() - self.started, 1), "metrics": self.metrics}


@dataclass
class LiveSweep:
    sweep_id: str
    step: str
    total: int
    run_ids: list[str] = field(default_factory=list)
    cancelled: bool = False
    current: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class RunManager:
    """Launches step scripts, streams their output and records them in the store. One per process."""

    def __init__(self, store: Store, python: str | None = None) -> None:
        self.store = store
        self.python = python
        self._live: dict[str, LiveRun] = {}
        self._sweeps: dict[str, LiveSweep] = {}
        self._lock = threading.Lock()

    # -- planning
    def defaults(self, step: str) -> dict[str, Any]:
        """Params to prefill the run form with: the last run's params, else the step's (scanned) params."""
        info = self.store.get_step(step)
        last = None
        for r in self.store.list_runs(step=info["name"], limit=50):
            if r["kind"] != "sweep" and r["params"]:
                last = r
                break
        if last:
            return {"params": {**info["params"], **last["params"]}, "source": "last run", "source_run_id": last["id"]}
        return {"params": dict(info["params"]), "source": "step params", "source_run_id": None}

    def default_parent(self, step: str) -> str | None:
        """A run started from the map branches from the newest succeeded run of a direct upstream step,
        so `evaluate` hangs under the `train` run that produced its model in the experiment tree."""
        with self.store._conn() as c:  # noqa: SLF001
            row = c.execute(
                "SELECT r.id FROM runs r JOIN edges e ON e.source = r.step WHERE e.target = ? AND r.status = 'succeeded' "
                "AND r.kind != 'sweep' ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1", (step,)).fetchone()
        return row["id"] if row else None

    def plan(self, step: str, params: dict[str, Any] | None = None, extra_args: str | list[str] | None = None) -> dict[str, Any]:
        info = self.store.get_step(step)
        base = self.defaults(info["name"])
        if params is None:
            params = base["params"]
        if not isinstance(params, dict):
            raise MLTreeError("params must be an object")
        result: dict[str, Any] = {"step": info["name"], "params": params, "defaults": base, "runnable": False,
                                  "reason": None, "argv": None, "cwd": str(self.store.root), "warnings": [],
                                  "parent_run_id": self.default_parent(info["name"])}
        sp = info.get("script_path")
        if not sp:
            result["reason"] = "This step has no script_path; set one with add_step."
            return result
        script = (self.store.root / sp)
        if not script.is_file():
            result["reason"] = f"{sp} does not exist."
            return result
        if script.suffix not in RUNNABLE_SUFFIXES:
            result["reason"] = f"Only Python scripts can be run from mltree ({script.suffix} is not supported yet)."
            return result
        try:
            script.resolve().relative_to(self.store.root.resolve())
        except ValueError:
            result["reason"] = f"{sp} is outside the project."
            return result
        spec = detect_cli(script.read_text(encoding="utf-8", errors="replace"))
        args, env_only = build_args(spec, params)
        if isinstance(extra_args, str):
            extra = shlex.split(extra_args)
        else:
            extra = [str(a) for a in (extra_args or [])]
        python = self.python or find_python(self.store.root)
        argv = [python, "-u", sp, *args, *extra]
        if env_only:
            result["warnings"].append(
                "Not CLI arguments of the script, passed only as $MLTREE_PARAMS / $MLTREE_PARAM_<NAME>: " + ", ".join(env_only))
        result.update({"runnable": True, "argv": argv, "command": shlex.join(argv), "style": spec.style,
                       "python": python, "env_only": env_only, "cli_options": sorted(spec.options) or sorted(spec.argv.values())})
        return result

    # -- launching
    def start(self, step: str, params: dict[str, Any] | None = None, extra_args: str | list[str] | None = None,
              sweep_id: str | None = None, parent_run_id: str | None = None, notes: str = "",
              tags: list[str] | None = None) -> LiveRun:
        plan = self.plan(step, params, extra_args)
        if not plan["runnable"]:
            raise MLTreeError(plan["reason"])
        params = plan["params"]
        if parent_run_id is None and sweep_id is None:
            parent_run_id = plan["parent_run_id"]
        started_at = _now()
        rec = self.store.log_run(plan["step"], params=params, status="running", started_at=started_at,
                                 sweep_id=sweep_id, parent_run_id=parent_run_id, notes=notes,
                                 tags=["mltree-run", *(tags or [])], command=plan["argv"],
                                 record_data="inputs")
        run_id = rec["id"]
        run_dir = self.store.runs_dir / run_id
        env = dict(os.environ)
        env.update({
            "MLTREE_RUN_ID": run_id,
            "MLTREE_STEP": plan["step"],
            "MLTREE_PROJECT": str(self.store.root),
            "MLTREE_PARAMS": json.dumps(params),
            "MLTREE_METRICS_FILE": str(run_dir / "metrics.json"),
            "PYTHONUNBUFFERED": "1",
        })
        for k, v in params.items():
            key = re.sub(r"\W", "_", str(k)).upper()
            env[f"MLTREE_PARAM_{key}"] = _fmt(v)
        live = LiveRun(run_id=run_id, step=plan["step"], argv=plan["argv"], started=time.monotonic())
        live.wall_start = time.time()  # type: ignore[attr-defined]
        try:
            live.proc = subprocess.Popen(  # noqa: S603 - explicit, opt-in execution of the user's own script
                plan["argv"], cwd=self.store.root, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
        except OSError as exc:
            self.store.update_run(run_id, status="failed", notes=(notes + "\n" if notes else "") + f"could not start: {exc}")
            raise MLTreeError(f"Could not start {plan['command']}: {exc}") from exc
        with self._lock:
            self._live[run_id] = live
        log_path = run_dir / "log.txt"
        log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115 - closed by the waiter thread
        file_lock = threading.Lock()

        def pump(stream, name: str) -> None:
            for raw in stream:
                line = raw.rstrip("\n")
                entry = {"t": round(time.monotonic() - live.started, 3), "s": name, "line": line}
                m = parse_metric_line(line) if name == "out" else None
                with file_lock:
                    log_file.write(("" if name == "out" else "[stderr] ") + line + "\n")
                    log_file.flush()
                with self._lock:
                    if m:
                        point = dict(m)
                        point.setdefault("step", float(len(live.history)))
                        live.history.append(point)
                        live.metrics.update({k: v for k, v in m.items() if k != "step"})
                    live.lines.append(entry)
                    if len(live.lines) > MAX_LOG_LINES:
                        n = len(live.lines) - MAX_LOG_LINES
                        del live.lines[:n]
                        live.dropped += n
            stream.close()

        threads = [threading.Thread(target=pump, args=(live.proc.stdout, "out"), daemon=True),
                   threading.Thread(target=pump, args=(live.proc.stderr, "err"), daemon=True)]
        for t in threads:
            t.start()

        def waiter() -> None:
            code = live.proc.wait()
            for t in threads:
                t.join(timeout=5)
            log_file.close()
            self._finish(live, code, plan)

        threading.Thread(target=waiter, daemon=True).start()
        return live

    def _finish(self, live: LiveRun, code: int, plan: dict[str, Any]) -> None:
        run_dir = self.store.runs_dir / live.run_id
        metrics = dict(live.metrics)
        sources = []
        if live.history:
            (run_dir / "metrics_history.jsonl").write_text("".join(json.dumps(p) + "\n" for p in live.history), encoding="utf-8")
            sources.append("stdout")
        found = self._metrics_files(plan["step"], run_dir, getattr(live, "wall_start", 0.0))
        for path in found:
            try:
                metrics.update(flatten_metrics(json.loads(path.read_text(encoding="utf-8"))))
                sources.append(self._rel(path))
            except (OSError, ValueError):
                continue
        outputs = self._changed_outputs(plan["step"], getattr(live, "wall_start", 0.0))
        status = "cancelled" if live.cancelled else ("succeeded" if code == 0 else "failed")
        duration = round(time.monotonic() - live.started, 3)
        note_bits = [f"exit code {code}", f"{duration:.1f}s"]
        if sources:
            note_bits.append("metrics from " + ", ".join(sources))
        err_tail = [ln["line"] for ln in live.lines if ln.get("s") == "err"][-5:] if status == "failed" else []
        try:
            run = self.store.get_run(live.run_id)
            notes = (run["notes"] + "\n" if run["notes"] else "") + "mltree run: " + "; ".join(note_bits)
            if err_tail:
                notes += "\nstderr (last lines):\n" + "\n".join(err_tail)
            data = None
            if status == "succeeded":
                try:
                    data = self.store.record_outputs(live.run_id)
                except (OSError, MLTreeError):
                    data = None
            self.store.update_run(live.run_id, status=status, metrics=metrics or None, finished_at=_now(),
                                  exit_code=code, notes=notes, data=data,
                                  artifacts=[{"name": Path(p).name, "source": p, "path": p, "bytes": s, "stored": "reference"}
                                             for p, s in outputs] or None)
        finally:
            with self._lock:
                live.metrics = metrics
                live.exit_code = code
                live.status = status
            live.done.set()

    def _rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.store.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def _candidates(self, name: str, step: str) -> list[Path]:
        root = self.store.root
        info = self.store.get_step(step)
        script_dir = (root / info["script_path"]).parent if info.get("script_path") else root
        out = [root / name, script_dir / name]
        base = Path(name).name
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            if Path(dirpath).relative_to(root).parts.__len__() > 4:
                dirnames[:] = []
            if base in filenames:
                out.append(Path(dirpath) / base)
        seen, uniq = set(), []
        for p in out:
            rp = p.resolve()
            if rp not in seen and p.is_file():
                seen.add(rp)
                uniq.append(p)
        return uniq

    def _metrics_files(self, step: str, run_dir: Path, since: float) -> list[Path]:
        found = []
        own = run_dir / "metrics.json"
        if own.is_file():
            found.append(own)
        for name in self.store.get_step(step)["outputs"]:
            if not str(name).lower().endswith(".json") or "metric" not in Path(str(name)).name.lower():
                continue
            for p in self._candidates(str(name), step):
                if p.stat().st_mtime >= since - 1 and p.stat().st_size < 5_000_000:
                    found.append(p)
                    break
        return found

    def _changed_outputs(self, step: str, since: float) -> list[tuple[str, int]]:
        res = []
        for name in self.store.get_step(step)["outputs"]:
            for p in self._candidates(str(name), step):
                st = p.stat()
                if st.st_mtime >= since - 1:
                    res.append((self._rel(p), st.st_size))
                    break
        return res

    # -- control
    def get(self, run_id: str) -> LiveRun | None:
        with self._lock:
            return self._live.get(run_id)

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            lives = list(self._live.values())
        return [lv.summary() for lv in lives if not lv.done.is_set()]

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            sweep = self._sweeps.get(run_id)
        if sweep:
            sweep.cancelled = True
            if sweep.current:
                self.cancel(sweep.current)
            return {"sweep_id": run_id, "cancelled": True}
        live = self.get(run_id)
        if live is None or live.done.is_set():
            raise MLTreeError(f"Run {run_id!r} is not running in this process")
        live.cancelled = True
        self._signal(live, signal.SIGTERM)

        def killer() -> None:
            if not live.done.wait(5):
                self._signal(live, signal.SIGKILL)

        threading.Thread(target=killer, daemon=True).start()
        return {"run_id": run_id, "cancelled": True}

    @staticmethod
    def _signal(live: LiveRun, sig: int) -> None:
        if live.proc is None or live.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(live.proc.pid), sig)
        except (ProcessLookupError, PermissionError, AttributeError, OSError):
            try:
                live.proc.send_signal(sig)
            except OSError:
                pass

    def shutdown(self) -> None:
        for s in list(self._sweeps.values()):
            s.cancelled = True
        for lv in list(self._live.values()):
            if not lv.done.is_set():
                lv.cancelled = True
                self._signal(lv, signal.SIGTERM)

    def wait(self, run_id: str, timeout: float | None = None) -> bool:
        live = self.get(run_id)
        return True if live is None else live.done.wait(timeout)

    def log(self, run_id: str, offset: int = 0, tail: int | None = None) -> dict[str, Any]:
        live = self.get(run_id)
        if live is not None:
            with self._lock:
                total = len(live.lines) + live.dropped
                start = max(offset - live.dropped, 0)
                lines = live.lines[start:]
                done = live.done.is_set()
            if tail:
                lines = lines[-tail:]
            return {"run_id": run_id, "lines": lines, "next": total, "done": done, "status": live.status, "live": True}
        self.store.get_run(run_id)  # raises for unknown ids
        p = self.store.runs_dir / run_id / "log.txt"
        text = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.is_file() else []
        lines = [{"t": None, "s": "err" if ln.startswith("[stderr] ") else "out", "line": ln.removeprefix("[stderr] ")}
                 for ln in text]
        total = len(lines)
        lines = lines[offset:]
        if tail:
            lines = lines[-tail:]
        return {"run_id": run_id, "lines": lines, "next": total, "done": True,
                "status": self.store.get_run(run_id)["status"], "live": False}

    def history(self, run_id: str) -> list[dict[str, float]]:
        live = self.get(run_id)
        if live is not None:
            with self._lock:
                return list(live.history)
        return self.store.metric_history(run_id)

    # -- sweeps
    def start_sweep(self, step: str, grid: dict[str, list[Any]], base_params: dict[str, Any] | None = None,
                    metric: str | None = None, mode: str = "max", extra_args: str | list[str] | None = None,
                    name: str | None = None, max_runs: int = 64, parent_run_id: str | None = None) -> dict[str, Any]:
        combos = grid_combinations(grid, max_runs=max_runs)
        plan = self.plan(step, base_params, extra_args)
        if not plan["runnable"]:
            raise MLTreeError(plan["reason"])
        mode = (mode or "max").lower()
        if mode not in ("min", "max"):
            raise MLTreeError("mode must be 'min' or 'max'")
        base = dict(plan["params"])
        if parent_run_id is None:
            parent_run_id = plan["parent_run_id"]
        sweep = self.store.log_run(
            plan["step"], params={**base, **{k: list(v) for k, v in grid.items()}}, status="running", kind="sweep",
            started_at=_now(), parent_run_id=parent_run_id, tags=["sweep", "grid", "mltree-run"],
            notes=name or f"grid sweep over {', '.join(grid)} ({len(combos)} runs)",
            sweep={"source": "grid", "metric": metric, "mode": mode, "grid": grid, "total": len(combos), "name": name},
        )
        live = LiveSweep(sweep_id=sweep["id"], step=plan["step"], total=len(combos))
        with self._lock:
            self._sweeps[sweep["id"]] = live

        def loop() -> None:
            failed = 0
            try:
                for combo in combos:
                    if live.cancelled:
                        break
                    try:
                        child = self.start(plan["step"], {**base, **combo}, extra_args, sweep_id=live.sweep_id,
                                           tags=["sweep"])
                    except MLTreeError:
                        failed += 1
                        continue
                    live.current = child.run_id
                    live.run_ids.append(child.run_id)
                    child.done.wait()
                    if child.status == "failed":
                        failed += 1
                status = "cancelled" if live.cancelled else ("failed" if failed == len(combos) else "succeeded")
                summ = self.store.sweep_summary(live.sweep_id)
                metrics = {f"best_{metric}": summ["best_value"]} if metric and summ["best_value"] is not None else None
                self.store.update_run(live.sweep_id, status=status, metrics=metrics, finished_at=_now(),
                                      sweep={"failed": failed, "completed": len(live.run_ids),
                                             "best_run_id": summ["best_run_id"]})
            finally:
                live.current = None
                live.done.set()

        threading.Thread(target=loop, daemon=True).start()
        return {"sweep_id": sweep["id"], "step": plan["step"], "total": len(combos), "combinations": combos,
                "metric": metric, "mode": mode}

    def wait_sweep(self, sweep_id: str, timeout: float | None = None) -> bool:
        with self._lock:
            live = self._sweeps.get(sweep_id)
        return True if live is None else live.done.wait(timeout)

    def sweep_progress(self, sweep_id: str) -> dict[str, Any] | None:
        with self._lock:
            live = self._sweeps.get(sweep_id)
        if live is None:
            return None
        return {"sweep_id": sweep_id, "step": live.step, "total": live.total, "launched": len(live.run_ids), "current": live.current,
                "done": live.done.is_set(), "cancelled": live.cancelled}


def grid_combinations(grid: dict[str, list[Any]], max_runs: int = 64) -> list[dict[str, Any]]:
    if not isinstance(grid, dict) or not grid:
        raise MLTreeError('grid must be an object of param -> list of values, e.g. {"alpha": [0.1, 1, 10]}')
    keys, values = [], []
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            raise MLTreeError(f"grid[{k!r}] must be a non-empty list of values")
        keys.append(str(k))
        values.append(v)
    total = 1
    for v in values:
        total *= len(v)
    cap = min(max(int(max_runs), 1), MAX_SWEEP_RUNS)
    if total > cap:
        raise MLTreeError(f"The grid has {total} combinations; the limit is {cap} (raise max_runs, up to {MAX_SWEEP_RUNS}).")
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def parse_grid_text(text: str) -> list[Any]:
    """'0.1, 0.5, 1' -> [0.1, 0.5, 1]; JSON lists are accepted too."""
    text = (text or "").strip()
    if text.startswith("["):
        v = json.loads(text)
        return v if isinstance(v, list) else [v]
    out: list[Any] = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        try:
            out.append(json.loads(part))
        except ValueError:
            out.append(part)
    return out
