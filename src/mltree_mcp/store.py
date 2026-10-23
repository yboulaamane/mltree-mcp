"""Project store for mltree-mcp.

Layout inside a project::

    <project>/.mltree/
        pipeline.yaml        # nodes, edges, default params (git-friendly, rewritten on every change)
        mltree.db            # SQLite: steps, edges, runs (source of truth for queries)
        runs/<run_id>/       # run.json, a snapshot of the step script, copied artifacts

SQLite is the source of truth at runtime. ``pipeline.yaml`` is regenerated on
every pipeline change so it diffs nicely in git; if the database is missing (for
example a fresh clone where only the YAML was committed) the steps and edges are
re-imported from the YAML.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

STORE_DIRNAME = ".mltree"
DB_FILENAME = "mltree.db"
PIPELINE_FILENAME = "pipeline.yaml"
RUNS_DIRNAME = "runs"
SCHEMA_VERSION = 2

RUN_STATUSES = ("pending", "running", "succeeded", "failed", "cancelled")
STEP_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
MAX_COPY_BYTES = 50 * 1024 * 1024  # artifacts bigger than this are referenced, not copied

RUN_KINDS = ("run", "sweep")
# columns added in schema v2 (0.5.0): sweeps and runs launched by mltree itself
RUN_COLUMNS_V2 = [
    ("kind", "TEXT NOT NULL DEFAULT 'run'"),   # 'run' or 'sweep' (a parent that groups child runs)
    ("sweep_id", "TEXT"),                      # the sweep this run belongs to
    ("sweep", "TEXT"),                         # JSON spec on sweep rows: {metric, mode, grid, source, ...}
    ("command", "TEXT"),                       # JSON argv when mltree launched the run
    ("exit_code", "INTEGER"),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    script_path TEXT,
    description TEXT NOT NULL DEFAULT '',
    inputs      TEXT NOT NULL DEFAULT '[]',   -- JSON list of data names
    outputs     TEXT NOT NULL DEFAULT '[]',   -- JSON list of data names
    params      TEXT NOT NULL DEFAULT '{}',   -- JSON object, default params
    kind        TEXT NOT NULL DEFAULT 'step', -- reserved for the viewer (load, split, train ...)
    pos_x       REAL,                         -- reserved for the viewer (node position)
    pos_y       REAL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL REFERENCES steps(name) ON UPDATE CASCADE ON DELETE CASCADE,
    target     TEXT NOT NULL REFERENCES steps(name) ON UPDATE CASCADE ON DELETE CASCADE,
    label      TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE (source, target)
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    step          TEXT NOT NULL REFERENCES steps(name) ON UPDATE CASCADE ON DELETE CASCADE,
    status        TEXT NOT NULL CHECK (status IN ('pending','running','succeeded','failed','cancelled')),
    params        TEXT NOT NULL DEFAULT '{}',  -- JSON object
    metrics       TEXT NOT NULL DEFAULT '{}',  -- JSON object of name -> number
    artifacts     TEXT NOT NULL DEFAULT '[]',  -- JSON list of {name, path, stored, bytes}
    parent_run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    notes         TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '[]',
    script_sha256 TEXT,
    git_commit    TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_state (      -- what scan_project last wrote for a step
    step       TEXT PRIMARY KEY,
    script_path TEXT,
    fields     TEXT NOT NULL,                  -- JSON of the fields as written by the scan
    scanned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS imports (         -- external results already imported as runs
    key         TEXT PRIMARY KEY,
    run_id      TEXT,
    source      TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_step    ON runs(step, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_parent  ON runs(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
"""


class MLTreeError(ValueError):
    """A user-facing error with a message meant for the assistant to read."""


class ProjectNotInitialized(MLTreeError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def resolve_project_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Project root: explicit arg > $MLTREE_PROJECT > $CLAUDE_PROJECT_DIR > cwd."""
    raw = explicit or os.environ.get("MLTREE_PROJECT") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise MLTreeError(f"Project directory does not exist: {root}")
    if not root.is_dir():
        raise MLTreeError(f"Project path is not a directory: {root}")
    return root


# ----------------------------------------------------------------- validation


def _check_name(name: Any, what: str = "step name") -> str:
    if not isinstance(name, str) or not name.strip():
        raise MLTreeError(f"{what} must be a non-empty string")
    name = name.strip()
    if not STEP_NAME_RE.match(name):
        raise MLTreeError(
            f"Invalid {what} {name!r}: use 1-64 letters, digits, '_', '-' or '.', starting with a letter, digit or '_'"
        )
    return name


def _check_str_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise MLTreeError(f"{field} must be a list of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise MLTreeError(f"{field} must contain only non-empty strings, got {item!r}")
        if item.strip() not in out:
            out.append(item.strip())
    return out


def _check_json_obj(value: Any, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise MLTreeError(f"{field} must be an object (dict), got {type(value).__name__}")
    for k in value:
        if not isinstance(k, str):
            raise MLTreeError(f"{field} keys must be strings, got {k!r}")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MLTreeError(f"{field} must be JSON-serialisable (no NaN/inf): {exc}") from exc
    return value


def _check_metrics(value: Any) -> dict[str, float | int]:
    metrics = _check_json_obj(value, "metrics")
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise MLTreeError(f"metric {k!r} must be a number, got {v!r}")
        if isinstance(v, float) and not math.isfinite(v):
            raise MLTreeError(f"metric {k!r} must be finite, got {v!r}")
    return metrics


def _duration(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except ValueError:
        return None
    if (a.tzinfo is None) != (b.tzinfo is None):
        return None
    return round((b - a).total_seconds(), 3)


def _loads(s: str | None, default: Any) -> Any:
    if s is None:
        return default
    return json.loads(s)


# ---------------------------------------------------------------------- store


@dataclass
class Store:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()

    # paths
    @property
    def dir(self) -> Path:
        return self.root / STORE_DIRNAME

    @property
    def db_path(self) -> Path:
        return self.dir / DB_FILENAME

    @property
    def pipeline_path(self) -> Path:
        return self.dir / PIPELINE_FILENAME

    @property
    def runs_dir(self) -> Path:
        return self.dir / RUNS_DIRNAME

    @classmethod
    def from_env(cls) -> "Store":
        return cls(resolve_project_root())

    def is_initialized(self) -> bool:
        return self.dir.is_dir() and (self.db_path.exists() or self.pipeline_path.exists())

    # connection
    @staticmethod
    def _bump(c: sqlite3.Connection) -> None:
        c.execute(
            "INSERT INTO meta(key, value) VALUES ('revision', '1') "
            "ON CONFLICT(key) DO UPDATE SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT)"
        )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _require(self) -> None:
        if not self.is_initialized():
            raise ProjectNotInitialized(
                f"No mltree project at {self.root} (missing {STORE_DIRNAME}/). Call init_project first, "
                "or set MLTREE_PROJECT to the project directory."
            )
        if not self.db_path.exists():
            # Fresh clone with only pipeline.yaml committed: rebuild the index.
            self._create_db()
            self._import_yaml()
        elif not getattr(self, "_schema_ok", False):
            self._create_db()  # idempotent; adds tables introduced by newer versions
        self._schema_ok = True

    # -------------------------------------------------------------- project

    def init(self, name: str | None = None) -> dict[str, Any]:
        created = not self.dir.exists()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        db_existed = self.db_path.exists()
        self._create_db()
        with self._conn() as c:
            if name:
                c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('project_name', ?)", (name.strip(),))
            else:
                c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('project_name', ?)", (self.root.name,))
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('created_at', ?)", (utcnow(),))
            self._bump(c)
        if not db_existed and self.pipeline_path.exists():
            self._import_yaml()
        self._write_yaml()
        return {
            "project": self.project_name(),
            "root": str(self.root),
            "store": str(self.dir),
            "created": created,
            "message": ("Initialized" if created else "Already initialized") + f" mltree project at {self.dir}",
        }

    def _create_db(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            have = {r["name"] for r in c.execute("PRAGMA table_info(runs)")}
            for col, decl in RUN_COLUMNS_V2:
                if col not in have:  # migrate stores created by 0.4.x and older
                    c.execute(f"ALTER TABLE runs ADD COLUMN {col} {decl}")
            c.execute("CREATE INDEX IF NOT EXISTS idx_runs_sweep ON runs(sweep_id)")
            c.execute("UPDATE meta SET value=? WHERE key='schema_version' AND CAST(value AS INTEGER) < ?",
                      (str(SCHEMA_VERSION), SCHEMA_VERSION))
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def project_name(self) -> str:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key='project_name'").fetchone()
        return row["value"] if row else self.root.name

    # ---------------------------------------------------------------- steps

    def _rel_script(self, script_path: str | None) -> str | None:
        if script_path is None or (isinstance(script_path, str) and not script_path.strip()):
            return None
        if not isinstance(script_path, str):
            raise MLTreeError("script_path must be a string path relative to the project root")
        p = Path(script_path.strip()).expanduser()
        abs_p = (p if p.is_absolute() else self.root / p).resolve()
        try:
            rel = abs_p.relative_to(self.root)
        except ValueError as exc:
            raise MLTreeError(f"script_path {script_path!r} is outside the project root {self.root}") from exc
        return rel.as_posix()

    def add_step(
        self,
        name: str,
        script_path: str | None = None,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        params: dict[str, Any] | None = None,
        description: str = "",
        kind: str = "step",
    ) -> dict[str, Any]:
        self._require()
        name = _check_name(name)
        script = self._rel_script(script_path)
        inputs = _check_str_list(inputs, "inputs")
        outputs = _check_str_list(outputs, "outputs")
        params = _check_json_obj(params, "params")
        if description is None:
            description = ""
        if not isinstance(description, str):
            raise MLTreeError("description must be a string")
        kind = (kind or "step").strip() or "step"
        now = utcnow()
        with self._conn() as c:
            existing = c.execute("SELECT 1 FROM steps WHERE name=?", (name,)).fetchone()
            if existing:
                c.execute(
                    "UPDATE steps SET script_path=?, description=?, inputs=?, outputs=?, params=?, kind=?, updated_at=? "
                    "WHERE name=?",
                    (script, description, json.dumps(inputs), json.dumps(outputs), json.dumps(params), kind, now, name),
                )
            else:
                c.execute(
                    "INSERT INTO steps(name, script_path, description, inputs, outputs, params, kind, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (name, script, description, json.dumps(inputs), json.dumps(outputs), json.dumps(params), kind, now, now),
                )
            self._bump(c)
        self._write_yaml()
        step = self.get_step(name)
        step["action"] = "updated" if existing else "created"
        warnings = []
        if script and not step["script_exists"]:
            warnings.append(f"script {script} does not exist yet")
        if warnings:
            step["warnings"] = warnings
        return step

    def _step_row_to_dict(self, r: sqlite3.Row) -> dict[str, Any]:
        script = r["script_path"]
        return {
            "name": r["name"],
            "kind": r["kind"],
            "script_path": script,
            "script_exists": bool(script) and (self.root / script).is_file(),
            "description": r["description"],
            "inputs": _loads(r["inputs"], []),
            "outputs": _loads(r["outputs"], []),
            "params": _loads(r["params"], {}),
            "position": None if r["pos_x"] is None else {"x": r["pos_x"], "y": r["pos_y"]},
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }

    def get_step(self, name: str) -> dict[str, Any]:
        self._require()
        name = _check_name(name)
        with self._conn() as c:
            r = c.execute("SELECT * FROM steps WHERE name=?", (name,)).fetchone()
        if r is None:
            raise MLTreeError(f"Unknown step {name!r}. Known steps: {', '.join(self.step_names()) or '(none)'}")
        return self._step_row_to_dict(r)

    def step_names(self) -> list[str]:
        with self._conn() as c:
            return [r["name"] for r in c.execute("SELECT name FROM steps ORDER BY id")]

    # ---------------------------------------------------------------- edges

    def connect_steps(self, source: str, target: str, label: str = "") -> dict[str, Any]:
        self._require()
        source, target = _check_name(source, "source step"), _check_name(target, "target step")
        if source == target:
            raise MLTreeError("A step cannot be connected to itself")
        names = set(self.step_names())
        missing = [n for n in (source, target) if n not in names]
        if missing:
            raise MLTreeError(
                f"Unknown step(s): {', '.join(missing)}. Register them with add_step first. "
                f"Known steps: {', '.join(sorted(names)) or '(none)'}"
            )
        with self._conn() as c:
            if c.execute("SELECT 1 FROM edges WHERE source=? AND target=?", (source, target)).fetchone():
                return {"source": source, "target": target, "action": "exists", "message": "Edge already exists"}
            # cycle check: is source reachable from target?
            adj: dict[str, list[str]] = {}
            for e in c.execute("SELECT source, target FROM edges"):
                adj.setdefault(e["source"], []).append(e["target"])
            stack, seen = [target], set()
            while stack:
                n = stack.pop()
                if n == source:
                    raise MLTreeError(f"Connecting {source} -> {target} would create a cycle; the pipeline must stay a DAG")
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(adj.get(n, []))
            c.execute(
                "INSERT INTO edges(source, target, label, created_at) VALUES (?,?,?,?)",
                (source, target, label or "", utcnow()),
            )
            self._bump(c)
        self._write_yaml()
        src, tgt = self.get_step(source), self.get_step(target)
        shared = [x for x in src["outputs"] if x in tgt["inputs"]]
        result: dict[str, Any] = {"source": source, "target": target, "action": "created", "shared_data": shared}
        if src["outputs"] and tgt["inputs"] and not shared:
            result["warnings"] = [f"{source} outputs {src['outputs']} do not match any of {target} inputs {tgt['inputs']}"]
        return result

    def edges(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return [
                {"source": r["source"], "target": r["target"], "label": r["label"]}
                for r in c.execute("SELECT source, target, label FROM edges ORDER BY id")
            ]

    # ----------------------------------------------------------------- runs

    def _new_run_id(self) -> str:
        return f"r_{datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')}_{secrets.token_hex(3)}"

    def _git_commit(self) -> str | None:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.root, capture_output=True, text=True, timeout=3
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() or None if out.returncode == 0 else None

    def _normalize_artifacts(self, artifacts: Any) -> list[tuple[str, str]]:
        if artifacts is None:
            return []
        if isinstance(artifacts, str):
            artifacts = [artifacts]
        pairs: list[tuple[str, str]] = []
        if isinstance(artifacts, dict):
            for k, v in artifacts.items():
                if not isinstance(k, str) or not isinstance(v, str) or not v.strip():
                    raise MLTreeError("artifacts dict must map names to path strings")
                pairs.append((k, v.strip()))
        elif isinstance(artifacts, (list, tuple)):
            for v in artifacts:
                if not isinstance(v, str) or not v.strip():
                    raise MLTreeError("artifacts must be a list of path strings or a {name: path} object")
                pairs.append((Path(v.strip()).name, v.strip()))
        else:
            raise MLTreeError("artifacts must be a list of path strings or a {name: path} object")
        return pairs

    def log_run(
        self,
        step: str,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        artifacts: list[str] | dict[str, str] | None = None,
        parent_run_id: str | None = None,
        notes: str = "",
        status: str = "succeeded",
        tags: list[str] | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
        copy_artifacts: bool = True,
        sweep_id: str | None = None,
        kind: str = "run",
        sweep: dict[str, Any] | None = None,
        command: list[str] | None = None,
        exit_code: int | None = None,
    ) -> dict[str, Any]:
        self._require()
        step_info = self.get_step(step)
        kind = (kind or "run").strip().lower()
        if kind not in RUN_KINDS:
            raise MLTreeError(f"kind must be one of {', '.join(RUN_KINDS)}")
        if sweep is not None and not isinstance(sweep, dict):
            raise MLTreeError("sweep must be an object like {\"metric\": \"val_rmse\", \"mode\": \"min\"}")
        step = step_info["name"]
        params = _check_json_obj(params, "params")
        metrics = _check_metrics(metrics)
        tags = _check_str_list(tags, "tags")
        status = (status or "succeeded").strip().lower()
        if status not in RUN_STATUSES:
            raise MLTreeError(f"status must be one of {', '.join(RUN_STATUSES)}, got {status!r}")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise MLTreeError("notes must be a string")
        if parent_run_id is not None and not (isinstance(parent_run_id, str) and parent_run_id.strip()):
            parent_run_id = None
        if parent_run_id:
            parent_run_id = parent_run_id.strip()
            with self._conn() as c:
                prow = c.execute("SELECT id, kind, sweep_id FROM runs WHERE id=?", (parent_run_id,)).fetchone()
                if not prow:
                    raise MLTreeError(f"parent_run_id {parent_run_id!r} does not exist; use list_runs to find run ids")
            if not sweep_id and prow["kind"] == "sweep":
                sweep_id = prow["id"]  # a child of a sweep row belongs to that sweep
        if sweep_id is not None and not (isinstance(sweep_id, str) and sweep_id.strip()):
            sweep_id = None
        if sweep_id:
            sweep_id = sweep_id.strip()
            with self._conn() as c:
                srow = c.execute("SELECT kind FROM runs WHERE id=?", (sweep_id,)).fetchone()
            if srow is None or srow["kind"] != "sweep":
                raise MLTreeError(f"sweep_id {sweep_id!r} is not a sweep; create one with start_sweep")
            parent_run_id = parent_run_id or sweep_id

        # resolve artifacts before writing anything
        resolved: list[tuple[str, str, Path]] = []
        for art_name, raw in self._normalize_artifacts(artifacts):
            p = Path(raw).expanduser()
            abs_p = (p if p.is_absolute() else self.root / p).resolve()
            if not abs_p.exists():
                raise MLTreeError(f"artifact {raw!r} not found (paths are relative to {self.root})")
            resolved.append((art_name, raw, abs_p))

        run_id = self._new_run_id()
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "artifacts").mkdir()

        art_records: list[dict[str, Any]] = []
        used: set[str] = set()
        for art_name, raw, abs_p in resolved:
            size = abs_p.stat().st_size if abs_p.is_file() else None
            rec: dict[str, Any] = {"name": art_name, "source": raw, "bytes": size}
            if copy_artifacts and abs_p.is_file() and size is not None and size <= MAX_COPY_BYTES:
                fname = art_name if "." in art_name else (art_name + abs_p.suffix)
                base, i = fname, 1
                while fname in used:
                    fname = f"{Path(base).stem}_{i}{Path(base).suffix}"
                    i += 1
                used.add(fname)
                shutil.copy2(abs_p, run_dir / "artifacts" / fname)
                rec["path"] = (run_dir / "artifacts" / fname).relative_to(self.root).as_posix()
                rec["stored"] = "copied"
            else:
                try:
                    rec["path"] = abs_p.relative_to(self.root).as_posix()
                except ValueError:
                    rec["path"] = str(abs_p)
                rec["stored"] = "reference"
            art_records.append(rec)

        script_sha = None
        if step_info["script_exists"]:
            src = self.root / step_info["script_path"]
            data = src.read_bytes()
            script_sha = hashlib.sha256(data).hexdigest()
            (run_dir / ("script" + (src.suffix or ".py"))).write_bytes(data)

        now = utcnow()
        record = {
            "id": run_id,
            "step": step,
            "status": status,
            "params": params,
            "metrics": metrics,
            "artifacts": art_records,
            "parent_run_id": parent_run_id,
            "notes": notes,
            "tags": tags,
            "script_sha256": script_sha,
            "git_commit": self._git_commit(),
            "started_at": started_at,
            "finished_at": finished_at or (now if status in ("succeeded", "failed", "cancelled") else None),
            "created_at": now,
            "kind": kind,
            "sweep_id": sweep_id,
            "sweep": sweep,
            "command": command,
            "exit_code": exit_code,
        }
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs(id, step, status, params, metrics, artifacts, parent_run_id, notes, tags, "
                "script_sha256, git_commit, started_at, finished_at, created_at, kind, sweep_id, sweep, command, exit_code) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, step, status, json.dumps(params), json.dumps(metrics), json.dumps(art_records),
                    parent_run_id, notes, json.dumps(tags), script_sha, record["git_commit"],
                    started_at, record["finished_at"], now, kind, sweep_id,
                    json.dumps(sweep) if sweep is not None else None,
                    json.dumps(command) if command is not None else None, exit_code,
                ),
            )
            self._bump(c)
        (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        record["run_dir"] = run_dir.relative_to(self.root).as_posix()
        return record

    def _run_row_to_dict(self, r: sqlite3.Row) -> dict[str, Any]:
        keys = r.keys()
        return {
            "id": r["id"],
            "step": r["step"],
            "status": r["status"],
            "params": _loads(r["params"], {}),
            "metrics": _loads(r["metrics"], {}),
            "artifacts": _loads(r["artifacts"], []),
            "parent_run_id": r["parent_run_id"],
            "notes": r["notes"],
            "tags": _loads(r["tags"], []),
            "script_sha256": r["script_sha256"],
            "git_commit": r["git_commit"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
            "created_at": r["created_at"],
            "kind": r["kind"] if "kind" in keys else "run",
            "sweep_id": r["sweep_id"] if "sweep_id" in keys else None,
            "sweep": _loads(r["sweep"], None) if "sweep" in keys and r["sweep"] else None,
            "command": _loads(r["command"], None) if "command" in keys and r["command"] else None,
            "exit_code": r["exit_code"] if "exit_code" in keys else None,
            "duration_s": _duration(r["started_at"], r["finished_at"]),
        }

    def update_run(
        self,
        run_id: str,
        status: str | None = None,
        metrics: dict[str, float] | None = None,
        finished_at: str | None = None,
        exit_code: int | None = None,
        notes: str | None = None,
        sweep: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Update a run in place (used by the runner while a run is live). Metrics are merged."""
        run = self.get_run(run_id)
        sets: dict[str, Any] = {}
        if status is not None:
            status = status.strip().lower()
            if status not in RUN_STATUSES:
                raise MLTreeError(f"status must be one of {', '.join(RUN_STATUSES)}")
            sets["status"] = status
            if status in ("succeeded", "failed", "cancelled") and not (finished_at or run["finished_at"]):
                finished_at = utcnow()
        if metrics:
            sets["metrics"] = json.dumps({**run["metrics"], **_check_metrics(metrics)})
        if finished_at is not None:
            sets["finished_at"] = finished_at
        if exit_code is not None:
            sets["exit_code"] = int(exit_code)
        if notes is not None:
            sets["notes"] = notes
        if sweep is not None:
            sets["sweep"] = json.dumps({**(run["sweep"] or {}), **sweep})
        if artifacts is not None:
            sets["artifacts"] = json.dumps(run["artifacts"] + artifacts)
        if sets:
            with self._conn() as c:
                c.execute(f"UPDATE runs SET {', '.join(k + '=?' for k in sets)} WHERE id=?", [*sets.values(), run_id])
                self._bump(c)
        new = self.get_run(run_id)
        run_dir = self.runs_dir / run_id
        if run_dir.is_dir():
            (run_dir / "run.json").write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        return new

    def sweep_children(self, sweep_id: str) -> list[dict[str, Any]]:
        self._require()
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs WHERE sweep_id=? ORDER BY created_at, rowid", (sweep_id,)).fetchall()
        return [self._run_row_to_dict(r) for r in rows]

    def sweep_summary(self, sweep_id: str, metric: str | None = None, mode: str | None = None) -> dict[str, Any]:
        """Children count by status and the best child by the sweep's metric (or the one given)."""
        sw = self.get_run(sweep_id)
        spec = sw["sweep"] or {}
        metric = metric or spec.get("metric")
        mode = (mode or spec.get("mode") or "max").lower()
        kids = self.sweep_children(sweep_id)
        counts: dict[str, int] = {}
        for k in kids:
            counts[k["status"]] = counts.get(k["status"], 0) + 1
        best = None
        if metric:
            ok = [k for k in kids if k["status"] == "succeeded" and isinstance(k["metrics"].get(metric), (int, float))]
            if ok:
                best = (min if mode == "min" else max)(ok, key=lambda k: k["metrics"][metric])
        return {"sweep_id": sweep_id, "n": len(kids), "counts": counts, "metric": metric, "mode": mode,
                "best_run_id": best["id"] if best else None, "best_value": best["metrics"][metric] if best else None,
                "best_params": best["params"] if best else None}

    def get_run(self, run_id: str) -> dict[str, Any]:
        self._require()
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if r is None:
            raise MLTreeError(f"Unknown run id {run_id!r}")
        return self._run_row_to_dict(r)

    def list_runs(
        self,
        step: str | None = None,
        status: str | None = None,
        parent_run_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self._require()
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 1000:
            raise MLTreeError("limit must be an integer between 1 and 1000")
        where, args = [], []
        if step:
            where.append("step=?")
            args.append(self.get_step(step)["name"])
        if status:
            status = status.strip().lower()
            if status not in RUN_STATUSES:
                raise MLTreeError(f"status must be one of {', '.join(RUN_STATUSES)}")
            where.append("status=?")
            args.append(status)
        if parent_run_id:
            where.append("parent_run_id=?")
            args.append(parent_run_id)
        sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [self._run_row_to_dict(r) for r in c.execute(sql, args)]

    def best_run(self, metric: str, mode: str = "max", step: str | None = None) -> dict[str, Any]:
        self._require()
        if not isinstance(metric, str) or not metric.strip():
            raise MLTreeError("metric must be a non-empty string, e.g. 'val_rmse'")
        metric = metric.strip()
        mode = (mode or "max").strip().lower()
        if mode not in ("min", "max"):
            raise MLTreeError("mode must be 'min' or 'max'")
        if '"' in metric or "\\" in metric:
            raise MLTreeError("metric name must not contain quotes or backslashes")
        path = '$."' + metric + '"'
        where = ["status='succeeded'", "json_type(metrics, ?) IN ('integer','real')"]
        args: list[Any] = [path]
        if step:
            where.append("step=?")
            args.append(self.get_step(step)["name"])
        order = "DESC" if mode == "max" else "ASC"
        cond = " AND ".join(where)
        sql = f"SELECT *, json_extract(metrics, ?) AS mv FROM runs WHERE {cond} ORDER BY mv {order}, created_at ASC, rowid ASC LIMIT 1"
        with self._conn() as c:
            r = c.execute(sql, [path, *args]).fetchone()
            n = c.execute(f"SELECT COUNT(*) AS n FROM runs WHERE {cond}", args).fetchone()["n"]
        if r is None:
            scope = f" for step {step!r}" if step else ""
            raise MLTreeError(f"No succeeded runs{scope} have a numeric metric {metric!r}")
        run = self._run_row_to_dict(r)
        return {"metric": metric, "mode": mode, "value": run["metrics"][metric], "candidates": n, "run": run}

    # ------------------------------------------------------------- pipeline

    def get_pipeline(self) -> dict[str, Any]:
        self._require()
        with self._conn() as c:
            steps = [self._step_row_to_dict(r) for r in c.execute("SELECT * FROM steps ORDER BY id")]
            latest = {
                r["step"]: r
                for r in c.execute(
                    "SELECT r.* FROM runs r JOIN (SELECT step, MAX(rowid) AS mr FROM runs GROUP BY step) m "
                    "ON r.rowid = m.mr"
                )
            }
            counts = {r["step"]: r["n"] for r in c.execute("SELECT step, COUNT(*) AS n FROM runs GROUP BY step")}
        for s in steps:
            lr = latest.get(s["name"])
            s["run_count"] = counts.get(s["name"], 0)
            s["latest_run"] = (
                None
                if lr is None
                else {
                    "id": lr["id"],
                    "status": lr["status"],
                    "metrics": _loads(lr["metrics"], {}),
                    "created_at": lr["created_at"],
                }
            )
            s["status"] = lr["status"] if lr is not None else "not_run"
        return {
            "project": self.project_name(),
            "root": str(self.root),
            "nodes": steps,
            "edges": self.edges(),
            "order": self.topological_order(),
        }

    def topological_order(self) -> list[str]:
        names = self.step_names()
        indeg = {n: 0 for n in names}
        adj: dict[str, list[str]] = {n: [] for n in names}
        for e in self.edges():
            adj[e["source"]].append(e["target"])
            indeg[e["target"]] += 1
        queue = [n for n in names if indeg[n] == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for t in adj[n]:
                indeg[t] -= 1
                if indeg[t] == 0:
                    queue.append(t)
        return order

    # ----------------------------------------------------------------- yaml

    def _write_yaml(self) -> None:
        with self._conn() as c:
            rows = list(c.execute("SELECT * FROM steps ORDER BY id"))
        steps: dict[str, Any] = {}
        for r in rows:
            d: dict[str, Any] = {}
            if r["kind"] != "step":
                d["kind"] = r["kind"]
            if r["script_path"]:
                d["script"] = r["script_path"]
            if r["description"]:
                d["description"] = r["description"]
            d["inputs"] = _loads(r["inputs"], [])
            d["outputs"] = _loads(r["outputs"], [])
            d["params"] = _loads(r["params"], {})
            steps[r["name"]] = d
        doc = {
            "version": SCHEMA_VERSION,
            "project": self.project_name(),
            "steps": steps,
            "edges": [{"from": e["source"], "to": e["target"], **({"label": e["label"]} if e["label"] else {})} for e in self.edges()],
        }
        header = "# Generated by mltree-mcp. Safe to commit; edits are re-imported only when mltree.db is missing.\n"
        tmp = self.pipeline_path.with_suffix(".yaml.tmp")
        tmp.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
        tmp.replace(self.pipeline_path)

    def _import_yaml(self) -> None:
        try:
            doc = yaml.safe_load(self.pipeline_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise MLTreeError(f"Could not parse {self.pipeline_path}: {exc}") from exc
        if not isinstance(doc, dict):
            raise MLTreeError(f"{self.pipeline_path} must contain a mapping")
        now = utcnow()
        with self._conn() as c:
            if doc.get("project"):
                c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('project_name', ?)", (str(doc["project"]),))
            for name, d in (doc.get("steps") or {}).items():
                d = d or {}
                name = _check_name(name)
                c.execute(
                    "INSERT OR IGNORE INTO steps(name, script_path, description, inputs, outputs, params, kind, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        name, d.get("script"), d.get("description", "") or "",
                        json.dumps(_check_str_list(d.get("inputs"), "inputs")),
                        json.dumps(_check_str_list(d.get("outputs"), "outputs")),
                        json.dumps(_check_json_obj(d.get("params"), "params")),
                        d.get("kind", "step"), now, now,
                    ),
                )
            for e in doc.get("edges") or []:
                c.execute(
                    "INSERT OR IGNORE INTO edges(source, target, label, created_at) VALUES (?,?,?,?)",
                    (e["from"], e["to"], e.get("label", ""), now),
                )

    # ------------------------------------------------------- viewer helpers

    def revision(self) -> int:
        """Monotonic counter bumped on every write; the viewer polls it for live refresh."""
        if not self.db_path.exists():
            return 0
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key='revision'").fetchone()
        return int(row["value"]) if row else 0

    def set_positions(self, positions: dict[str, Any]) -> dict[str, Any]:
        """Save node positions: {step: {"x": float, "y": float} | None}. None clears a position."""
        self._require()
        if not isinstance(positions, dict) or not positions:
            raise MLTreeError("positions must be a non-empty object of {step: {x, y}}")
        names = set(self.step_names())
        clean: list[tuple[float | None, float | None, str]] = []
        for name, pos in positions.items():
            if name not in names:
                raise MLTreeError(f"Unknown step {name!r}")
            if pos is None:
                clean.append((None, None, name))
                continue
            try:
                x, y = float(pos["x"]), float(pos["y"])
            except (TypeError, KeyError, ValueError) as exc:
                raise MLTreeError(f"position for {name!r} must be {{x: number, y: number}}") from exc
            if not (math.isfinite(x) and math.isfinite(y)):
                raise MLTreeError(f"position for {name!r} must be finite")
            clean.append((x, y, name))
        with self._conn() as c:
            c.executemany("UPDATE steps SET pos_x=?, pos_y=? WHERE name=?", clean)
            self._bump(c)
        return {"updated": [n for _, _, n in clean]}

    def _snapshot_file(self, run_id: str) -> Path | None:
        d = self.runs_dir / run_id
        if not d.is_dir():
            return None
        files = sorted(d.glob("script.*"))
        return files[0] if files else None

    def script(self, step: str, run_id: str | None = None) -> dict[str, Any]:
        """The step's current script, or the snapshot stored with one of its runs."""
        info = self.get_step(step)
        if run_id:
            run = self.get_run(run_id)
            if run["step"] != info["name"]:
                raise MLTreeError(f"Run {run_id!r} belongs to step {run['step']!r}, not {info['name']!r}")
            f = self._snapshot_file(run_id)
            if f is None:
                raise MLTreeError(f"Run {run_id!r} has no script snapshot (the script did not exist when it was logged)")
            data = f.read_bytes()
            return {
                "step": info["name"], "source": "snapshot", "run_id": run_id,
                "path": f.relative_to(self.root).as_posix(), "script_path": info["script_path"],
                "content": data.decode("utf-8", errors="replace"), "sha256": hashlib.sha256(data).hexdigest(),
                "created_at": run["created_at"],
            }
        if not info["script_path"]:
            raise MLTreeError(f"Step {info['name']!r} has no script_path")
        f = self.root / info["script_path"]
        if not f.is_file():
            raise MLTreeError(f"Script {info['script_path']} does not exist")
        data = f.read_bytes()
        return {
            "step": info["name"], "source": "current", "run_id": None, "path": info["script_path"],
            "script_path": info["script_path"], "content": data.decode("utf-8", errors="replace"),
            "sha256": hashlib.sha256(data).hexdigest(), "created_at": None,
        }

    def snapshots(self, step: str) -> list[dict[str, Any]]:
        """Runs of a step that carry a script snapshot, newest first, with whether it differs from the current file."""
        info = self.get_step(step)
        current_sha = None
        if info["script_exists"]:
            current_sha = hashlib.sha256((self.root / info["script_path"]).read_bytes()).hexdigest()
        out = []
        for r in self.list_runs(step=info["name"], limit=1000):
            if r["script_sha256"] and self._snapshot_file(r["id"]) is not None:
                out.append({
                    "run_id": r["id"], "created_at": r["created_at"], "status": r["status"],
                    "sha256": r["script_sha256"], "same_as_current": r["script_sha256"] == current_sha,
                })
        return out

    def metric_names(self) -> list[str]:
        self._require()
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT j.key AS k FROM runs, json_each(runs.metrics) AS j "
                "WHERE j.type IN ('integer','real') ORDER BY k"
            ).fetchall()
        return [r["k"] for r in rows]

    def run_tree(self, metric: str | None = None, mode: str = "max", step: str | None = None) -> dict[str, Any]:
        """All runs as a forest built from parent_run_id, plus the best run for `metric` if given."""
        runs = self.list_runs(step=step, limit=1000)
        ids = {r["id"] for r in runs}
        edges = [{"source": r["parent_run_id"], "target": r["id"]} for r in runs if r["parent_run_id"] in ids]
        best = None
        if metric:
            try:
                best = self.best_run(metric, mode=mode, step=step)
            except MLTreeError:
                best = None
        runs.reverse()  # oldest first reads better in a tree
        return {
            "runs": runs,
            "edges": edges,
            "roots": [r["id"] for r in runs if r["parent_run_id"] not in ids],
            "metric": metric,
            "mode": mode,
            "best_run_id": best["run"]["id"] if best else None,
            "best_value": best["value"] if best else None,
            "metrics": self.metric_names(),
            "sweeps": {r["id"]: self.sweep_summary(r["id"], metric=None if (r["sweep"] or {}).get("metric") else metric,
                                                   mode=None if (r["sweep"] or {}).get("metric") else mode)
                       for r in runs if r["kind"] == "sweep"},
        }

    def compare(self, run_ids: list[str]) -> dict[str, Any]:
        """Side-by-side params and metrics for two or more runs."""
        self._require()
        run_ids = _check_str_list(run_ids, "run_ids")
        if len(run_ids) < 2:
            raise MLTreeError("Pick at least 2 runs to compare")
        runs = [self.get_run(r) for r in run_ids]
        pkeys = sorted({k for r in runs for k in r["params"]})
        mkeys = sorted({k for r in runs for k in r["metrics"]})

        def differs(key: str, field: str) -> bool:
            vals = [json.dumps(r[field].get(key), sort_keys=True) for r in runs]
            return len(set(vals)) > 1

        return {
            "runs": runs,
            "param_keys": pkeys,
            "metric_keys": mkeys,
            "differing_params": [k for k in pkeys if differs(k, "params")],
            "differing_metrics": [k for k in mkeys if differs(k, "metrics")],
        }

    # ------------------------------------------------------ scan helpers

    def scan_state(self) -> dict[str, dict[str, Any]]:
        self._require()
        with self._conn() as c:
            return {r["step"]: {"script_path": r["script_path"], "fields": json.loads(r["fields"])}
                    for r in c.execute("SELECT * FROM scan_state")}

    def set_scan_state(self, step: str, script_path: str | None, fields: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO scan_state(step, script_path, fields, scanned_at) VALUES (?,?,?,?) "
                "ON CONFLICT(step) DO UPDATE SET script_path=excluded.script_path, fields=excluded.fields, "
                "scanned_at=excluded.scanned_at",
                (step, script_path, json.dumps(fields, sort_keys=True), utcnow()),
            )

    def imported_keys(self) -> set[str]:
        self._require()
        with self._conn() as c:
            return {r["key"] for r in c.execute("SELECT key FROM imports")}

    def imported_run_ids(self) -> dict[str, str]:
        self._require()
        with self._conn() as c:
            return {r["key"]: r["run_id"] for r in c.execute(
                "SELECT i.key, i.run_id FROM imports i JOIN runs r ON r.id = i.run_id")}

    def mark_imported(self, key: str, run_id: str | None, source: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO imports(key, run_id, source, imported_at) VALUES (?,?,?,?)",
                      (key, run_id, source, utcnow()))
