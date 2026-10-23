"""Data versioning: content fingerprints of the files a step reads and writes.

A step's ``inputs`` / ``outputs`` are data names. A name is versioned when it resolves to a file or directory:
``<root>/<name>``, ``<script dir>/<name>``, else a unique file with that basename found within 4 levels of the
project (the same rule the runner uses to find outputs). Nothing is ever executed; this only reads files.

Fingerprint = SHA-256 of the content (directories: of the sorted ``path\\0sha`` list of their files). Hashes are
cached in ``mltree.db`` keyed by path + size + mtime_ns, so unchanged files are not re-read (the fast path).
Very large files and directories fall back to a size+mtime digest and are marked ``quick``.

Runs record the fingerprints they saw (``run["data"] = {"inputs": {...}, "outputs": {...}}``): inputs when the
run starts or is logged, outputs when it succeeds. ``data_status`` compares the files on disk with what the
latest run of each step recorded and flags changed inputs.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import Store

SKIP_DIRS = {".git", ".mltree", ".venv", "venv", "env", "node_modules", "__pycache__", ".ipynb_checkpoints", ".tox",
             ".mypy_cache", ".pytest_cache"}
MAX_DIR_FILES = 5000              # bigger directories get a quick (size+mtime) digest
MAX_HASH_BYTES = 4 * 1024 ** 3    # bigger files get a quick digest
SEARCH_DEPTH = 4
SEARCH_MAX_ENTRIES = 50_000
CHUNK = 1 << 20
VERSION_LEN = 12


def short(sha: str | None) -> str | None:
    return sha[:VERSION_LEN] if sha else None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


class DataIndex:
    """Resolves data names to paths and fingerprints them, reusing cached hashes. One instance per request."""

    def __init__(self, store: "Store") -> None:
        self.store = store
        self.root = store.root
        self._basenames: dict[str, list[Path]] | None = None
        self._cache: dict[str, tuple[int, int, str]] = store.hash_cache()
        self._new: dict[str, tuple[int, int, str]] = {}
        self._fp: dict[tuple[str, str], dict[str, Any]] = {}

    # -- resolution
    def _index(self) -> dict[str, list[Path]]:
        if self._basenames is None:
            idx: dict[str, list[Path]] = {}
            n = 0
            for dirpath, dirnames, filenames in os.walk(self.root):
                rel = Path(dirpath).relative_to(self.root)
                dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
                if len(rel.parts) >= SEARCH_DEPTH:
                    dirnames[:] = []
                for f in filenames:
                    idx.setdefault(f, []).append(Path(dirpath) / f)
                n += len(filenames) + len(dirnames)
                if n > SEARCH_MAX_ENTRIES:
                    break
            self._basenames = idx
        return self._basenames

    def resolve(self, name: str, script_path: str | None = None) -> tuple[Path | None, list[str]]:
        """(path, other candidates). Absolute names are used as is."""
        raw = Path(str(name)).expanduser()
        if raw.is_absolute():
            return (raw if raw.exists() else None), []
        cands = [self.root / raw]
        if script_path:
            cands.append((self.root / script_path).parent / raw)
        for c in cands:
            if c.exists():
                return c, []
        if len(raw.parts) == 1:
            hits = self._index().get(raw.name, [])
            if hits:
                hits = sorted(hits, key=lambda p: (len(p.parts), str(p)))
                return hits[0], [self.rel(h) for h in hits[1:4]]
        return None, []

    def rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    # -- hashing
    def _hash(self, p: Path, st: os.stat_result) -> tuple[str, bool]:
        key = str(p.resolve())
        hit = self._cache.get(key) or self._new.get(key)
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2], False
        if st.st_size > MAX_HASH_BYTES:
            return hashlib.sha256(f"quick:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest(), True
        sha = _sha_file(p)
        self._new[key] = (st.st_size, st.st_mtime_ns, sha)
        return sha, False

    def fingerprint(self, name: str, script_path: str | None = None) -> dict[str, Any]:
        memo = (str(name), script_path or "")
        if memo in self._fp:
            return self._fp[memo]
        path, others = self.resolve(name, script_path)
        out: dict[str, Any] = {"name": str(name), "exists": False, "path": None}
        if path is None:
            self._fp[memo] = out
            return out
        out["path"] = self.rel(path)
        if others:
            out["also_found"] = others
        try:
            if path.is_file():
                st = path.stat()
                sha, quick = self._hash(path, st)
                out.update(exists=True, kind="file", bytes=st.st_size, mtime=_iso(st.st_mtime), sha256=sha)
                if quick:
                    out["quick"] = True
            elif path.is_dir():
                out.update(self._dir(path), exists=True, kind="dir")
        except OSError as exc:
            out["error"] = str(exc)
        if out.get("sha256"):
            out["version"] = short(out["sha256"])
        self._fp[memo] = out
        return out

    def _dir(self, d: Path) -> dict[str, Any]:
        files: list[tuple[str, Path, os.stat_result]] = []
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = sorted(x for x in dirnames if x not in SKIP_DIRS)
            for f in filenames:
                p = Path(dirpath) / f
                try:
                    files.append((p.relative_to(d).as_posix(), p, p.stat()))
                except OSError:
                    continue
            if len(files) > MAX_DIR_FILES:
                break
        files.sort(key=lambda t: t[0])
        total = sum(st.st_size for _, _, st in files)
        mtime = max((st.st_mtime for _, _, st in files), default=d.stat().st_mtime)
        h = hashlib.sha256()
        quick = len(files) > MAX_DIR_FILES
        for rel, p, st in files:
            if quick:
                h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
            else:
                sha, q = self._hash(p, st)
                quick = quick or q
                h.update(f"{rel}\0{sha}\n".encode())
        out = {"bytes": total, "files": len(files), "mtime": _iso(mtime), "sha256": h.hexdigest()}
        if quick:
            out["quick"] = True
        return out

    def flush(self) -> None:
        if self._new:
            self.store.put_hashes(self._new)
            self._cache.update(self._new)
            self._new = {}

    # -- steps
    def step_data(self, step: dict[str, Any], which: tuple[str, ...] = ("inputs", "outputs")) -> dict[str, Any]:
        out = {}
        for role in which:
            out[role] = {n: compact(self.fingerprint(n, step.get("script_path"))) for n in step.get(role) or []}
        return out


def compact(fp: dict[str, Any]) -> dict[str, Any]:
    """What a run stores per data name."""
    keep = ("path", "exists", "kind", "bytes", "files", "mtime", "sha256", "version", "quick")
    return {k: fp[k] for k in keep if k in fp}


# ------------------------------------------------------------- status

_MEMO: dict[str, tuple[float, int, dict[str, Any]]] = {}
_MEMO_LOCK = threading.Lock()
MEMO_TTL = 2.0


def data_status(store: "Store", step: str | None = None, fresh: bool = False) -> dict[str, Any]:
    """Current version of every step's inputs/outputs vs what its latest run recorded; plus a data catalog."""
    key = str(store.root)
    rev = store.revision()
    if not fresh and step is None:
        with _MEMO_LOCK:
            hit = _MEMO.get(key)
        if hit and hit[1] == rev and time.monotonic() - hit[0] < MEMO_TTL:
            return hit[2]
    idx = DataIndex(store)
    pipe_steps = store.steps_raw()
    recorded = store.runs_with_data()  # oldest first
    latest: dict[str, dict[str, Any]] = {}
    producers: dict[str, str] = {}      # sha -> newest run id that wrote it
    versions: dict[str, dict[str, dict[str, Any]]] = {}
    for r in recorded:
        d = r["data"] or {}
        if r["kind"] != "sweep" and r["status"] == "succeeded" and d.get("inputs") is not None:
            latest[r["step"]] = r
        for role in ("inputs", "outputs"):
            for name, fp in (d.get(role) or {}).items():
                sha = fp.get("sha256")
                if not sha:
                    continue
                v = versions.setdefault(name, {}).setdefault(short(sha), {"version": short(sha), "first_seen": r["created_at"],
                                                                           "produced_by": [], "used_by": []})
                (v["produced_by"] if role == "outputs" else v["used_by"]).append(r["id"])
                if role == "outputs" and r["status"] == "succeeded":
                    producers[sha] = r["id"]

    steps_out: dict[str, Any] = {}
    catalog: dict[str, Any] = {}
    for s in pipe_steps:
        if step and s["name"] != step:
            continue
        cur = {role: [idx.fingerprint(n, s["script_path"]) for n in s[role]] for role in ("inputs", "outputs")}
        last = latest.get(s["name"])
        ld = (last or {}).get("data") or {}
        entry: dict[str, Any] = {"step": s["name"], "last_run_id": last["id"] if last else None,
                                 "inputs": [], "outputs": [], "changed_inputs": [], "changed_outputs": [],
                                 "missing_inputs": []}
        for role in ("inputs", "outputs"):
            for fp in cur[role]:
                was = (ld.get(role) or {}).get(fp["name"]) or {}
                item = {**compact(fp), "name": fp["name"], "run_version": was.get("version") or short(was.get("sha256"))}
                if fp.get("also_found"):
                    item["also_found"] = fp["also_found"]
                item["changed"] = bool(was.get("sha256") and fp.get("sha256") and was["sha256"] != fp["sha256"]) or \
                    bool(was.get("sha256") and not fp["exists"])
                if fp.get("sha256") and fp["sha256"] in producers:
                    item["producer_run_id"] = producers[fp["sha256"]]
                entry[role].append(item)
                if item["changed"]:
                    entry["changed_" + role].append(fp["name"])
                if role == "inputs" and not fp["exists"]:
                    entry["missing_inputs"].append(fp["name"])
                c = catalog.setdefault(fp["name"], {"name": fp["name"], "path": fp.get("path"), "exists": fp["exists"],
                                                    "version": fp.get("version"), "bytes": fp.get("bytes"),
                                                    "kind": fp.get("kind"), "produced_by_steps": [], "used_by_steps": []})
                c["produced_by_steps" if role == "outputs" else "used_by_steps"].append(s["name"])
        entry["status"] = ("changed" if entry["changed_inputs"] else "missing" if entry["missing_inputs"]
                           else "ok" if last else "no_record")
        steps_out[s["name"]] = entry
    for name, c in catalog.items():
        c["versions"] = sorted(versions.get(name, {}).values(), key=lambda v: v["first_seen"])
    idx.flush()
    changed = [n for n, e in steps_out.items() if e["changed_inputs"]]
    res = {"steps": steps_out, "data": catalog, "changed_steps": changed,
           "summary": (f"{len(changed)} step(s) have inputs that changed since their last run: "
                       + "; ".join(f"{n} ({', '.join(steps_out[n]['changed_inputs'])})" for n in changed))
           if changed else "No input changed since the last recorded run of each step."}
    if step is None:
        with _MEMO_LOCK:
            _MEMO[key] = (time.monotonic(), rev, res)
    return res
