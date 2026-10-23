"""Multi-project viewer registry: ``~/.mltree/projects.json`` (or ``$MLTREE_HOME/projects.json``).

Format (hand-editable)::

    {"projects": [{"path": "/home/you/qsar-gnn", "name": "qsar-gnn"}, {"path": "/home/you/tox21"}]}

A plain list of paths is accepted too. ``mltree-mcp projects add|remove|list`` edits it, and
``mltree-mcp view --projects`` (no folders) serves every project in it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .store import MLTreeError


def registry_path() -> Path:
    home = os.environ.get("MLTREE_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".mltree") / "projects.json"


def load_registry(path: Path | None = None) -> list[dict[str, str]]:
    p = path or registry_path()
    if not p.is_file():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MLTreeError(f"Can't read {p}: {exc}") from exc
    items = raw.get("projects", []) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise MLTreeError(f"{p}: expected {{\"projects\": [...]}} or a list of paths")
    out: list[dict[str, str]] = []
    for it in items:
        if isinstance(it, str):
            out.append({"path": it})
        elif isinstance(it, dict) and isinstance(it.get("path"), str):
            out.append({k: str(v) for k, v in it.items() if k in ("path", "name") and v})
        else:
            raise MLTreeError(f"{p}: each project must be a path or {{\"path\": ..., \"name\": ...}}")
    return out


def _save(items: list[dict[str, str]], path: Path | None = None) -> Path:
    p = path or registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"projects": items}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p


def add_project(folder: str | Path, name: str | None = None, path: Path | None = None) -> dict[str, Any]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise MLTreeError(f"Not a directory: {root}")
    items = [i for i in load_registry(path) if Path(i["path"]).expanduser().resolve() != root]
    items.append({"path": str(root), **({"name": name} if name else {})})
    return {"registry": str(_save(items, path)), "projects": items}


def remove_project(folder_or_name: str, path: Path | None = None) -> dict[str, Any]:
    items = load_registry(path)
    target = Path(folder_or_name).expanduser().resolve()
    keep = [i for i in items if i.get("name") != folder_or_name and Path(i["path"]).expanduser().resolve() != target]
    if len(keep) == len(items):
        raise MLTreeError(f"Unknown project {folder_or_name!r} in {path or registry_path()}")
    return {"registry": str(_save(keep, path)), "projects": keep}


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-.")
    return s or "project"


def resolve_projects(entries: list[str | dict[str, str]]) -> list[dict[str, Any]]:
    """Folders (or registry entries) -> [{slug, name, root}], unique slugs, missing folders rejected."""
    out: list[dict[str, Any]] = []
    seen_roots: set[Path] = set()
    slugs: set[str] = set()
    for e in entries:
        item = {"path": e} if isinstance(e, (str, Path)) else e
        root = Path(item["path"]).expanduser().resolve()
        if not root.is_dir():
            raise MLTreeError(f"Project folder not found: {root}")
        if root in seen_roots:
            continue
        seen_roots.add(root)
        name = item.get("name") or root.name
        base = slug = _slugify(name)
        n = 2
        while slug in slugs:
            slug, n = f"{base}-{n}", n + 1
        slugs.add(slug)
        out.append({"slug": slug, "name": name, "root": root, "named": bool(item.get("name"))})
    if not out:
        raise MLTreeError("No projects: pass folders to `view --projects` or add them with `mltree-mcp projects add`.")
    return out
