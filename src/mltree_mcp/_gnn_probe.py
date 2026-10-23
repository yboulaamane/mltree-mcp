"""Graph-dataset statistics, run by mltree in a subprocess with the *project's* interpreter.

Standalone on purpose (stdlib + whatever torch / torch_geometric / dgl the project has): mltree copies nothing into
the project and this file imports nothing from mltree_mcp. It is only launched when the user explicitly asks for
dataset introspection, because loading a pickled ``.pt`` file can execute code stored in it.

Usage: python _gnn_probe.py '<json list of absolute paths>'  ->  prints one line ``MLTREE_GNN_STATS {json}``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

MAX_GRAPHS = 20_000       # graphs read one by one from a list / Dataset (collated files are read in full)
MAX_VALUES = 2_000_000    # label values / edges scanned for class counts, self loops and isolated nodes
MARK = "MLTREE_GNN_STATS "


# ----------------------------------------------------------------- duck-typed accessors

def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return getattr(obj, key, None)
    except Exception:  # noqa: BLE001 - some containers raise on unknown attributes
        return None


def _shape(t: Any) -> tuple[int, ...] | None:
    if t is None:
        return None
    sh = getattr(t, "shape", None)
    if sh is not None:
        try:
            return tuple(int(x) for x in sh)
        except TypeError:
            return None
    if isinstance(t, (list, tuple)):
        if t and isinstance(t[0], (list, tuple)):
            return (len(t), len(t[0]))
        return (len(t),)
    return None


def _flat(t: Any, limit: int = MAX_VALUES) -> list[Any] | None:
    if t is None:
        return None
    try:
        if hasattr(t, "numel") and t.numel() > limit:
            return None
        vals = t.reshape(-1).tolist() if hasattr(t, "reshape") and hasattr(t, "tolist") \
            else t.tolist() if hasattr(t, "tolist") else t
    except Exception:  # noqa: BLE001
        return None
    out: list[Any] = []
    stack = [vals]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
        else:
            out.append(v)
        if len(out) > limit:
            return None
    return out


def _rows(t: Any) -> tuple[list[int], list[int]] | None:
    """edge_index [2, E] -> (src, dst) lists."""
    sh = _shape(t)
    if not sh or len(sh) != 2 or sh[0] != 2 or sh[1] > MAX_VALUES:
        return None
    try:
        rows = t.tolist() if hasattr(t, "tolist") else t
        return [int(x) for x in rows[0]], [int(x) for x in rows[1]]
    except Exception:  # noqa: BLE001
        return None


def _dist(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "mean": 0, "max": 0, "total": 0}
    tot = sum(values)
    return {"min": min(values), "mean": round(tot / len(values), 2), "max": max(values), "total": tot}


def _is_pyg(obj: Any) -> bool:
    return _get(obj, "edge_index") is not None or (_get(obj, "x") is not None and _get(obj, "num_nodes") is not None)


def _is_dgl(obj: Any) -> bool:
    return callable(getattr(obj, "num_nodes", None)) and callable(getattr(obj, "num_edges", None)) \
        and hasattr(obj, "ndata")


def _feat_dim(t: Any) -> int | None:
    sh = _shape(t)
    if not sh:
        return None
    return sh[1] if len(sh) >= 2 else 1


def _dgl_feat(store: Any) -> tuple[str | None, int | None]:
    try:
        keys = list(store.keys())
    except Exception:  # noqa: BLE001
        return None, None
    for k in ("feat", "x", "h", "attr", "node_attr", "edge_attr"):
        if k in keys:
            return k, _feat_dim(store[k])
    return (keys[0], _feat_dim(store[keys[0]])) if keys else (None, None)


# ----------------------------------------------------------------- labels

def _labels(ys: list[Any], shape: tuple[int, ...] | None) -> dict[str, Any] | None:
    vals = [v for v in ys if isinstance(v, (int, float, bool)) and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    out: dict[str, Any] = {"shape": list(shape) if shape else [len(ys)]}
    width = shape[1] if shape and len(shape) == 2 else 1
    if all(float(v).is_integer() for v in vals) and len({int(v) for v in vals}) <= max(50, int(len(vals) ** 0.5)):
        counts: dict[int, int] = {}
        for v in vals:
            counts[int(v)] = counts.get(int(v), 0) + 1
        out["task"] = "multi-label" if width > 1 else "classification"
        out["num_classes"] = width if width > 1 else len(counts)
        out["class_counts"] = {str(k): counts[k] for k in sorted(counts)[:20]}
    else:
        n = len(vals)
        mean = sum(vals) / n
        out.update(task="regression", targets=width, min=round(min(vals), 4), max=round(max(vals), 4),
                   mean=round(mean, 4), std=round((sum((v - mean) ** 2 for v in vals) / n) ** 0.5, 4))
    if len(vals) < len(ys):
        out["missing"] = len(ys) - len(vals)
    return out


# ----------------------------------------------------------------- summaries

def _graph_list(graphs: list[Any], total: int | None = None, labels: Any = None) -> dict[str, Any]:
    nodes, edges, ys, ylen = [], [], [], None
    nfeat = efeat = None
    self_loops = isolated = 0
    checked = True
    fmt = "pyg"
    for g in graphs:
        if _is_dgl(g):
            fmt = "dgl"
            n, e = int(g.num_nodes()), int(g.num_edges())
            if nfeat is None:
                nfeat = _dgl_feat(g.ndata)[1]
                efeat = _dgl_feat(g.edata)[1] if hasattr(g, "edata") else None
            src_dst = None
            try:
                s, d = g.edges()
                src_dst = ([int(x) for x in s.tolist()], [int(x) for x in d.tolist()])
            except Exception:  # noqa: BLE001
                pass
        else:
            ei = _get(g, "edge_index")
            x = _get(g, "x")
            n = _get(g, "num_nodes")
            n = int(n) if isinstance(n, (int, float)) else (_shape(x) or (0,))[0]
            e = (_shape(ei) or (0, 0))[1] if ei is not None else 0
            if nfeat is None:
                nfeat = _feat_dim(x)
                efeat = _feat_dim(_get(g, "edge_attr"))
            src_dst = _rows(ei) if ei is not None else ([], [])
            y = _get(g, "y")
            if y is not None:
                fy = _flat(y, 10_000)
                if fy is not None:
                    ys.extend(fy)
                    ylen = len(fy)
        nodes.append(n)
        edges.append(e)
        if src_dst is None:
            checked = False
        elif checked:
            s, d = src_dst
            self_loops += sum(1 for a, b in zip(s, d) if a == b)
            seen = set(s) | set(d)
            isolated += sum(1 for i in range(n) if i not in seen)
    out: dict[str, Any] = {"format": fmt, "num_graphs": total if total is not None else len(graphs),
                           "nodes": _dist(nodes), "edges": _dist(edges), "node_features": nfeat,
                           "edge_features": efeat}
    if total is not None and total > len(graphs):
        out["sampled"] = len(graphs)
    if checked:
        out["self_loops"] = self_loops
        out["isolated_nodes"] = isolated
    if nodes and sum(nodes):
        out["avg_degree"] = round(sum(edges) / sum(nodes), 2)
    if labels is not None and not ys:
        for k, v in (labels.items() if isinstance(labels, dict) else [("y", labels)]):
            fy = _flat(v)
            if fy:
                out["labels"] = _labels(fy, _shape(v))
                out["labels"]["key"] = k
                break
    elif ys:
        out["labels"] = _labels(ys, (len(graphs), ylen) if ylen and ylen > 1 else None)
    return out


def _collated(data: Any, slices: dict[str, Any]) -> dict[str, Any]:
    """PyG InMemoryDataset storage: one big graph + per-attribute offsets (``torch.save((data, slices))``)."""
    def offsets(key: str) -> list[int] | None:
        v = slices.get(key)
        fl = _flat(v)
        return [int(x) for x in fl] if fl else None

    any_off = next((offsets(k) for k in slices if offsets(k)), None)
    if not any_off:
        raise ValueError("empty slices")
    num_graphs = len(any_off) - 1
    x = _get(data, "x")
    off_x = offsets("x")
    if off_x:
        nodes = [b - a for a, b in zip(off_x, off_x[1:])]
    else:
        nn = _get(data, "num_nodes")
        nodes = [int(v) for v in _flat(nn) or []] if nn is not None and not isinstance(nn, int) else []
    off_e = offsets("edge_index")
    ei = _get(data, "edge_index")
    edges = [b - a for a, b in zip(off_e, off_e[1:])] if off_e else []
    out: dict[str, Any] = {"format": "pyg-collated", "num_graphs": num_graphs, "nodes": _dist(nodes),
                           "edges": _dist(edges), "node_features": _feat_dim(x),
                           "edge_features": _feat_dim(_get(data, "edge_attr"))}
    rows = _rows(ei) if ei is not None else None
    if rows is not None:
        s, d = rows
        out["self_loops"] = sum(1 for a, b in zip(s, d) if a == b)
        if off_e and nodes and len(nodes) == num_graphs:
            seen: set[int] = set()
            base = 0
            gi = 0
            node_base = [0]
            for n in nodes:
                node_base.append(node_base[-1] + n)
            for j, (a, b) in enumerate(zip(s, d)):
                while gi < num_graphs and j >= off_e[gi + 1]:
                    gi += 1
                base = node_base[gi]
                seen.add(base + a)
                seen.add(base + b)
            out["isolated_nodes"] = node_base[-1] - len(seen)
    if nodes and sum(nodes):
        out["avg_degree"] = round(sum(edges) / sum(nodes), 2)
    y = _get(data, "y")
    fy = _flat(y)
    if fy:
        sh = _shape(y)
        out["labels"] = _labels(fy, sh if sh and len(sh) == 2 else None)
    return out


def _single(g: Any) -> dict[str, Any]:
    out = _graph_list([g])
    out["level"] = "node"  # one big graph: node-level task (Cora-style) or link prediction
    for name in ("train_mask", "val_mask", "test_mask"):
        m = _get(g, name) if not _is_dgl(g) else (g.ndata.get(name) if hasattr(g.ndata, "get") else None)
        fm = _flat(m)
        if fm:
            out.setdefault("masks", {})[name] = int(sum(1 for v in fm if v))
    y = _get(g, "y") if not _is_dgl(g) else (g.ndata.get("label") if hasattr(g.ndata, "get") else None)
    fy = _flat(y)
    if fy:
        out["labels"] = _labels(fy, None)
    return out


def _state_dict(sd: dict[str, Any]) -> dict[str, Any]:
    total = 0
    layers: dict[str, int] = {}
    for k, v in sd.items():
        sh = _shape(v)
        n = 1
        for d in sh or ():
            n *= d
        total += n
        prefix = k.split(".")[0]
        layers[prefix] = layers.get(prefix, 0) + n
    return {"format": "state_dict", "tensors": len(sd), "parameters": total,
            "by_module": dict(sorted(layers.items(), key=lambda kv: -kv[1])[:12])}


def summarize(obj: Any) -> dict[str, Any]:
    """Stats for whatever a graph-dataset file contained. Raises ValueError when it is not graph data."""
    # checkpoint: {"model_state_dict": {...}} or a plain state dict
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if isinstance(obj.get(k), dict):
                return _state_dict(obj[k])
        if obj and all(isinstance(k, str) for k in obj) and all(_shape(v) is not None and hasattr(v, "dtype")
                                                                 for v in obj.values()):
            if not {"x", "edge_index"} & set(obj):
                return _state_dict(obj)
    # PyG collated (data, slices[, sizes, data_cls]) tuple
    if isinstance(obj, tuple) and len(obj) >= 2 and isinstance(obj[1], dict) and not isinstance(obj[0], list):
        if obj[1] and (_is_pyg(obj[0]) or (isinstance(obj[0], dict) and "x" in obj[0])):
            return _collated(obj[0], obj[1])
    # DGL load_graphs: (graphs, labels)
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], list) and isinstance(obj[1], dict):
        return _graph_list(obj[0][:MAX_GRAPHS], total=len(obj[0]), labels=obj[1] or None)
    # a pickled InMemoryDataset
    data, slices = _get(obj, "_data") or _get(obj, "data"), _get(obj, "slices")
    if data is not None and isinstance(slices, dict) and slices:
        return _collated(data, slices)
    if _is_pyg(obj) or _is_dgl(obj):
        return _single(obj)
    if isinstance(obj, (list, tuple)) or (hasattr(obj, "__len__") and hasattr(obj, "__getitem__")
                                          and not isinstance(obj, (dict, str, bytes))):
        try:
            n = len(obj)
            graphs = [obj[i] for i in range(min(n, MAX_GRAPHS))]
        except Exception:  # noqa: BLE001 - not an indexable dataset
            graphs, n = [], 0
        if graphs and all(_is_pyg(g) or _is_dgl(g) for g in graphs[:20]):
            return _graph_list(graphs, total=n)
    raise ValueError(f"not recognised as graph data ({type(obj).__name__})")


# ----------------------------------------------------------------- loading

def load(path: str) -> tuple[Any, str]:
    p = Path(path)
    if p.suffix in (".bin", ".dgl"):
        import dgl  # noqa: PLC0415

        return dgl.load_graphs(str(p)), "dgl.load_graphs"
    import torch  # noqa: PLC0415

    try:
        return torch.load(str(p), map_location="cpu", weights_only=True), "torch.load(weights_only=True)"
    except Exception:  # noqa: BLE001 - PyG objects need a full unpickle
        try:
            return torch.load(str(p), map_location="cpu", weights_only=False), "torch.load(weights_only=False)"
        except TypeError:  # torch < 1.13
            return torch.load(str(p), map_location="cpu"), "torch.load"


def main(argv: list[str]) -> int:
    paths = json.loads(argv[1]) if len(argv) > 1 else []
    try:  # torch_geometric registers its classes for unpickling when imported
        import torch_geometric  # noqa: F401, PLC0415
    except Exception:  # noqa: BLE001
        pass
    result: dict[str, Any] = {"python": sys.version.split()[0], "files": {}}
    for path in paths:
        try:
            obj, how = load(path)
            stats = summarize(obj)
            stats["loaded_with"] = how
        except Exception as exc:  # noqa: BLE001
            stats = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        result["files"][path] = stats
    for mod in ("torch", "torch_geometric", "dgl"):
        m = sys.modules.get(mod)
        if m is not None:
            result.setdefault("versions", {})[mod] = getattr(m, "__version__", "?")
    sys.stdout.write("\n" + MARK + json.dumps(result, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
