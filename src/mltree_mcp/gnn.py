"""GNN views: a static architecture summary of a step's models, and graph-dataset stats.

* Architecture: the step's script (and the project-local modules it imports) is parsed with :mod:`ast`; nothing is
  imported or run. It finds ``nn.Module`` subclasses and the layers their ``__init__`` assigns (GCNConv, GATConv,
  GINConv, SAGEConv, DGL GraphConv, ...), resolves layer sizes from defaults, module constants, the arguments the
  script instantiates the model with and the step's params, and reads the order of calls in ``forward``.
  Ready-made PyG models (``GCN(...)``, ``GIN(...)``, ``AttentiveFP(...)``) and dataset classes (``TUDataset``,
  ``MoleculeNet``, ``QM9``, OGB, DGL datasets) are detected too.
* Dataset stats: only on explicit request (:func:`introspect`). It runs mltree's standalone ``_gnn_probe.py`` with
  the project's interpreter, which loads the step's ``.pt`` / DGL ``.bin`` files. Unpickling a ``.pt`` file can run
  code stored in it, so callers must make that an explicit user action. Results are cached in the store
  (``gnn_stats``) and show in the viewer and the HTML export.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .store import Store

from .store import MLTreeError

CONV = {
    # PyG
    "GCNConv", "GATConv", "GATv2Conv", "GINConv", "GINEConv", "SAGEConv", "GraphConv", "ChebConv", "TransformerConv",
    "GatedGraphConv", "EdgeConv", "DynamicEdgeConv", "NNConv", "PNAConv", "GENConv", "ResGatedGraphConv", "ARMAConv",
    "SGConv", "APPNP", "TAGConv", "GCN2Conv", "RGCNConv", "FastRGCNConv", "GPSConv", "MFConv", "CGConv", "GMMConv",
    "SplineConv", "PointNetConv", "LEConv", "ClusterGCNConv", "FiLMConv", "SuperGATConv", "FAConv", "EGConv",
    "GeneralConv", "HypergraphConv", "HeteroConv", "HGTConv", "HANConv", "SSGConv", "PDNConv", "AntiSymmetricConv",
    "GravNetConv", "FeaStConv", "PPFConv", "XConv", "PointTransformerConv", "WLConv", "SimpleConv", "DirGNNConv",
    # DGL (GraphConv, GATConv, GINConv, SAGEConv, ChebConv, SGConv, TAGConv, GatedGraphConv, EdgeConv, NNConv above)
    "RelGraphConv", "EGATConv", "GATv2Conv", "DotGatConv", "APPNPConv", "GMMConv", "AGNNConv", "CFConv", "EGNNConv",
    "PNAConv", "TWIRLSConv", "GCN2Conv", "HGTConv", "GINEConv", "EdgeGATConv", "GraphormerLayer",
    # misc
    "MessagePassing",
}
POOL = {
    "global_mean_pool", "global_add_pool", "global_max_pool", "global_sort_pool", "GlobalAttention",
    "AttentionalAggregation", "Set2Set", "SortAggregation", "TopKPooling", "SAGPooling", "EdgePooling", "ASAPooling",
    "MeanAggregation", "SumAggregation", "MaxAggregation", "MultiAggregation", "SumPooling", "AvgPooling",
    "MaxPooling", "SortPooling", "GlobalAttentionPooling", "Set2SetPooling", "WeightAndSum", "mean_nodes",
    "sum_nodes", "max_nodes", "readout_nodes",
}
NORM = {"BatchNorm", "BatchNorm1d", "LayerNorm", "GraphNorm", "InstanceNorm", "PairNorm", "DiffGroupNorm",
        "MessageNorm", "GraphSizeNorm", "MeanSubtractionNorm"}
LINEAR = {"Linear", "LazyLinear", "Bilinear", "MLP"}
ACT = {"ReLU", "ELU", "LeakyReLU", "PReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "Softmax", "LogSoftmax", "Mish",
       "relu", "elu", "leaky_relu", "gelu", "silu", "tanh", "sigmoid", "softmax", "log_softmax", "mish"}
DROPOUT = {"Dropout", "dropout", "DropEdge", "dropout_edge", "dropout_adj"}
EMBED = {"Embedding", "AtomEncoder", "BondEncoder"}
CONTAINERS = {"ModuleList", "Sequential", "ModuleDict"}
PREBUILT = {  # torch_geometric.nn.models: positional argument names
    "GCN": ["in_channels", "hidden_channels", "num_layers", "out_channels"],
    "GraphSAGE": ["in_channels", "hidden_channels", "num_layers", "out_channels"],
    "GIN": ["in_channels", "hidden_channels", "num_layers", "out_channels"],
    "GAT": ["in_channels", "hidden_channels", "num_layers", "out_channels"],
    "PNA": ["in_channels", "hidden_channels", "num_layers", "out_channels"],
    "EdgeCNN": ["in_channels", "hidden_channels", "num_layers", "out_channels"],
    "AttentiveFP": ["in_channels", "hidden_channels", "out_channels", "edge_dim", "num_layers", "num_timesteps"],
    "SchNet": ["hidden_channels", "num_filters", "num_interactions"],
    "DimeNet": ["hidden_channels", "out_channels", "num_blocks"],
    "DimeNetPlusPlus": ["hidden_channels", "out_channels", "num_blocks"],
    "GraphUNet": ["in_channels", "hidden_channels", "out_channels", "depth"],
}
DATASETS = {
    "TUDataset", "Planetoid", "MoleculeNet", "QM9", "QM7b", "ZINC", "PPI", "Reddit", "Reddit2", "Amazon", "Coauthor",
    "CitationFull", "Flickr", "Yelp", "WikiCS", "GNNBenchmarkDataset", "LRGBDataset", "MD17", "PCQM4Mv2",
    "AQSOL", "OGB_MAG", "FakeDataset", "KarateClub", "WebKB", "Actor", "HeterophilousGraphDataset",
    "PygGraphPropPredDataset", "PygNodePropPredDataset", "PygLinkPropPredDataset", "DglGraphPropPredDataset",
    "DglNodePropPredDataset", "DglLinkPropPredDataset",
    "CoraGraphDataset", "CiteseerGraphDataset", "PubmedGraphDataset", "QM9Dataset", "QM9EdgeDataset", "GINDataset",
    "LegacyTUDataset", "MiniGCDataset", "ZINCDataset", "PPIDataset", "RedditDataset", "AmazonCoBuyComputerDataset",
    "CoauthorCSDataset", "FraudDataset",
}
GRAPH_SUFFIXES = {".pt", ".pth", ".bin", ".dgl"}
SKIP_FILES = {"pre_filter.pt", "pre_transform.pt"}
MAX_FILES = 12
MAX_FILE_BYTES = 1_000_000
PROBE_TIMEOUT = 600


def _lib(full: str) -> str:
    if full.startswith(("torch_geometric", "pyg")):
        return "pyg"
    if full.startswith("dgl"):
        return "dgl"
    if full.startswith("ogb"):
        return "ogb"
    if full.startswith("torch"):
        return "torch"
    return ""


def family(name: str) -> str:
    if name in CONV:
        return "conv"
    if name in POOL:
        return "pool"
    if name in NORM:
        return "norm"
    if name in LINEAR:
        return "linear"
    if name in ACT:
        return "act"
    if name in DROPOUT:
        return "dropout"
    if name in EMBED:
        return "embed"
    if name in CONTAINERS:
        return "container"
    return "other"


# ----------------------------------------------------------------- expression helpers

_ARITH = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Pow, ast.Mod)


def _eval(node: ast.AST, env: dict[str, Any], depth: int = 0) -> Any:
    """Evaluate a tiny, safe subset (numbers, strings, names from ``env``, + - * // ...). None = unknown."""
    if depth > 20:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool)):
        return node.value
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Attribute):  # args.hidden / cfg.hidden / self.hidden -> the attribute name
        return env.get(ast.unparse(node), env.get(node.attr) if isinstance(node.value, ast.Name) else None)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval(node.operand, env, depth + 1)
        return -v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ARITH):
        a, b = _eval(node.left, env, depth + 1), _eval(node.right, env, depth + 1)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
            return None
        try:
            v = {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op))  # type: ignore[operator]
            if v is None:
                if isinstance(node.op, ast.FloorDiv):
                    v = a // b
                elif isinstance(node.op, ast.Div):
                    v = a / b
                elif isinstance(node.op, ast.Mod):
                    v = a % b
                else:
                    v = a ** b if abs(b) < 64 else None
            return v
        except (ZeroDivisionError, OverflowError):
            return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("int", "len") and node.args:
        v = _eval(node.args[0], env, depth + 1)
        if node.func.id == "int" and isinstance(v, (int, float, str)):
            try:
                return int(v)
            except ValueError:
                return None
    return None


def _dim(node: ast.AST | None, env: dict[str, Any]) -> dict[str, Any] | None:
    if node is None:
        return None
    text = ast.unparse(node)
    if len(text) > 60:
        text = text[:57] + "..."
    v = _eval(node, env)
    out: dict[str, Any] = {"text": text}
    if v is not None and str(v) != text:
        out["value"] = v
    return out


def _dim_str(d: dict[str, Any] | None) -> str:
    if not d:
        return "?"
    return str(d["value"]) if "value" in d else d["text"]


def _consts(body: list[ast.stmt]) -> dict[str, Any]:
    env: dict[str, Any] = {}
    for st in body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = _eval(st.value, env)
            if v is not None:
                env[st.targets[0].id] = v
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            v = _eval(st.value, env)
            if v is not None:
                env[st.target.id] = v
    return env


def _argparse_defaults(tree: ast.AST) -> dict[str, Any]:
    """add_argument("--hidden", default=64) -> {"hidden": 64} (also ``args.hidden``)."""
    out: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flag = next((a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                         and a.value.startswith("--")), None)
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
            if flag and default is not None:
                v = _eval(default, {})
                if v is not None:
                    out[flag.lstrip("-").replace("-", "_")] = v
    return out


def _func_defaults(fn: ast.FunctionDef, consts: dict[str, Any] | None = None) -> dict[str, Any]:
    env: dict[str, Any] = {}
    pos = fn.args.posonlyargs + fn.args.args
    for a, d in zip(pos[len(pos) - len(fn.args.defaults):], fn.args.defaults):
        v = _eval(d, consts or {})
        if v is not None:
            env[a.arg] = v
    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if d is not None:
            v = _eval(d, consts or {})
            if v is not None:
                env[a.arg] = v
    return env


# ----------------------------------------------------------------- per file


class _File:
    def __init__(self, rel: str, tree: ast.Module) -> None:
        self.rel = rel
        self.tree = tree
        self.imports: dict[str, str] = {}   # local name -> dotted origin
        self.local_modules: list[str] = []  # dotted module names imported (maybe project-local)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[(a.asname or a.name).split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                    self.local_modules.append(a.name)
            elif isinstance(node, ast.ImportFrom):
                mod = ("." * node.level) + (node.module or "")
                for a in node.names:
                    self.imports[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
                    if node.level or node.module:
                        self.local_modules.append(mod)
                        self.local_modules.append(f"{mod}.{a.name}")
        self.consts = _consts(tree.body)

    def resolve(self, func: ast.AST) -> tuple[str, str]:
        """(short name, full dotted name) of a called object."""
        text = ast.unparse(func)
        head, _, rest = text.partition(".")
        full = self.imports.get(head, head) + ("." + rest if rest else "")
        return text.rsplit(".", 1)[-1], full


def _read_source(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        if path.suffix == ".ipynb":
            from .scan import _nb_code, _notebook_cells

            return _nb_code([c["source"] for c in _notebook_cells(path) if c["type"] == "code"])
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _gather(root: Path, script: Path) -> tuple[list[_File], list[str]]:
    """The script plus project-local modules it imports (2 levels deep, at most MAX_FILES files)."""
    files: list[_File] = []
    warnings: list[str] = []
    seen: set[Path] = set()
    queue: list[tuple[Path, int]] = [(script, 0)]
    root_r = root.resolve()
    while queue and len(files) < MAX_FILES:
        path, depth = queue.pop(0)
        rp = path.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        src = _read_source(path)
        if src is None:
            warnings.append(f"could not read {path.name}")
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            warnings.append(f"{path.name}: syntax error line {exc.lineno}")
            continue
        try:
            rel = rp.relative_to(root_r).as_posix()
        except ValueError:
            rel = str(path)
        f = _File(rel, tree)
        files.append(f)
        if depth >= 2:
            continue
        for mod in dict.fromkeys(f.local_modules):
            for cand in _module_paths(root_r, rp.parent, mod):
                if cand.resolve() not in seen:
                    queue.append((cand, depth + 1))
    return files, warnings


def _module_paths(root: Path, here: Path, mod: str) -> list[Path]:
    level = len(mod) - len(mod.lstrip("."))
    parts = [p for p in mod.lstrip(".").split(".") if p]
    if not parts:
        return []
    bases = [here] if level else [here, root, root / "src"]
    if level > 1:
        bases = [here.parents[level - 2]] if len(here.parents) >= level - 1 else []
    out = []
    for base in bases:
        for cand in (base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts) / "__init__.py"):
            try:
                cr = cand.resolve()
                if cand.is_file() and cr.is_relative_to(root) and ".venv" not in cr.parts and "site-packages" not in cr.parts:
                    out.append(cand)
                    break
            except (OSError, ValueError):
                continue
    return out[:1]


# ----------------------------------------------------------------- models


def _is_module_class(cls: ast.ClassDef) -> bool:
    for b in cls.bases:
        t = ast.unparse(b)
        last = t.rsplit(".", 1)[-1]
        if last in ("Module", "MessagePassing", "LightningModule", "BasicGNN") or t.endswith("nn.Module"):
            return True
    return False


def _layer(call: ast.Call, f: _File, env: dict[str, Any], attr: str, repeat: Any = 1) -> dict[str, Any]:
    name, full = f.resolve(call.func)
    fam = family(name)
    lay: dict[str, Any] = {"attr": attr, "type": name, "family": fam, "lib": _lib(full), "line": call.lineno,
                           "file": f.rel}
    if repeat != 1:
        lay["repeat"] = repeat
    args = [ast.unparse(a) for a in call.args]
    kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
    lay["call"] = f"{name}(" + ", ".join(args + [f"{k}={v}" for k, v in kwargs.items()]) + ")"
    if len(lay["call"]) > 120:
        lay["call"] = lay["call"][:117] + "..."
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    if fam in ("conv", "linear", "norm", "embed"):
        pos = list(call.args)
        if name in ("GINConv", "GINEConv", "EdgeConv", "NNConv") and (pos or "nn" in kw):
            inner = kw.get("nn") or pos[0]
            dims = _mlp_dims(inner, f, env)
            if dims:
                lay["in"], lay["out"] = dims
                lay["mlp"] = ast.unparse(inner)[:80]
        else:
            in_key = "in_channels" if "in_channels" in kw else ("in_feats" if "in_feats" in kw else "in_features")
            out_key = "out_channels" if "out_channels" in kw else ("out_feats" if "out_feats" in kw else "out_features")
            if name in ("BatchNorm", "BatchNorm1d", "LayerNorm", "GraphNorm", "InstanceNorm"):
                lay["in"] = _dim(kw.get("in_channels") or kw.get("num_features") or (pos[0] if pos else None), env)
            elif name in ("Embedding",):
                lay["in"] = _dim(kw.get("num_embeddings") or (pos[0] if pos else None), env)
                lay["out"] = _dim(kw.get("embedding_dim") or (pos[1] if len(pos) > 1 else None), env)
            elif name == "MLP" and (pos and isinstance(pos[0], ast.List) or "channel_list" in kw):
                lst = kw.get("channel_list") or pos[0]
                if isinstance(lst, ast.List) and lst.elts:
                    lay["in"], lay["out"] = _dim(lst.elts[0], env), _dim(lst.elts[-1], env)
            else:
                lay["in"] = _dim(kw.get(in_key) or (pos[0] if pos else None), env)
                lay["out"] = _dim(kw.get(out_key) or (pos[1] if len(pos) > 1 else None), env)
        for k in ("heads", "num_heads"):
            if k in kw:
                lay["heads"] = _dim(kw[k], env)
        if name in ("GATConv", "GATv2Conv") and "heads" not in lay and len(call.args) > 2:
            lay["heads"] = _dim(call.args[2], env)
    elif fam == "dropout":
        p = kw.get("p") or (call.args[0] if call.args else None)
        if p is not None:
            lay["p"] = _dim(p, env)
    return {k: v for k, v in lay.items() if v is not None}


def _mlp_dims(node: ast.AST, f: _File, env: dict[str, Any]) -> tuple[Any, Any] | None:
    lins = [n for n in ast.walk(node) if isinstance(n, ast.Call) and f.resolve(n.func)[0] in ("Linear", "MLP")]
    lins.sort(key=lambda n: (n.lineno, n.col_offset))
    if not lins:
        return None
    first, last = lins[0], lins[-1]
    if f.resolve(first.func)[0] == "MLP" and first.args and isinstance(first.args[0], ast.List):
        el = first.args[0].elts
        return _dim(el[0], env), _dim(el[-1], env)
    a = _dim(first.args[0], env) if first.args else None
    b = _dim(last.args[1], env) if len(last.args) > 1 else None
    return a, b


def _range_count(it: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and it.args:
        if len(it.args) == 1:
            v = _eval(it.args[0], env)
            return v if isinstance(v, int) else ast.unparse(it.args[0])
        a, b = _eval(it.args[0], env), _eval(it.args[1], env)
        if isinstance(a, int) and isinstance(b, int):
            return max(0, b - a)
        return f"{ast.unparse(it.args[1])} - {ast.unparse(it.args[0])}"
    return "n"


def _init_layers(fn: ast.FunctionDef, f: _File, env: dict[str, Any], classes: set[str]) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    local_calls: dict[str, ast.Call] = {}

    def known(call: ast.Call) -> bool:
        name = f.resolve(call.func)[0]
        return family(name) != "other" or name in classes

    def add_container(attr: str, call: ast.Call, repeat: Any) -> None:
        name = f.resolve(call.func)[0]
        items: list[tuple[ast.AST, Any]] = []
        for a in call.args:
            if isinstance(a, (ast.List, ast.Tuple)):
                items += [(e, repeat) for e in a.elts]
            elif isinstance(a, ast.ListComp) and a.generators:
                cnt = _range_count(a.generators[0].iter, env)
                items.append((a.elt, cnt))
            else:
                items.append((a, repeat))
        if not items:
            layers.append({"attr": attr, "type": name, "family": "container", "line": call.lineno, "file": f.rel})
        for i, (e, rep) in enumerate(items):
            if isinstance(e, ast.Call) and known(e):
                sub = f"{attr}[{'*' if rep != 1 else i}]"
                layers.append(_layer(e, f, env, sub, rep) if f.resolve(e.func)[0] not in CONTAINERS
                              else {"attr": sub, "type": f.resolve(e.func)[0], "family": "container",
                                    "line": e.lineno, "file": f.rel})

    def visit(stmts: list[ast.stmt], repeat: Any) -> None:
        for st in stmts:
            if isinstance(st, ast.For):
                visit(st.body, _range_count(st.iter, env) if repeat == 1 else "n")
                continue
            if isinstance(st, (ast.If, ast.With, ast.Try)):
                visit(st.body, repeat)
                visit(getattr(st, "orelse", []) or [], repeat)
                continue
            if isinstance(st, ast.Assign) and len(st.targets) == 1:
                tgt, val = st.targets[0], st.value
                if isinstance(tgt, ast.Name):
                    v = _eval(val, env)
                    if v is not None:
                        env[tgt.id] = v
                    if isinstance(val, ast.Call):
                        local_calls[tgt.id] = val
                    continue
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                    if isinstance(val, ast.Call):
                        name = f.resolve(val.func)[0]
                        if name in CONTAINERS:
                            add_container(tgt.attr, val, repeat)
                        elif known(val):
                            lay = _layer(val, f, env, tgt.attr, repeat)
                            if name in classes:
                                lay["family"] = "submodule"
                            layers.append(lay)
                    elif isinstance(val, ast.Name) and val.id in local_calls and known(local_calls[val.id]):
                        layers.append(_layer(local_calls[val.id], f, env, tgt.attr, repeat))
                    else:
                        v = _eval(val, env)
                        if v is not None:
                            env[f"self.{tgt.attr}"] = v
                    continue
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                c = st.value
                if isinstance(c.func, ast.Attribute) and c.func.attr in ("append", "extend") and \
                        isinstance(c.func.value, ast.Attribute) and c.args:
                    attr = c.func.value.attr
                    arg = c.args[0]
                    if isinstance(arg, ast.Name) and arg.id in local_calls:
                        arg = local_calls[arg.id]
                    if isinstance(arg, ast.Call) and known(arg):
                        layers.append(_layer(arg, f, env, f"{attr}[*]", repeat))

    visit(fn.body, 1)
    # an empty container (ModuleList() filled by .append later) is implied by its elements
    layers = [lay for lay in layers if not (lay["family"] == "container" and lay.get("type") in CONTAINERS
                                            and any(x["attr"].startswith(lay["attr"] + "[") for x in layers))]
    # one entry per container element pattern: merge repeated appends into the existing [*] entry
    merged: list[dict[str, Any]] = []
    for lay in layers:
        same = next((m for m in merged if m["attr"] == lay["attr"] and m["type"] == lay["type"]
                     and m.get("call") == lay.get("call")), None)
        if same is not None and lay["attr"].endswith("[*]"):
            r1, r2 = same.get("repeat", 1), lay.get("repeat", 1)
            same["repeat"] = r1 + r2 if isinstance(r1, int) and isinstance(r2, int) else f"{r1}+{r2}"
        else:
            merged.append(lay)
    return merged


_FN_TOKENS = {"relu": "relu", "elu": "elu", "leaky_relu": "leaky_relu", "gelu": "gelu", "silu": "silu",
              "tanh": "tanh", "sigmoid": "sigmoid", "softmax": "softmax", "log_softmax": "log_softmax",
              "dropout": "dropout", "dropout_edge": "dropout_edge", "cat": "cat"}


def _forward_flow(fn: ast.FunctionDef, f: _File, attrs: set[str]) -> list[str]:
    tokens: list[str] = []

    def call_token(c: ast.Call, loopvars: dict[str, str]) -> str | None:
        func = c.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            return func.attr if func.attr in attrs else None
        if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Attribute):
            return f"{func.value.attr}[i]"
        if isinstance(func, ast.Name) and func.id in loopvars:
            return loopvars[func.id]
        name = f.resolve(func)[0]
        if name in POOL:
            return name
        if name in _FN_TOKENS and not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                                       and func.value.id in ("np", "math")):
            return _FN_TOKENS[name]
        if isinstance(func, ast.Attribute) and func.attr in ("relu", "relu_", "sigmoid", "tanh"):
            return func.attr.rstrip("_")
        return None

    def calls_in(node: ast.AST, loopvars: dict[str, str]) -> list[str]:
        found: list[tuple[int, int, str]] = []
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                t = call_token(n, loopvars)
                if t:
                    # inner calls run first: order by end position
                    found.append((getattr(n, "end_lineno", n.lineno), getattr(n, "end_col_offset", 0), t))
        return [t for _, _, t in sorted(found)]

    def visit(stmts: list[ast.stmt], loopvars: dict[str, str], out: list[str]) -> None:
        for st in stmts:
            if isinstance(st, ast.For):
                lv = dict(loopvars)
                it = st.iter
                if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id in ("zip", "enumerate"):
                    srcs = [a for a in it.args if isinstance(a, ast.Attribute)]
                    names = [e.id for e in (st.target.elts if isinstance(st.target, ast.Tuple) else [st.target])
                             if isinstance(e, ast.Name)]
                    if it.func.id == "enumerate" and isinstance(st.target, ast.Tuple):
                        names = names[1:]
                    for nm, src in zip(names, srcs):
                        lv[nm] = f"{src.attr}[i]"
                elif isinstance(it, ast.Attribute) and isinstance(st.target, ast.Name):
                    lv[st.target.id] = f"{it.attr}[i]"
                body: list[str] = []
                visit(st.body, lv, body)
                if body:
                    out.append("(" + " → ".join(body) + ") × " + ("n" if not isinstance(it, ast.Attribute)
                                                                   else f"len({it.attr})"))
                continue
            if isinstance(st, (ast.If, ast.With, ast.Try)):
                visit(st.body, loopvars, out)
                continue
            for t in calls_in(st, loopvars):
                if not out or out[-1] != t:
                    out.append(t)

    visit(fn.body, {}, tokens)
    return tokens[:40]


def _class_models(f: _File, step_env: dict[str, Any], classes: set[str],
                  inst: dict[str, ast.Call], inst_file: dict[str, _File]) -> list[dict[str, Any]]:
    models = []
    for cls in [n for n in ast.walk(f.tree) if isinstance(n, ast.ClassDef)]:
        if not _is_module_class(cls):
            continue
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        fwd = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
        env = dict(f.consts)
        if init is not None:
            env.update(_func_defaults(init, f.consts))
            call = inst.get(cls.name)
            if call is not None:  # model = GCN(in_dim, hidden=args.hidden): bind the arguments
                cf = inst_file[cls.name]
                cenv = {**cf.consts, **step_env}
                names = [a.arg for a in init.args.args[1:]]
                for a, v in zip(names, call.args):
                    val = _eval(v, cenv)
                    if val is not None:
                        env[a] = val
                for k in call.keywords:
                    if k.arg:
                        val = _eval(k.value, cenv)
                        if val is not None:
                            env[k.arg] = val
            for k, v in step_env.items():  # a step param with the same name as an __init__ argument wins
                if init and any(a.arg == k for a in init.args.args + init.args.kwonlyargs):
                    env[k] = v
        layers = _init_layers(init, f, env, classes) if init is not None else []
        flow = _forward_flow(fwd, f, {lay["attr"].split("[")[0] for lay in layers}) if fwd is not None else []
        if fwd is not None:
            for t in flow:
                if t in POOL and not any(lay["type"] == t for lay in layers):
                    layers.append({"attr": "(forward)", "type": t, "family": "pool", "line": fwd.lineno,
                                   "file": f.rel})
        if not layers:
            continue
        m: dict[str, Any] = {"name": cls.name, "kind": "class", "file": f.rel, "line": cls.lineno,
                             "bases": [ast.unparse(b) for b in cls.bases], "layers": layers, "forward": flow}
        if cls.name in inst:
            m["instantiated"] = {"file": inst_file[cls.name].rel, "line": inst[cls.name].lineno,
                                 "call": ast.unparse(inst[cls.name])[:160]}
        models.append(m)
    return models


def _prebuilt(f: _File, env: dict[str, Any], classes: set[str]) -> list[dict[str, Any]]:
    out = []
    for n in ast.walk(f.tree):
        if not isinstance(n, ast.Call):
            continue
        name, full = f.resolve(n.func)
        if name not in PREBUILT or name in classes or _lib(full) != "pyg":
            continue
        cenv = {**f.consts, **env}
        argn = PREBUILT[name]
        dims: dict[str, Any] = {}
        for k, v in zip(argn, n.args):
            dims[k] = _dim(v, cenv)
        for k in n.keywords:
            if k.arg and (k.arg in argn or k.arg in ("heads", "dropout", "jk", "edge_dim", "num_timesteps")):
                dims[k.arg] = _dim(k.value, cenv)
        dims = {k: v for k, v in dims.items() if v}
        layer = {"attr": "(model)", "type": name, "family": "conv", "lib": "pyg", "line": n.lineno, "file": f.rel,
                 "call": ast.unparse(n)[:160]}
        if dims.get("in_channels"):
            layer["in"] = dims["in_channels"]
        if dims.get("out_channels") or dims.get("hidden_channels"):
            layer["out"] = dims.get("out_channels") or dims["hidden_channels"]
        if dims.get("num_layers"):
            nl = dims["num_layers"]
            layer["repeat"] = nl["value"] if nl.get("value") is not None else int(nl["text"]) if nl["text"].isdigit() \
                else nl["text"]
        if dims.get("heads"):
            layer["heads"] = dims["heads"]
        out.append({"name": name, "kind": "prebuilt", "lib": "pyg", "file": f.rel, "line": n.lineno,
                    "call": ast.unparse(n)[:160], "dims": dims, "layers": [layer]})
    return out


def summarize_model(m: dict[str, Any]) -> dict[str, Any]:
    """Counts and a one-line label, e.g. 'GCNConv ×3 · hidden 64 · global_mean_pool'."""
    if m["kind"] == "prebuilt":
        d = m["dims"]
        parts = [m["name"]]
        if d.get("num_layers"):
            parts.append(f"{_dim_str(d['num_layers'])} layers")
        if d.get("hidden_channels"):
            parts.append(f"hidden {_dim_str(d['hidden_channels'])}")
        if d.get("out_channels"):
            parts.append(f"out {_dim_str(d['out_channels'])}")
        hid = [_dim_str(d["hidden_channels"])] if d.get("hidden_channels") else []
        return {"label": " · ".join(parts), "conv": {m["name"]: _dim_str(d.get("num_layers")) if d.get("num_layers")
                                                     else 1}, "num_conv_layers": None, "hidden": hid,
                "heads": [_dim_str(d["heads"])] if d.get("heads") else [], "pooling": [], "head": None, "is_gnn": True}
    conv: dict[str, Any] = {}
    hidden: list[str] = []
    heads: list[str] = []
    for lay in m["layers"]:
        if lay["family"] != "conv":
            continue
        r = lay.get("repeat", 1)
        prev = conv.get(lay["type"], 0)
        conv[lay["type"]] = prev + r if isinstance(prev, int) and isinstance(r, int) else f"{prev}+{r}" if prev else r
        o = _dim_str(lay.get("out")) if lay.get("out") else None
        if o and o not in hidden:
            hidden.append(o)
        if lay.get("heads"):
            h = _dim_str(lay["heads"])
            if h not in heads:
                heads.append(h)
    pools = list(dict.fromkeys(lay["type"] for lay in m["layers"] if lay["family"] == "pool"))
    lins = [lay for lay in m["layers"] if lay["family"] == "linear"]
    head = None
    if lins:
        h = lins[-1]
        head = f"{h['type']}({_dim_str(h.get('in'))} → {_dim_str(h.get('out'))})"
    parts = [" + ".join(f"{t} ×{n}" if n != 1 else t for t, n in conv.items())] if conv else []
    if hidden:
        parts.append("hidden " + "/".join(hidden[:3]))
    if heads:
        parts.append("heads " + "/".join(heads))
    parts += pools[:2]
    total = sum(v for v in conv.values() if isinstance(v, int)) if all(isinstance(v, int) for v in conv.values()) \
        else None
    return {"label": " · ".join(parts) or m["name"], "conv": conv, "num_conv_layers": total, "hidden": hidden,
            "heads": heads, "pooling": pools, "head": head, "is_gnn": bool(conv or pools)}


def _datasets(f: _File, env: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    found, paths = [], []
    for n in ast.walk(f.tree):
        if not isinstance(n, ast.Call):
            continue
        name, full = f.resolve(n.func)
        if name in DATASETS:
            kw = {k.arg: k.value for k in n.keywords if k.arg}
            args = {k: _dim(v, {**f.consts, **env}) for k, v in kw.items()}
            for i, a in enumerate(n.args[:3]):
                args[f"arg{i}"] = _dim(a, {**f.consts, **env})
            d = {"class": name, "lib": _lib(full) or ("pyg" if name in DATASETS else ""), "file": f.rel,
                 "line": n.lineno, "call": ast.unparse(n)[:160]}
            root = kw.get("root") or (n.args[0] if n.args else None)
            dname = kw.get("name") or (n.args[1] if len(n.args) > 1 else None)
            for key, node in (("root", root), ("name", dname)):
                v = _eval(node, {**f.consts, **env}) if node is not None else None
                if isinstance(v, str):
                    d[key] = v
            found.append(d)
            if isinstance(d.get("root"), str):
                paths.append(d["root"] + ("/" + d["name"] if isinstance(d.get("name"), str) else ""))
                paths.append(d["root"])
        elif (name in ("load", "save") and full.startswith("torch")) or name in ("load_graphs", "save_graphs"):
            for a in n.args[:2]:
                v = _eval(a, f.consts)
                if isinstance(v, str) and Path(v).suffix in GRAPH_SUFFIXES:
                    paths.append(v)
    return found, paths


# ----------------------------------------------------------------- step level

_MEMO: dict[tuple, dict[str, Any]] = {}
_MEMO_LOCK = threading.Lock()


def analyze_step(store: "Store", step: str | dict[str, Any]) -> dict[str, Any]:
    """Static GNN summary of one step (never imports or runs project code)."""
    st = store.get_step(step) if isinstance(step, str) else step
    out: dict[str, Any] = {"step": st["name"], "models": [], "datasets": [], "files": [], "warnings": [],
                           "data_paths": []}
    sp = st.get("script_path")
    if not sp:
        return out
    script = store.root / sp
    if not script.is_file():
        out["warnings"].append(f"{sp} not found")
        return out
    try:
        stat = script.stat()
        key = (str(script), stat.st_size, stat.st_mtime_ns, json.dumps(st.get("params") or {}, sort_keys=True,
                                                                        default=str))
    except OSError:
        return out
    with _MEMO_LOCK:
        hit = _MEMO.get(key)
    if hit is not None and all(_fresh(p, s) for p, s in hit["_stamps"].items()):
        return {k: v for k, v in hit.items() if k != "_stamps"}
    files, warnings = _gather(store.root, script)
    env: dict[str, Any] = {}
    for f in files[:1]:
        env.update(_argparse_defaults(f.tree))
    for k, v in (st.get("params") or {}).items():
        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
            env[k] = v
            env[f"args.{k}"] = v
    classes = {c.name for f in files for c in ast.walk(f.tree) if isinstance(c, ast.ClassDef)}
    inst: dict[str, ast.Call] = {}
    inst_file: dict[str, _File] = {}
    for f in files:
        for n in ast.walk(f.tree):
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute)):
                nm = f.resolve(n.func)[0]
                if nm in classes and nm not in inst and (n.args or n.keywords):
                    inst[nm], inst_file[nm] = n, f
    models, datasets, paths = [], [], []
    for f in files:
        models += _class_models(f, env, classes, inst, inst_file)
        models += _prebuilt(f, env, classes)
        d, p = _datasets(f, env)
        datasets += d
        paths += p
    # a class used only as a submodule of another model is listed after it
    for m in models:
        m["summary"] = summarize_model(m)
    models.sort(key=lambda m: (not m["summary"]["is_gnn"], m["name"] not in inst, m["file"] != files[0].rel
                               if files else False))
    out.update(models=models[:8], datasets=datasets[:8], files=[f.rel for f in files], warnings=warnings,
               data_paths=list(dict.fromkeys(paths))[:12])
    stamps = {}
    for f in files:
        try:
            s = (store.root / f.rel).stat()
            stamps[str(store.root / f.rel)] = (s.st_size, s.st_mtime_ns)
        except OSError:
            pass
    with _MEMO_LOCK:
        if len(_MEMO) > 256:
            _MEMO.clear()
        _MEMO[key] = {**out, "_stamps": stamps}
    return out


def _fresh(path: str, stamp: tuple[int, int]) -> bool:
    try:
        s = os.stat(path)
        return (s.st_size, s.st_mtime_ns) == tuple(stamp)
    except OSError:
        return False


def badge(analysis: dict[str, Any]) -> dict[str, Any] | None:
    """Compact form for the map node: {'label': 'GCNConv ×3 · hidden 64', 'datasets': ['TUDataset']}."""
    gnn = [m for m in analysis.get("models", []) if m["summary"]["is_gnn"]]
    ds = list(dict.fromkeys(d["class"] + (f"({d['name']})" if d.get("name") else "") for d in
                            analysis.get("datasets", [])))
    if not gnn and not ds:
        return None
    out: dict[str, Any] = {}
    if gnn:
        out["label"] = gnn[0]["summary"]["label"]
        out["model"] = gnn[0]["name"]
    if ds:
        out["datasets"] = ds[:3]
    return out


def candidate_files(store: "Store", st: dict[str, Any], analysis: dict[str, Any]) -> list[Path]:
    """Data files of a step that may hold graphs: .pt/.pth/.bin/.dgl outputs and inputs, files found statically
    (torch.load/save literals, dataset roots), and the ``processed/`` files inside directories among them."""
    from .dataver import DataIndex

    idx = DataIndex(store)
    names = list(st.get("outputs") or []) + list(st.get("inputs") or []) + list(analysis.get("data_paths") or [])
    found: list[Path] = []
    root = store.root.resolve()
    for name in dict.fromkeys(names):
        p, _ = idx.resolve(name, st.get("script_path"))
        if p is None:
            continue
        try:
            p = p.resolve()
            if not p.is_relative_to(root):
                continue
        except (OSError, ValueError):
            continue
        if p.is_file() and p.suffix in GRAPH_SUFFIXES and p.name not in SKIP_FILES:
            found.append(p)
        elif p.is_dir():
            n = 0
            for dirpath, dirnames, filenames in os.walk(p):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "raw")
                for fn in sorted(filenames):
                    q = Path(dirpath) / fn
                    if q.suffix in GRAPH_SUFFIXES and fn not in SKIP_FILES:
                        found.append(q)
                        n += 1
                if n >= 4 or len(Path(dirpath).relative_to(p).parts) >= 3:
                    dirnames[:] = []
    out: list[Path] = []
    for p in found:
        if p not in out:
            out.append(p)
    return out[:MAX_FILES]


def step_view(store: "Store", step: str, run_allowed: bool = False) -> dict[str, Any]:
    """Everything the viewer's GNN section shows for a step: static analysis + cached dataset stats."""
    from .dataver import DataIndex

    st = store.get_step(step)
    a = analyze_step(store, st)
    files = candidate_files(store, st, a)
    cache = store.gnn_stats()
    idx = DataIndex(store)
    rows = []
    for p in files:
        rel = idx.rel(p)
        fp = idx.fingerprint(rel)
        c = cache.get(rel)
        row: dict[str, Any] = {"path": rel, "bytes": fp.get("bytes"), "version": fp.get("version")}
        if c:
            row.update(stats=c["stats"], computed_at=c["computed_at"], python=c["python"],
                       stale=bool(c["sha256"] and fp.get("sha256") and c["sha256"] != fp["sha256"]))
        rows.append(row)
    # stats cached for files that are no longer candidates (e.g. renamed) are still shown, marked missing
    for rel, c in cache.items():
        if c["step"] == st["name"] and rel not in {r["path"] for r in rows}:
            rows.append({"path": rel, "stats": c["stats"], "computed_at": c["computed_at"], "python": c["python"],
                         "missing": not (store.root / rel).exists()})
    idx.flush()
    a = {k: v for k, v in a.items() if k != "data_paths"}
    a["dataset_files"] = rows
    a["can_introspect"] = bool(run_allowed and files)
    a["introspect_note"] = INTROSPECT_NOTE
    return a


INTROSPECT_NOTE = ("Inspecting loads these files with torch / dgl in a subprocess that uses the project's Python. "
                   "Unpickling a .pt file can run code stored in it: only do this for data you trust. "
                   "mltree tries torch.load(weights_only=True) first.")


def introspect(store: "Store", step: str, python: str | None = None, timeout: float = PROBE_TIMEOUT,
               paths: list[str] | None = None) -> dict[str, Any]:
    """Load the step's graph data files in a subprocess (project interpreter) and cache their stats.

    Only call this on an explicit user request: it executes torch.load / dgl.load_graphs on project files."""
    from .dataver import DataIndex
    from .runner import find_python

    st = store.get_step(step)
    a = analyze_step(store, st)
    files = candidate_files(store, st, a)
    if paths:
        wanted = {str((store.root / p).resolve()) for p in paths}
        files = [p for p in files if str(p) in wanted]
    if not files:
        raise MLTreeError(f"Step {st['name']!r} has no .pt / .pth / .bin graph-data files among its inputs, outputs "
                          "or the paths its script loads; add them as outputs with add_step, or run the step first.")
    py = python or find_python(store.root)
    probe = Path(__file__).with_name("_gnn_probe.py")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLTREE_METRICS")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run([py, str(probe), json.dumps([str(p) for p in files])], cwd=store.root, env=env,
                              capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MLTreeError(f"Dataset introspection timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise MLTreeError(f"Could not start {py}: {exc}") from exc
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("MLTREE_GNN_STATS ")), None)
    if line is None:
        err = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        raise MLTreeError(f"Introspection failed (exit {proc.returncode}) with {py}: " + " | ".join(err))
    res = json.loads(line[len("MLTREE_GNN_STATS "):])
    idx = DataIndex(store)
    entries: dict[str, tuple[str | None, dict[str, Any]]] = {}
    for path, stats in res.get("files", {}).items():
        rel = idx.rel(Path(path))
        stats["versions"] = res.get("versions", {})
        entries[rel] = (idx.fingerprint(rel).get("sha256"), stats)
    idx.flush()
    store.put_gnn_stats(st["name"], entries, f"{py} (Python {res.get('python', '?')})")
    return {"step": st["name"], "python": py, "versions": res.get("versions", {}),
            "files": {rel: stats for rel, (_, stats) in entries.items()}}


def summary(store: "Store", step: str | None = None) -> dict[str, Any]:
    """MCP view: one step in full, or every step that has GNN models, datasets or cached stats."""
    if step:
        return step_view(store, step)
    out = {}
    cache_steps = {c["step"] for c in store.gnn_stats().values()}
    for st in store.steps_raw():
        a = analyze_step(store, st)
        if badge(a) or st["name"] in cache_steps:
            out[st["name"]] = step_view(store, st["name"])
    return {"steps": out}


__all__ = ["analyze_step", "badge", "candidate_files", "introspect", "step_view", "summary", "summarize_model",
           "family", "INTROSPECT_NOTE"]
