"""Static scanner: turn an existing ML project into an mltree map.

Nothing here imports or executes user code. Scripts are parsed with :mod:`ast`,
notebooks are read as JSON, and existing results (metrics files, MLflow
``mlruns/``, Optuna SQLite studies, Lightning CSV logs, wandb offline runs) are
read as plain files / read-only SQLite.

Entry point: :func:`scan_project`.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import posixpath
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .store import STEP_NAME_RE, MLTreeError, Store

DEFAULT_INCLUDE = ["**/*.py", "**/*.ipynb"]
DEFAULT_EXCLUDE = [
    "**/.venv/**", "**/venv/**", "**/env/**", "**/.env/**", "**/site-packages/**", "**/.git/**",
    "**/tests/**", "**/test/**", "**/test_*.py", "**/*_test.py", "**/conftest.py", "**/__pycache__/**",
    "**/.mltree/**", "**/node_modules/**", "**/.ipynb_checkpoints/**", "**/.tox/**", "**/build/**",
    "**/dist/**", "**/*.egg-info/**", "**/setup.py", "**/wandb/**", "**/mlruns/**", "**/lightning_logs/**",
]
# directories never descended into (fast pruning; patterns above still apply to files)
PRUNE_DIRS = {".git", ".venv", "venv", "env", ".env", "__pycache__", ".mltree", "node_modules", ".ipynb_checkpoints",
              ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages", ".idea", ".vscode"}
MAX_FILES = 3000
MAX_PY_BYTES = 2_000_000
MAX_NB_BYTES = 30_000_000
MAX_IMPORTED_RUNS = 1000

DATA_EXTS = {
    ".csv", ".tsv", ".parquet", ".pq", ".feather", ".arrow", ".json", ".jsonl", ".xlsx", ".xls", ".h5", ".hdf5",
    ".hdf", ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx",
    ".sdf", ".smi", ".smiles", ".mol", ".mol2", ".pdb", ".pdbqt", ".xyz", ".db", ".sqlite", ".txt", ".yaml", ".yml",
    ".png", ".svg", ".pdf", ".model", ".zip", ".gz", ".tar", ".lmdb", ".graphml", ".gml",
}
WEAK_EXTS = {".txt", ".yaml", ".yml", ".png", ".svg", ".pdf", ".json", ".gz", ".zip", ".tar", ".bin", ".model"}

KINDS = ("data", "clean", "split", "featurize", "hpo", "train", "eval", "util")

NAME_HINTS: list[tuple[str, str]] = [
    (r"util|helper|common|const|setting|config|__init__", "util"),
    (r"hpo|tun(e|ing)|optuna|sweep|search|hyperparam", "hpo"),
    (r"split|fold|partition", "split"),
    (r"feat|descriptor|fingerprint|graph|embed|encod|vectori", "featurize"),
    (r"clean|preproc|prep\b|prepare|filter|curat|standardi|sanit", "clean"),
    (r"eval|test|score|metric|predict|infer|benchmark|valid", "eval"),
    (r"train|fit|finetune|learn", "train"),
    (r"load|ingest|download|fetch|raw|collect|get_data|data|extract", "data"),
]
CALL_HINTS: dict[str, tuple[str, int]] = {
    "create_study": ("hpo", 6), "optimize": ("hpo", 2), "GridSearchCV": ("hpo", 5), "RandomizedSearchCV": ("hpo", 5),
    "BayesSearchCV": ("hpo", 5), "fmin": ("hpo", 4), "suggest_float": ("hpo", 3), "suggest_int": ("hpo", 3),
    "train_test_split": ("split", 5), "KFold": ("split", 4), "StratifiedKFold": ("split", 4), "GroupKFold": ("split", 4),
    "ShuffleSplit": ("split", 4), "scaffold_split": ("split", 5), "random_split": ("split", 4),
    "ScaffoldSplitter": ("split", 5), "RandomSplitter": ("split", 4),
    "fit": ("train", 3), "backward": ("train", 3), "train_on_batch": ("train", 3), "zero_grad": ("train", 2),
    "mean_squared_error": ("eval", 2), "mean_absolute_error": ("eval", 2), "r2_score": ("eval", 2),
    "roc_auc_score": ("eval", 2), "accuracy_score": ("eval", 2), "f1_score": ("eval", 2), "precision_score": ("eval", 2),
    "recall_score": ("eval", 2), "classification_report": ("eval", 2), "confusion_matrix": ("eval", 2),
    "matthews_corrcoef": ("eval", 2), "average_precision_score": ("eval", 2), "log_loss": ("eval", 2),
    "balanced_accuracy_score": ("eval", 2), "evaluate": ("eval", 1), "predict": ("eval", 1),
    "dropna": ("clean", 2), "drop_duplicates": ("clean", 2), "fillna": ("clean", 2), "Uncharger": ("clean", 2),
    "SaltRemover": ("clean", 3), "StripMol": ("clean", 2), "Standardizer": ("clean", 2), "Cleanup": ("clean", 2),
    "GetMorganFingerprintAsBitVect": ("featurize", 3), "GetMorganGenerator": ("featurize", 3),
    "GetMorganFingerprint": ("featurize", 3), "CalcMolDescriptors": ("featurize", 3), "MACCSkeys": ("featurize", 2),
    "from_smiles": ("featurize", 3), "smiles_to_graph": ("featurize", 3), "CircularFingerprint": ("featurize", 3),
    "MolGraphConvFeaturizer": ("featurize", 3), "RDKitDescriptors": ("featurize", 3), "OneHotEncoder": ("featurize", 1),
    "TfidfVectorizer": ("featurize", 2), "CountVectorizer": ("featurize", 2), "fit_transform": ("featurize", 1),
    "urlretrieve": ("data", 3), "download_url": ("data", 3), "read_csv": ("data", 1), "read_parquet": ("data", 1),
}
IMPORT_HINTS: list[tuple[str, str, int]] = [
    ("optuna", "hpo", 4), ("hyperopt", "hpo", 4), ("ray.tune", "hpo", 4), ("skopt", "hpo", 3),
    ("sklearn.metrics", "eval", 1), ("torchmetrics", "eval", 1),
    ("sklearn.model_selection", "split", 1), ("rdkit.Chem.AllChem", "featurize", 1),
    ("rdkit.Chem.rdFingerprintGenerator", "featurize", 2), ("rdkit.Chem.Descriptors", "featurize", 1),
    ("rdkit.Chem.MolStandardize", "clean", 2), ("deepchem.feat", "featurize", 2),
    ("pytorch_lightning", "train", 1), ("lightning", "train", 1),
]
# params worth recording when passed as literal keyword arguments anywhere
KW_PARAMS = {
    "test_size", "train_size", "val_size", "valid_size", "random_state", "seed", "n_splits", "n_trials", "n_estimators",
    "max_depth", "learning_rate", "lr", "epochs", "max_epochs", "num_epochs", "batch_size", "hidden_channels",
    "hidden_dim", "num_layers", "n_layers", "dropout", "weight_decay", "radius", "nBits", "n_bits", "fpSize",
    "patience", "num_heads", "heads",
}

READ_FIRST = {"read_csv", "read_table", "read_parquet", "read_excel", "read_json", "read_pickle", "read_feather",
              "read_hdf", "read_orc", "read_fwf", "loadtxt", "genfromtxt", "load_npz", "LoadSDF",
              "SDMolSupplier", "ForwardSDMolSupplier", "SmilesMolSupplier", "MolFromMolFile", "MolFromPDBFile",
              "MolFromMol2File", "MaeMolSupplier", "load_from_checkpoint", "load_model", "imread",
              "load_from_disk", "safe_open", "load_file"}
LOAD_MODULES = {"np", "numpy", "torch", "joblib", "dill", "cloudpickle", "sp", "sparse", "scipy", "jnp"}
WRITE_FIRST = {"to_csv", "to_parquet", "to_json", "to_pickle", "to_excel", "to_feather", "to_hdf", "to_orc",
               "savefig", "save_pretrained", "savetxt", "savez", "savez_compressed", "save_npz", "SDWriter",
               "SmilesWriter", "ExcelWriter", "save_model", "write_image", "write_html", "save_to_disk"}
WRITE_SECOND = {"MolToMolFile", "WriteSDF", "MolToPDBFile", "imwrite", "save_file"}
PYG_DATASETS = {"MoleculeNet", "TUDataset", "QM9", "QM7b", "ZINC", "Planetoid", "PygGraphPropPredDataset",
                "PygNodePropPredDataset", "PCQM4Mv2", "MD17", "PPI", "Amazon", "Coauthor"}
PATH_FUNCS = {"Path", "PurePath", "PosixPath", "WindowsPath"}
ARGS_NAMES = ("args", "opt", "opts", "cfg", "config", "FLAGS", "hparams", "params", "a")


# ------------------------------------------------------------------ globbing


def _glob_re(pattern: str) -> re.Pattern[str]:
    i, out = 0, ""
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            out += ".*"
            i += 2
        elif pattern[i] == "*":
            out += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(pattern[i])
            i += 1
    return re.compile("^" + out + "$")


def _matches(rel: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(p.match(rel) for p in patterns)


# ------------------------------------------------------------------ models


HELPER_IO_RE = re.compile(r"^(load|read|open|save|write|dump|store|export)(?:_\w+)?$", re.I)


@dataclass
class PathRef:
    raw: str
    path: str  # normalised, relative to the project root when possible
    alt: str | None = None  # same string interpreted relative to the script's folder
    how: str = ""
    line: int = 0

    def keys(self) -> list[str]:
        return [k for k in (self.path, self.alt) if k]


@dataclass
class ScriptInfo:
    rel: str
    name: str = ""
    is_notebook: bool = False
    section: str | None = None
    docstring: str = ""
    reads: list[PathRef] = field(default_factory=list)
    writes: list[PathRef] = field(default_factory=list)
    mentions: list[PathRef] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    local_imports: list[str] = field(default_factory=list)
    import_names: list[str] = field(default_factory=list)
    kind: str = "step"
    kind_scores: dict[str, int] = field(default_factory=dict)
    has_main: bool = False
    top_level_calls: int = 0
    order: float | None = None
    util_reason: str | None = None


# ------------------------------------------------------------ AST analysis


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):
        parts.append(_dotted(node.func) + "()")
    return ".".join(reversed(parts))


def _jsonable(v: Any) -> bool:
    try:
        json.dumps(v, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def _plausible_path(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 400 or "\n" in s:
        return False
    if re.match(r"^[a-z][a-z0-9+.-]*://", s) and not s.startswith("sqlite:///"):
        return False
    if s in ("*", ".", "/", "-", "r", "w", "rb", "wb", "a"):
        return False
    return bool(re.search(r"[A-Za-z0-9_]", s))


def _is_main_guard(test: ast.AST) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators))


QUIET_CALLS = ("logging", "warnings", "print", "sys.path", "os.environ", "load_dotenv", "np.random.seed",
               "torch.manual_seed", "random.seed", "set_seed", "seed_everything", "plt.style", "sns.set",
               "pd.set_option", "matplotlib.use", "torch.set_", "os.makedirs", "Path")


class _Analyzer(ast.NodeVisitor):
    def __init__(self, info: ScriptInfo, rel_dir: str):
        self.info = info
        self.dir = rel_dir
        self.env: dict[str, str] = {}
        self.args_defaults: dict[str, Any] = {}
        self.handles: set[str] = set()
        self.seen: set[tuple[str, str]] = set()

    # ---- constant folding for path expressions
    def ev(self, node: ast.AST | None) -> str | None:  # noqa: C901 - a small interpreter
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return "@root/" + self.info.rel
            return self.env.get(node.id)
        if isinstance(node, ast.JoinedStr):
            out = ""
            for v in node.values:
                if isinstance(v, ast.Constant):
                    out += str(v.value)
                else:
                    inner = self.ev(v.value) if isinstance(v, ast.FormattedValue) else None
                    out += inner if inner is not None else "*"
            return out
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            a, b = self.ev(node.left), self.ev(node.right)
            if a is None:
                return None
            return posixpath.join(a, b if b is not None else "*")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self.ev(node.left), self.ev(node.right)
            return a + b if a is not None and b is not None else None
        if isinstance(node, ast.Attribute):
            if node.attr == "parent":
                base = self.ev(node.value)
                return posixpath.dirname(base) if base else None
            if isinstance(node.value, ast.Name) and node.value.id in ARGS_NAMES and node.attr in self.args_defaults:
                v = self.args_defaults[node.attr]
                return v if isinstance(v, str) else None
            return self.env.get(_dotted(node))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in ARGS_NAMES:
            k = node.slice
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                v = self.args_defaults.get(k.value)
                return v if isinstance(v, str) else None
        if isinstance(node, ast.Call):
            fn = _dotted(node.func)
            last = fn.split(".")[-1]
            if last in PATH_FUNCS:
                parts = [self.ev(a) for a in node.args]
                if not parts:
                    return "."
                if parts[0] is None:
                    return None
                return posixpath.join(*[p if p is not None else "*" for p in parts])
            if fn in ("os.path.join", "path.join", "osp.join"):
                parts = [self.ev(a) for a in node.args]
                if not parts or parts[0] is None:
                    return None
                return posixpath.join(*[p if p is not None else "*" for p in parts])
            if fn in ("os.path.dirname", "osp.dirname"):
                v = self.ev(node.args[0]) if node.args else None
                return posixpath.dirname(v) if v else None
            if fn in ("os.path.abspath", "os.path.realpath", "osp.abspath", "str", "os.fspath", "os.path.expanduser"):
                return self.ev(node.args[0]) if node.args else None
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("resolve", "absolute", "expanduser", "as_posix"):
                    return self.ev(node.func.value)
                if attr == "joinpath":
                    base = self.ev(node.func.value)
                    parts = [self.ev(a) for a in node.args]
                    if base is None:
                        return None
                    return posixpath.join(base, *[p if p is not None else "*" for p in parts])
                if attr == "with_suffix" and node.args:
                    base, suf = self.ev(node.func.value), self.ev(node.args[0])
                    return posixpath.splitext(base)[0] + suf if base and suf is not None else None
                if attr == "format":
                    tmpl = self.ev(node.func.value)
                    return re.sub(r"\{[^}]*\}", "*", tmpl) if tmpl else None
        return None

    def norm(self, raw: str) -> tuple[str, str | None]:
        raw = raw.strip().replace("\\", "/")
        if raw.startswith("sqlite:///"):
            raw = raw[len("sqlite:///"):]
        if raw.startswith("@root/"):
            return posixpath.normpath(raw[len("@root/"):]), None
        if raw.startswith(("/", "~")) or re.match(r"^[A-Za-z]:/", raw):
            return raw, None
        primary = posixpath.normpath(raw)
        alt = posixpath.normpath(posixpath.join(self.dir, raw)) if self.dir else None
        if alt and (self.info.is_notebook or primary.startswith("../")) and not alt.startswith("../"):
            primary, alt = alt, primary  # notebooks run from their own folder
        return primary, (alt if alt != primary else None)

    def record(self, kind: str, node: ast.AST | None, how: str, line: int) -> bool:
        raw = self.ev(node)
        if not raw or not _plausible_path(raw):
            return False
        p, alt = self.norm(raw)
        if (kind, p) in self.seen:
            return True
        self.seen.add((kind, p))
        ref = PathRef(raw=raw, path=p, alt=alt, how=how, line=line)
        (self.info.reads if kind == "read" else self.info.writes).append(ref)
        return True

    def score(self, kind: str, w: int) -> None:
        self.info.kind_scores[kind] = self.info.kind_scores.get(kind, 0) + w

    # ---- pass 1: params and simple assignments
    def prepass(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._param_call(node)
        for _ in range(3):  # resolve chains like BASE = Path(__file__).parent; DATA = BASE / "data"
            for node in ast.walk(tree):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    v = self.ev(node.value)
                    if v is not None:
                        for t in targets:
                            if isinstance(t, (ast.Name, ast.Attribute)):
                                self.env[_dotted(t)] = v
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = node.args
                    pos = a.posonlyargs + a.args
                    pairs = list(zip(pos[len(pos) - len(a.defaults):], a.defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
                    for arg, d in pairs:
                        v = self.ev(d) if d is not None else None
                        if v is not None:
                            self.env.setdefault(arg.arg, v)

    def _param_call(self, node: ast.Call) -> None:
        fn = _dotted(node.func)
        last = fn.split(".")[-1]
        if last in ("add_argument", "option", "argument") and node.args:
            flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if not flags:
                return
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if isinstance(kw.get("dest"), ast.Constant):
                name = str(kw["dest"].value)  # type: ignore[union-attr]
            else:
                longs = [f for f in flags if f.startswith("--")]
                name = (longs[0] if longs else flags[0]).lstrip("-").replace("-", "_")
            if not re.match(r"^[A-Za-z_]\w*$", name or ""):
                return
            default: Any = None
            if "default" in kw:
                try:
                    default = ast.literal_eval(kw["default"])
                except Exception:  # noqa: BLE001
                    default = self.ev(kw["default"])
            elif isinstance(kw.get("action"), ast.Constant) and kw["action"].value == "store_true":  # type: ignore[union-attr]
                default = False
            elif isinstance(kw.get("is_flag"), ast.Constant) and kw["is_flag"].value:  # type: ignore[union-attr]
                default = False
            if _jsonable(default):
                self.info.params[name] = default
                self.args_defaults[name] = default
                if last in ("option", "argument") and isinstance(default, str):
                    self.env.setdefault(name, default)  # click passes options as function arguments
        for k in node.keywords:
            if k.arg in KW_PARAMS and k.arg not in self.info.params:
                try:
                    v = ast.literal_eval(k.value)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(v, (int, float, str, bool)):
                    self.info.params[k.arg] = v

    # ---- pass 2: structure and I/O
    def visit_Module(self, node: ast.Module) -> None:
        doc = ast.get_docstring(node)
        if doc and not self.info.docstring:
            self.info.docstring = doc
        for stmt in node.body:
            if isinstance(stmt, ast.If) and _is_main_guard(stmt.test):
                self.info.has_main = True
            call = None
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
            elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                call = stmt.value
            elif isinstance(stmt, (ast.With, ast.For)):
                self.info.top_level_calls += 1
            if call is not None:
                fn = _dotted(call.func)
                if not fn.startswith(QUIET_CALLS) and fn.split(".")[-1] not in ("ArgumentParser", "getLogger", "TypeVar", "namedtuple", "dataclass", "Path"):
                    self.info.top_level_calls += 1
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(getattr(stmt, "value", None), ast.Constant):
                tgt = stmt.targets[0] if isinstance(stmt, ast.Assign) else stmt.target
                v = stmt.value.value  # type: ignore[union-attr]
                if isinstance(tgt, ast.Name) and tgt.id.isupper() and not tgt.id.startswith("_") and isinstance(v, (int, float, bool)):
                    self.info.params.setdefault(tgt.id.lower(), v)
            if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.IfExp)
                    and "sys.argv" in ast.unparse(stmt.value.body) and isinstance(stmt.value.orelse, ast.Constant)
                    and _jsonable(stmt.value.orelse.value)):
                self.info.params.setdefault(stmt.targets[0].id, stmt.value.orelse.value)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.info.import_names += [a.name for a in node.names]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = ("." * node.level) + (node.module or "")
        self.info.import_names.append(mod)
        for a in node.names:
            self.info.import_names.append(f"{mod}.{a.name}" if node.module else f"{mod}{a.name}")
        self.generic_visit(node)

    def _open(self, node: ast.Call) -> bool:
        fn = _dotted(node.func)
        mode_node: ast.AST | None = None
        if fn in ("open", "io.open", "codecs.open", "gzip.open", "bz2.open", "lzma.open", "h5py.File", "tarfile.open", "zipfile.ZipFile"):
            target = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ("file", "filename", "name")), None)
            mode_node = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "open" and self.ev(node.func.value) is not None:
            target = node.func.value
            mode_node = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "mode"), None)
        else:
            return False
        mode = mode_node.value if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str) else "r"
        self.record("write" if any(c in mode for c in "wax") else "read", target, f"open({mode})", node.lineno)
        return True

    def visit_Call(self, node: ast.Call) -> None:  # noqa: C901
        fn = _dotted(node.func)
        parts = fn.split(".")
        last, base = parts[-1], parts[0]
        if last in CALL_HINTS:
            k, w = CALL_HINTS[last]
            self.score(k, w)
        if last == "step" and isinstance(node.func, ast.Attribute) and re.search(r"optim|opt$", _dotted(node.func.value), re.I):
            self.score("train", 3)
        a0 = node.args[0] if node.args else None
        a1 = node.args[1] if len(node.args) > 1 else None
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if self._open(node):
            pass
        elif last in READ_FIRST:
            self.record("read", a0 if a0 is not None else kw.get("filepath_or_buffer") or kw.get("path") or kw.get("fileName") or kw.get("checkpoint_path"), fn, node.lineno)
        elif last == "load" and base in LOAD_MODULES:
            self.record("read", a0 if a0 is not None else kw.get("f") or kw.get("file") or kw.get("filename"), fn, node.lineno)
        elif last in ("read_text", "read_bytes") and isinstance(node.func, ast.Attribute):
            self.record("read", node.func.value, fn, node.lineno)
        elif last in ("write_text", "write_bytes") and isinstance(node.func, ast.Attribute):
            self.record("write", node.func.value, fn, node.lineno)
        elif last in WRITE_FIRST:
            self.record("write", a0 if a0 is not None else kw.get("path_or_buf") or kw.get("path") or kw.get("fname") or kw.get("filepath"), fn, node.lineno)
        elif last == "save" and base in ("np", "numpy", "jnp"):
            self.record("write", a0 if a0 is not None else kw.get("file"), fn, node.lineno)
        elif (last == "save" and base == "torch") or (last == "dump" and base in ("joblib", "dill")):
            self.record("write", a1 if a1 is not None else kw.get("f") or kw.get("filename"), fn, node.lineno)
        elif last in WRITE_SECOND:
            self.record("write", a1 if a1 is not None else kw.get("filename"), fn, node.lineno)
        elif last in ("save", "save_weights", "export") and isinstance(node.func, ast.Attribute) and a0 is not None:
            raw = self.ev(a0)
            if raw and posixpath.splitext(raw)[1].lower() in DATA_EXTS:
                self.record("write", a0, fn, node.lineno)
        elif base == "shutil" and last in ("copy", "copyfile", "copy2", "move", "copytree") and a0 is not None and a1 is not None:
            self.record("read", a0, fn, node.lineno)
            self.record("write", a1, fn, node.lineno)
        elif last in PYG_DATASETS:
            self.record("read", kw.get("root") or a0, fn, node.lineno)
        elif base == "sqlite3" and last == "connect":
            self.record("read", a0, fn, node.lineno)
        elif isinstance(node.func, ast.Name) and a0 is not None and HELPER_IO_RE.match(last):
            # project helpers such as load("graphs.json") / save_pickle(obj, "model.pkl")
            arg = next((a for a in node.args if (self.ev(a) or "").lower().endswith(tuple(DATA_EXTS))), None)
            if arg is not None:
                self.record("read" if HELPER_IO_RE.match(last).group(1) in ("load", "read", "open") else "write",  # type: ignore[union-attr]
                            arg, fn + "() helper", node.lineno)
        if last in ("create_study", "load_study") and "storage" in kw:
            self.record("read", kw["storage"], fn + "(storage)", node.lineno)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        v = node.value
        if isinstance(v, str) and 2 < len(v) < 300 and "\n" not in v and " " not in v.strip():
            ext = posixpath.splitext(v)[1].lower()
            if (ext in DATA_EXTS and (ext not in WEAK_EXTS or "/" in v)) or v.startswith("sqlite:///"):
                if _plausible_path(v):
                    p, alt = self.norm(v)
                    self.info.mentions.append(PathRef(raw=v, path=p, alt=alt, how="literal", line=getattr(node, "lineno", 0)))


def _analyze_source(src: str, info: ScriptInfo, rel_dir: str) -> None:
    tree = ast.parse(src)
    an = _Analyzer(info, rel_dir)
    an.prepass(tree)
    an.visit(tree)
    io_keys = {k for r in info.reads + info.writes for k in r.keys()}
    seen: set[str] = set()
    mentions = []
    for m in info.mentions:
        if set(m.keys()) & io_keys or m.path in seen:
            continue
        seen.add(m.path)
        mentions.append(m)
    info.mentions = mentions


# ------------------------------------------------------------ kind & naming


def _infer_kind(info: ScriptInfo) -> None:
    stem = (info.section or Path(info.rel).stem).lower()
    name_l = re.sub(r"^\d+[_\-. ]*", "", stem)
    scores = dict(info.kind_scores)
    for pat, k in NAME_HINTS:
        if re.search(pat, name_l):
            scores[k] = scores.get(k, 0) + 4
            break
    for mod, k, w in IMPORT_HINTS:
        if any(n == mod or n.startswith(mod + ".") for n in info.import_names):
            scores[k] = scores.get(k, 0) + w
    if not scores and info.reads:
        scores["data"] = 1
    info.kind_scores = scores
    real = {k: v for k, v in scores.items() if k != "util"}
    if real:
        info.kind = max(real.items(), key=lambda kv: (kv[1], -KINDS.index(kv[0])))[0]
    elif scores.get("util"):
        info.kind = "util"


def _order_of(stem: str) -> float | None:
    m = re.match(r"^(\d+)(?:[._-](\d+)(?=[._-]))?", stem)
    if m:
        return float(m.group(1)) + (float("0." + m.group(2)) if m.group(2) else 0.0)
    m = re.search(r"(?:step|stage|part)[_\-]?(\d+)", stem, re.I)
    return float(m.group(1)) if m else None


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip()).strip("_.-")
    return (re.sub(r"_+", "_", s) or "step")[:64]


def _step_name(rel: str, taken: set[str], section: str | None = None) -> str:
    p = Path(rel)
    stem = re.sub(r"^(?:\d+|(?:step|stage|part)[_\-]?\d+)[_\-. ]*", "", p.stem, flags=re.I) or p.stem
    base = _slug(stem.lower())
    if section:
        base = _slug(f"{base}.{section.lower()}")
    if not STEP_NAME_RE.match(base):
        base = "s_" + base
    name = base
    if name in taken and p.parent.name:
        name = _slug(f"{p.parent.name.lower()}.{base}")
    i = 2
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    taken.add(name)
    return name


# ------------------------------------------------------------ notebooks


def _notebook_cells(path: Path) -> list[dict[str, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb.get("cells") if isinstance(nb, dict) else None
    if not isinstance(cells, list):
        raise ValueError("not a Jupyter notebook (no cells)")
    out = []
    for c in cells:
        src = c.get("source", "")
        out.append({"type": str(c.get("cell_type")), "source": "".join(src) if isinstance(src, list) else str(src)})
    return out


def _nb_code(cells: list[str]) -> str:
    ok = []
    for c in cells:
        code = "\n".join("" if ln.lstrip().startswith(("%", "!")) else ln for ln in c.splitlines())
        try:
            ast.parse(code)
            ok.append(code)
        except SyntaxError:
            continue  # skip cells that are not plain Python
    return "\n\n".join(ok)


def _nb_infos(path: Path, rel: str, sections: bool) -> list[ScriptInfo]:
    cells = _notebook_cells(path)
    first_md = next((c["source"] for c in cells if c["type"] == "markdown" and c["source"].strip()), "")
    doc = re.sub(r"^#+\s*", "", first_md.strip())
    if not sections:
        info = ScriptInfo(rel=rel, is_notebook=True)
        _analyze_source(_nb_code([c["source"] for c in cells if c["type"] == "code"]), info, posixpath.dirname(rel))
        info.docstring = info.docstring or doc
        info.has_main = True
        return [info]
    groups: list[tuple[str, list[str]]] = [("setup", [])]
    for c in cells:
        if c["type"] == "markdown":
            m = re.match(r"^\s*#{1,3}\s+(.+)", c["source"])
            if m:
                groups.append((m.group(1).strip(), []))
        elif c["type"] == "code":
            groups[-1][1].append(c["source"])
    infos = []
    for title, codes in groups:
        if not any(x.strip() for x in codes):
            continue
        info = ScriptInfo(rel=rel, is_notebook=True, section=title)
        _analyze_source(_nb_code(codes), info, posixpath.dirname(rel))
        info.docstring = f"{Path(rel).name}, section: {title}"
        info.has_main = True
        infos.append(info)
    return infos


# ------------------------------------------------------------ imports


def _module_index(rels: Iterable[str]) -> dict[str, str]:
    idx: dict[str, str] = {}
    for rel in rels:
        mod = rel[:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        idx.setdefault(mod, rel)
        for prefix in ("src.", "lib.", "scripts.", "code.", "app."):
            if mod.startswith(prefix):
                idx.setdefault(mod[len(prefix):], rel)
    return idx


def _resolve_imports(info: ScriptInfo, idx: dict[str, str], by_dir_stem: dict[tuple[str, str], str]) -> None:
    here = posixpath.dirname(info.rel)
    found: list[str] = []
    for name in info.import_names:
        if name.startswith("."):
            level = len(name) - len(name.lstrip("."))
            b = here
            for _ in range(level - 1):
                b = posixpath.dirname(b)
            cands = [((b.replace("/", ".") + ".") if b else "") + name.lstrip(".")]
        else:
            cands = [name] + ([here.replace("/", ".") + "." + name] if here else [])
        for cand in cands:
            parts = [p for p in cand.split(".") if p]
            hit = None
            while parts:
                c = ".".join(parts)
                if c in idx:
                    hit = idx[c]
                    break
                if len(parts) == 1 and (here, parts[0]) in by_dir_stem:
                    hit = by_dir_stem[(here, parts[0])]
                    break
                parts.pop()
            if hit:
                if hit != info.rel and hit not in found:
                    found.append(hit)
                break
    info.local_imports = found


# ------------------------------------------------------------ existing results


@dataclass
class FoundRun:
    key: str
    source: str  # metrics_json | metrics_csv | mlflow | optuna | lightning | wandb
    origin: str
    step_hint: str | None
    kind_hint: str | None
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "succeeded"
    notes: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    parent_key: str | None = None
    tags: list[str] = field(default_factory=list)
    kind: str = "run"  # "sweep" for a study / grid search that groups child trials
    sweep: dict[str, Any] | None = None


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v) if isinstance(v, (int, float)) else float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _flatten(d: dict[str, Any], prefix: str = "", depth: int = 0) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and depth < 2:
            out.update(_flatten(v, key + ".", depth + 1))
        elif not isinstance(v, str):
            n = _num(v)
            if n is not None:
                out[key] = n
    return out


def _ts(v: Any) -> str | None:
    from datetime import datetime, timezone

    n = _num(v)
    if n is None:
        return None
    if n > 1e11:
        n /= 1000.0
    return datetime.fromtimestamp(n, tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iso(s: Any) -> str | None:
    if not s:
        return None
    s = str(s).replace(" ", "T").split(".")[0]
    return s if s.endswith("Z") or "+" in s[10:] else s + "Z"


def _key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _yaml(p: Path) -> dict[str, Any] | None:
    try:
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None
    return d if isinstance(d, dict) else None


def _read_small(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if re.match(r"^-?\d+$", s):
            return int(s)
        n = _num(s)
        if n is not None:
            return n
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
    return v


METRIC_FILE_RE = re.compile(r"(metric|score|result|eval|perf|summary|report|cv_results|grid|search)s?[^/]*\.(json|csv)$", re.I)


def find_results(root: Path, base: Path, max_depth: int = 6) -> tuple[list[FoundRun], list[dict[str, str]]]:
    runs: list[FoundRun] = []
    notes: list[dict[str, str]] = []
    depth0 = len(base.parts)
    for d, dirs, files in os.walk(base):
        p = Path(d)
        dirs[:] = sorted(x for x in dirs if x not in PRUNE_DIRS)
        if len(p.parts) - depth0 >= max_depth:
            dirs[:] = []
        if p.name == "mlruns":
            runs += _mlflow(root, p)
            dirs[:] = []
            continue
        if p.name == "wandb":
            runs += _wandb(root, p)
            dirs[:] = []
            continue
        if p.name == "lightning_logs":
            runs += _lightning(root, p, notes)
            dirs[:] = []
            continue
        if any(f.startswith("events.out.tfevents") for f in files):
            notes.append({"path": p.relative_to(root).as_posix(), "reason": "TensorBoard event files: not imported (binary format)"})
        for f in sorted(files):
            fp = p / f
            low = f.lower()
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            if low.endswith((".db", ".sqlite", ".sqlite3")) and size < 1_000_000_000:
                runs += _optuna(root, fp) or []
            elif METRIC_FILE_RE.search(low) and size < 5_000_000 and low not in ("package.json", "tsconfig.json"):
                if low.endswith(".json"):
                    r = _metrics_json(root, fp)
                    if r:
                        runs.append(r)
                else:
                    runs += _metrics_csv(root, fp)
    return runs, notes


def _metrics_json(root: Path, fp: Path) -> FoundRun | None:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    metrics = _flatten(data)
    if not metrics or len(metrics) > 200:
        return None
    params = {k: v for k, v in data.items() if isinstance(v, (str, bool)) and len(str(v)) < 200}
    rel = fp.relative_to(root).as_posix()
    st = fp.stat()
    return FoundRun(key=_key("metrics_json", rel, str(st.st_mtime_ns)), source="metrics_json", origin=rel, step_hint=None,
                    kind_hint="eval", params=params, metrics=metrics, notes=f"Imported from {rel}", finished_at=_ts(st.st_mtime))


def _metrics_csv(root: Path, fp: Path) -> list[FoundRun]:
    try:
        rows = list(csv.DictReader(io.StringIO(fp.read_text(encoding="utf-8"))))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    if not rows or not rows[0]:
        return []
    rel = fp.relative_to(root).as_posix()
    mt = str(fp.stat().st_mtime_ns)
    cols = list(rows[0].keys())
    if "mean_test_score" in cols and any(c.startswith("param_") for c in cols):
        return _grid_csv(rel, mt, rows, cols)
    if len(cols) == 2 and all(_num(r[cols[1]]) is not None and _num(r[cols[0]]) is None for r in rows):
        metrics = {str(r[cols[0]]): _num(r[cols[1]]) for r in rows}  # metric,value table
        return [FoundRun(key=_key("metrics_csv", rel, mt), source="metrics_csv", origin=rel, step_hint=None, kind_hint="eval",
                         metrics={k: v for k, v in metrics.items() if v is not None}, notes=f"Imported from {rel}")]
    out = []
    for i, r in enumerate(rows[:50]):
        metrics = {k: _num(v) for k, v in r.items() if k and _num(v) is not None}
        params = {k: v for k, v in r.items() if k and _num(v) is None and v not in (None, "")}
        if metrics:
            out.append(FoundRun(key=_key("metrics_csv", rel, str(i), mt), source="metrics_csv", origin=f"{rel}#row{i + 1}",
                                step_hint=None, kind_hint="eval", params=params, metrics=metrics,  # type: ignore[arg-type]
                                notes=f"Imported from {rel} (row {i + 1})"))
    return out


def _grid_csv(rel: str, mt: str, rows: list[dict[str, Any]], cols: list[str]) -> list[FoundRun]:
    """A scikit-learn cv_results_ table (GridSearchCV / RandomizedSearchCV saved with pandas) -> sweep + trials."""
    pcols = [c for c in cols if c.startswith("param_")]
    mcols = [c for c in cols if c.startswith(("mean_test_", "std_test_", "rank_test_", "mean_fit_time", "mean_train_"))]
    pkey = _key("grid-search", rel, mt)
    grid: dict[str, list[Any]] = {}
    out = [FoundRun(key=pkey, source="grid_search", origin=rel, step_hint=None, kind_hint="hpo", kind="sweep",
                    params={"n_candidates": len(rows)}, tags=["grid"],
                    sweep={"source": "grid_search", "metric": "mean_test_score", "mode": "max", "file": rel},
                    notes=f"Grid / randomized search results from {rel}; each candidate is a child run")]
    for i, r in enumerate(rows[:MAX_IMPORTED_RUNS]):
        params = {c[len("param_"):]: _coerce(r[c]) for c in pcols if r.get(c) not in (None, "")}
        for k, v in params.items():
            if _jsonable(v) and v not in grid.setdefault(k, []):
                grid[k].append(v)
        metrics = {c: _num(r[c]) for c in mcols if _num(r.get(c)) is not None}
        out.append(FoundRun(key=_key("grid-search", rel, str(i), mt), source="grid_search", origin=f"{rel}#row{i + 1}",
                            step_hint=None, kind_hint="hpo", params=params, metrics=metrics,  # type: ignore[arg-type]
                            parent_key=pkey, notes=f"Candidate {i + 1} of the search in {rel}"))
    out[0].params["grid"] = grid
    out[0].sweep["grid"] = grid  # type: ignore[index]
    return out


def _mlflow(root: Path, mlruns: Path) -> list[FoundRun]:
    out: list[FoundRun] = []
    status_map = {1: "running", 2: "pending", 3: "succeeded", 4: "failed", 5: "cancelled", "RUNNING": "running",
                  "FINISHED": "succeeded", "FAILED": "failed", "KILLED": "cancelled", "SCHEDULED": "pending"}
    for exp in sorted(x for x in mlruns.iterdir() if x.is_dir() and not x.name.startswith(".") and x.name != "models"):
        exp_name = (_yaml(exp / "meta.yaml") or {}).get("name", exp.name)
        for run in sorted(x for x in exp.iterdir() if x.is_dir()):
            meta = _yaml(run / "meta.yaml")
            if meta is None:
                continue
            params = {p.name: _coerce(_read_small(p)) for p in sorted((run / "params").glob("*")) if p.is_file()}
            metrics: dict[str, float] = {}
            for m in sorted((run / "metrics").rglob("*")):
                if m.is_file():
                    lines = (_read_small(m) or "").splitlines()
                    bits = lines[-1].split() if lines else []
                    v = _num(bits[1]) if len(bits) >= 2 else None
                    if v is not None:
                        metrics[m.relative_to(run / "metrics").as_posix()] = v
            tags = {p.name: _read_small(p) for p in (run / "tags").glob("*") if p.is_file()}
            name = meta.get("run_name") or tags.get("mlflow.runName") or run.name
            out.append(FoundRun(
                key=_key("mlflow", exp.name, run.name), source="mlflow", origin=run.relative_to(root).as_posix(),
                step_hint=tags.get("mlflow.source.name") or None, kind_hint="train", params=params, metrics=metrics,
                status=status_map.get(meta.get("status"), "succeeded"), started_at=_ts(meta.get("start_time")),
                finished_at=_ts(meta.get("end_time")), notes=f"Imported from MLflow experiment {exp_name}, run {name}"))
    return out


def _lightning(root: Path, logs: Path, notes: list[dict[str, str]]) -> list[FoundRun]:
    out: list[FoundRun] = []
    for ver in sorted(x for x in logs.iterdir() if x.is_dir() and x.name.startswith("version_")):
        hp = _yaml(ver / "hparams.yaml") or {}
        params = {k: v for k, v in hp.items() if _jsonable(v) and not isinstance(v, (dict, list))}
        metrics: dict[str, float] = {}
        mcsv = ver / "metrics.csv"
        if mcsv.is_file():
            try:
                for row in csv.DictReader(io.StringIO(mcsv.read_text(encoding="utf-8"))):
                    for k, v in row.items():
                        n = _num(v)
                        if k and n is not None and k != "step":
                            metrics[k] = n  # last logged value wins
            except (OSError, UnicodeDecodeError, csv.Error):
                pass
        elif any(f.name.startswith("events.out.tfevents") for f in ver.iterdir()):
            notes.append({"path": ver.relative_to(root).as_posix(), "reason": "TensorBoard events only: params imported, metrics not"})
        if params or metrics:
            rel = ver.relative_to(root).as_posix()
            out.append(FoundRun(key=_key("lightning", rel), source="lightning", origin=rel, step_hint=None, kind_hint="train",
                                params=params, metrics=metrics, notes=f"Imported from Lightning logs {rel}"))
    return out


def _wandb(root: Path, wdir: Path) -> list[FoundRun]:
    out: list[FoundRun] = []
    for run in sorted(x for x in wdir.iterdir() if x.is_dir() and re.match(r"^(offline-)?run-", x.name)):
        files = run / "files"
        try:
            summary = json.loads((files / "wandb-summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = {}
        try:
            meta = json.loads((files / "wandb-metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        params = {}
        for k, v in (_yaml(files / "config.yaml") or {}).items():
            if k.startswith("_") or k == "wandb_version":
                continue
            val = v.get("value") if isinstance(v, dict) and "value" in v else v
            if _jsonable(val) and not isinstance(val, (dict, list)):
                params[k] = val
        metrics = _flatten({k: v for k, v in summary.items() if not k.startswith("_")}) if isinstance(summary, dict) else {}
        if metrics or params:
            out.append(FoundRun(key=_key("wandb", run.name), source="wandb", origin=run.relative_to(root).as_posix(),
                                step_hint=meta.get("codePath") or meta.get("program"), kind_hint="train", params=params,
                                metrics=metrics, started_at=meta.get("startedAt"), notes=f"Imported from wandb run {run.name}"))
    return out


def _decode_optuna_param(value: Any, dist_json: Any) -> Any:
    try:
        dist = json.loads(dist_json) if dist_json else {}
    except ValueError:
        dist = {}
    name, attrs = dist.get("name", ""), dist.get("attributes", {})
    if name == "CategoricalDistribution" and isinstance(attrs.get("choices"), list):
        try:
            return attrs["choices"][int(value)]
        except (ValueError, IndexError, TypeError):
            return value
    if name.startswith("Int"):
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    return value


def _optuna(root: Path, fp: Path) -> list[FoundRun] | None:
    try:
        con = sqlite3.connect(f"file:{fp}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return None
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"studies", "trials", "trial_params", "trial_values"} <= tables:
            return None
        rel = fp.relative_to(root).as_posix()
        directions: dict[int, list[str]] = {}
        if "study_directions" in tables:
            for sid, d in con.execute("SELECT study_id, direction FROM study_directions ORDER BY objective"):
                directions.setdefault(sid, []).append(str(d).lower().replace("studydirection.", ""))
        out: list[FoundRun] = []
        state_map = {"COMPLETE": "succeeded", "FAIL": "failed", "PRUNED": "cancelled", "RUNNING": "running", "WAITING": "pending"}
        for sid, sname in con.execute("SELECT study_id, study_name FROM studies").fetchall():
            dirs = ",".join(directions.get(sid, [])) or "minimize"
            trials = con.execute("SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials "
                                 "WHERE study_id=? ORDER BY number", (sid,)).fetchall()
            pkey = _key("optuna-study", rel, str(sname))
            out.append(FoundRun(key=pkey, source="optuna", origin=f"{rel}#{sname}", step_hint=None, kind_hint="hpo",
                                params={"study": sname, "direction": dirs, "n_trials": len(trials)},
                                notes=f"Optuna study {sname!r} from {rel}; trials are child runs", tags=["study"],
                                kind="sweep", sweep={"source": "optuna", "study": sname, "file": rel,
                                                     "metric": "value" if "," not in dirs else "value_0",
                                                     "mode": "max" if dirs.split(",")[0] == "maximize" else "min"}))
            for tid, num, state, ds, dc in trials[:MAX_IMPORTED_RUNS]:
                params = {p: _decode_optuna_param(v, dj) for p, v, dj in
                          con.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?", (tid,))}
                vals = con.execute("SELECT objective, value FROM trial_values WHERE trial_id=? ORDER BY objective", (tid,)).fetchall()
                metrics = {("value" if len(vals) == 1 else f"value_{o}"): _num(v) for o, v in vals if _num(v) is not None}
                out.append(FoundRun(key=_key("optuna", rel, str(sname), str(num)), source="optuna",
                                    origin=f"{rel}#{sname}/trial{num}", step_hint=None, kind_hint="hpo", params=params,
                                    metrics=metrics, status=state_map.get(str(state).upper().split(".")[-1], "succeeded"),  # type: ignore[arg-type]
                                    started_at=_iso(ds), finished_at=_iso(dc), parent_key=pkey,
                                    notes=f"Optuna trial {num} of study {sname!r} ({dirs})"))
        return out
    except sqlite3.Error:
        return None
    finally:
        con.close()


# ------------------------------------------------------------ main entry


def _pattern_match(a: str, b: str) -> bool:
    """Same file? Either side may contain * (from f-strings / unknown variables)."""
    if a == b:
        return True
    if "*" in a and "*" not in b:
        return bool(_glob_re(a).match(b))
    if "*" in b and "*" not in a:
        return bool(_glob_re(b).match(a))
    return False


def _is_inside(path: str, folder: str) -> bool:
    return folder not in (".", "", "*") and "*" not in folder and path.startswith(folder.rstrip("/") + "/")


def _cycle(adj: dict[str, set[str]], src: str, dst: str) -> bool:
    stack, seen = [dst], set()
    while stack:
        n = stack.pop()
        if n == src:
            return True
        if n not in seen:
            seen.add(n)
            stack.extend(adj.get(n, ()))
    return False


def _dedupe(xs: Iterable[str]) -> list[str]:
    out: list[str] = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


FIELDS = ("script_path", "description", "kind", "inputs", "outputs", "params")


def scan_project(  # noqa: C901 - one linear pipeline, easier to read in one place
    store: Store,
    path: str = ".",
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    dry_run: bool = False,
    notebook_sections: bool = False,
    import_results: bool = True,
) -> dict[str, Any]:
    root = store.root
    base = (root / (path or ".")).resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise MLTreeError(f"path {path!r} is outside the project root {root}") from exc
    if not base.is_dir():
        raise MLTreeError(f"path {path!r} is not a directory under {root}")
    inc = [_glob_re(p) for p in (include or DEFAULT_INCLUDE)]
    exc_pats = [_glob_re(p) for p in (DEFAULT_EXCLUDE if exclude is None else exclude)]

    # 1. files
    files: list[str] = []
    skipped: list[dict[str, str]] = []
    for d, dirs, fnames in os.walk(base):
        pruned = [x for x in dirs if x in PRUNE_DIRS or x.endswith(".egg-info")]
        dirs[:] = sorted(x for x in dirs if x not in pruned)
        for f in sorted(fnames):
            rel = (Path(d) / f).relative_to(root).as_posix()
            if not _matches(rel, inc):
                continue
            if _matches(rel, exc_pats):
                skipped.append({"path": rel, "reason": "excluded"})
            else:
                files.append(rel)
    if len(files) > MAX_FILES:
        skipped += [{"path": r, "reason": f"over the {MAX_FILES}-file limit"} for r in files[MAX_FILES:]]
        files = files[:MAX_FILES]

    # 2. static analysis (never imports or runs the code)
    infos: list[ScriptInfo] = []
    for rel in files:
        fp = root / rel
        try:
            size = fp.stat().st_size
            if rel.endswith(".ipynb"):
                if size > MAX_NB_BYTES:
                    skipped.append({"path": rel, "reason": "notebook too large"})
                    continue
                infos += _nb_infos(fp, rel, notebook_sections)
            else:
                if size > MAX_PY_BYTES:
                    skipped.append({"path": rel, "reason": "file too large"})
                    continue
                info = ScriptInfo(rel=rel)
                _analyze_source(fp.read_text(encoding="utf-8", errors="replace"), info, posixpath.dirname(rel))
                if Path(rel).name == "__init__.py" and not (info.reads or info.writes or info.has_main):
                    skipped.append({"path": rel, "reason": "package __init__"})
                    continue
                infos.append(info)
        except SyntaxError as e:
            skipped.append({"path": rel, "reason": f"syntax error at line {e.lineno}: {e.msg}"})
        except (ValueError, UnicodeDecodeError, OSError) as e:
            skipped.append({"path": rel, "reason": f"unreadable: {e}"})

    # 3. imports, utilities, kinds, order
    py = [i for i in infos if not i.is_notebook]
    idx = _module_index(i.rel for i in py)
    by_dir_stem = {(posixpath.dirname(i.rel), Path(i.rel).stem): i.rel for i in py}
    imported_by: dict[str, list[str]] = {}
    for i in infos:
        _resolve_imports(i, idx, by_dir_stem)
        for m in i.local_imports:
            imported_by.setdefault(m, []).append(i.rel)
    for i in infos:
        _infer_kind(i)
        i.order = _order_of(Path(i.rel).stem)
        if i.is_notebook:
            if i.kind == "util":
                i.kind = "step"
            continue
        no_io = not i.reads and not i.writes
        if i.kind == "util" and no_io and not i.has_main:
            i.util_reason = "helper module (name)"
        elif i.rel in imported_by and not i.has_main and i.top_level_calls == 0:
            i.kind, i.util_reason = "util", "imported by " + ", ".join(imported_by[i.rel][:3])
        elif no_io and not i.has_main and i.top_level_calls == 0:
            i.kind, i.util_reason = "util", "library module without I/O or entry point"
        elif i.kind == "util":
            i.kind = max((k for k in i.kind_scores if k != "util"), key=lambda k: i.kind_scores[k], default="step")
    utils = [i for i in infos if i.util_reason]
    steps = [i for i in infos if not i.util_reason]
    steps.sort(key=lambda i: (i.order if i.order is not None else 1e9, i.rel))

    # 4. names (stable across scans)
    initialized = store.is_initialized()
    pipe = store.get_pipeline() if initialized else {"nodes": [], "edges": []}
    ex_nodes = {n["name"]: n for n in pipe["nodes"]}
    ex_by_script: dict[str, str] = {}
    for n in pipe["nodes"]:
        if n["script_path"]:
            ex_by_script.setdefault(n["script_path"], n["name"])
    state = store.scan_state() if initialized else {}
    state_by_src = {(v["script_path"], v["fields"].get("section")): k for k, v in state.items()}
    taken = set(ex_nodes)
    for i in steps:
        prev = state_by_src.get((i.rel, i.section))
        if prev and prev in ex_nodes:
            i.name = prev
        elif i.section is None and i.rel in ex_by_script:
            i.name = ex_by_script[i.rel]
        else:
            i.name = _step_name(i.rel, taken, i.section)
    util_rels = {u.rel for u in utils}
    step_of_rel: dict[str, str] = {}
    for i in steps:
        step_of_rel.setdefault(i.rel, i.name)

    # 5. edges: data flow > imports > notebook order > file numbering
    adj: dict[str, set[str]] = {}
    for e in pipe["edges"]:
        adj.setdefault(e["source"], set()).add(e["target"])
    have = {(e["source"], e["target"]) for e in pipe["edges"]}
    edges: list[dict[str, Any]] = []
    incoming: dict[str, int] = {}

    def add_edge(a: str, b: str, evidence: str, label: str = "") -> None:
        if a == b or any(e["source"] == a and e["target"] == b for e in edges):
            return
        new = (a, b) not in have
        if new and _cycle(adj, a, b):
            return
        adj.setdefault(a, set()).add(b)
        incoming[b] = incoming.get(b, 0) + 1
        edges.append({"source": a, "target": b, "evidence": evidence, "label": label, "action": "new" if new else "exists"})

    consumed: set[tuple[str, str]] = set()
    produced: set[tuple[str, str]] = set()
    writers = [(i.name, w) for i in steps for w in i.writes]
    for i in steps:
        for r in i.reads:
            for wname, w in writers:
                if wname == i.name:
                    continue
                if any(_pattern_match(a, b) or _is_inside(b, a) for a in w.keys() for b in r.keys()):
                    add_edge(wname, i.name, "data", Path(r.path).name)
                    consumed.add((wname, w.path))
                    produced.add((i.name, r.path))
    for i in steps:
        for m in i.local_imports:
            if m in step_of_rel and m not in util_rels:
                add_edge(step_of_rel[m], i.name, "import")
    for a, b in zip(steps, steps[1:]):
        if a.is_notebook and b.is_notebook and a.rel == b.rel:
            add_edge(a.name, b.name, "notebook order")
    numbered = [i for i in steps if i.order is not None and (i.section is None or not i.is_notebook)]
    for a, b in zip(numbered, numbered[1:]):
        if not incoming.get(b.name):
            add_edge(a.name, b.name, "file order")

    def on_disk(ref: PathRef) -> bool:
        for k in ref.keys():
            if "*" not in k and (Path(k).expanduser() if k.startswith(("/", "~")) else root / k).exists():
                return True
        return False

    unresolved_inputs = [{"step": i.name, "path": r.path, "exists_on_disk": on_disk(r), "line": r.line}
                         for i in steps for r in i.reads if (i.name, r.path) not in produced]
    unconsumed_outputs = [{"step": i.name, "path": w.path, "line": w.line}
                          for i in steps for w in i.writes if (i.name, w.path) not in consumed]

    # 6. merge with what is already in the store
    if not dry_run and not initialized:
        store.init()
    out_steps = []
    for i in steps:
        desc = i.docstring.strip().split("\n\n")[0].replace("\n", " ").strip()[:240]
        uses = [u for u in i.local_imports if u in util_rels]
        if uses:
            desc = (desc + " " if desc else "") + "(uses " + ", ".join(Path(u).name for u in uses) + ")"
        scanned = {"script_path": i.rel, "description": desc, "kind": i.kind if i.kind in KINDS else "step",
                   "inputs": _dedupe(r.path for r in i.reads), "outputs": _dedupe(w.path for w in i.writes),
                   "params": i.params}
        cur = ex_nodes.get(i.name)
        row: dict[str, Any] = {"name": i.name, **scanned, "uses": uses, "notebook_section": i.section,
                               "mentioned_paths": [m.path for m in i.mentions][:10]}
        if cur is None:
            target, row["action"] = scanned, "new"
        else:
            cur_f = {k: cur.get(k) for k in FIELDS}
            last = state.get(i.name, {}).get("fields")
            user_owned = bool(state.get(i.name, {}).get("fields", {}).get("user_edited"))
            if last is not None and not user_owned and {k: last.get(k) for k in FIELDS} == cur_f:
                target = scanned  # untouched since our last scan: safe to refresh
                row["action"] = "unchanged" if scanned == cur_f else "updated"
            else:
                target = {  # edited by the user/assistant, or created by hand: only add, never overwrite
                    "script_path": cur["script_path"] or scanned["script_path"],
                    "description": cur["description"] or scanned["description"],
                    "kind": cur["kind"] if cur["kind"] not in ("step", "", None) else scanned["kind"],
                    "inputs": _dedupe(cur["inputs"] + scanned["inputs"]),
                    "outputs": _dedupe(cur["outputs"] + scanned["outputs"]),
                    "params": {**scanned["params"], **cur["params"]},
                }
                row["action"] = "unchanged" if target == cur_f else "merged"
                row["kept_user_edits"] = True
            row.update({k: target[k] for k in FIELDS})
        if not dry_run:
            if row["action"] in ("new", "updated", "merged"):
                store.add_step(i.name, script_path=target["script_path"], inputs=target["inputs"], outputs=target["outputs"],
                               params=target["params"], description=target["description"], kind=target["kind"])
            store.set_scan_state(i.name, i.rel, {**{k: target[k] for k in FIELDS}, "section": i.section,
                                                 "user_edited": bool(row.get("kept_user_edits"))})
        out_steps.append(row)
    if not dry_run:
        for e in edges:
            if e["action"] == "new":
                try:
                    store.connect_steps(e["source"], e["target"], e["label"] or e["evidence"])
                except MLTreeError as exc:
                    e["action"] = f"skipped: {exc}"

    # 7. existing results -> runs
    found, result_notes = find_results(root, base) if import_results else ([], [])
    kinds = {r["name"]: r["kind"] for r in out_steps}
    for n in pipe["nodes"]:
        kinds.setdefault(n["name"], n["kind"])
    outputs_of = {r["name"]: r["outputs"] for r in out_steps}
    reads_of = {i.name: [x.path for x in i.reads + i.mentions] for i in steps}
    file_step = {i.rel: i.name for i in steps if i.section is None}
    done = store.imported_keys() if store.is_initialized() else set()

    def attach(fr: FoundRun) -> str | None:
        if fr.step_hint:
            h = fr.step_hint.replace("\\", "/")
            for rel, nm in file_step.items():
                if h.endswith(rel) or Path(h).name == Path(rel).name:
                    return nm
        origin = fr.origin.split("#")[0]
        if fr.source.startswith("metrics"):
            for nm, outs in outputs_of.items():
                if any(_pattern_match(o, origin) for o in outs):
                    return nm
        if fr.source == "optuna":
            for nm, paths in reads_of.items():
                if any(_pattern_match(p, origin) or Path(p).name == Path(origin).name for p in paths):
                    return nm
        for k in {"eval": ["eval", "train", "hpo"], "train": ["train", "hpo", "eval"], "hpo": ["hpo", "train"]}.get(fr.kind_hint or "", []):
            cands = [nm for nm, kd in kinds.items() if kd == k]
            if cands:
                return cands[-1] if k == "eval" else cands[0]
        return None

    imported: list[dict[str, Any]] = []
    key_to_run: dict[str, str] = store.imported_run_ids() if (store.is_initialized() and not dry_run) else {}
    n_imp = 0
    for fr in found:
        step = attach(fr)
        entry: dict[str, Any] = {"source": fr.source, "origin": fr.origin, "step": step, "status": fr.status,
                                 "metrics": dict(list(fr.metrics.items())[:8])}
        if step is None:
            entry["action"] = "skipped: no matching step"
        elif fr.key in done:
            entry["action"] = "already imported"
        elif n_imp >= MAX_IMPORTED_RUNS:
            entry["action"] = "skipped: import limit"
        elif dry_run:
            entry["action"] = "would import"
        else:
            run = store.log_run(step, params={k: v for k, v in fr.params.items() if _jsonable(v)}, metrics=fr.metrics,
                                parent_run_id=key_to_run.get(fr.parent_key) if fr.parent_key else None, notes=fr.notes,
                                status=fr.status, tags=_dedupe(["imported", fr.source, *fr.tags]),
                                started_at=fr.started_at, finished_at=fr.finished_at, kind=fr.kind, sweep=fr.sweep, record_data=False)
            store.mark_imported(fr.key, run["id"], fr.origin)
            key_to_run[fr.key] = run["id"]
            entry["action"], entry["run_id"] = "imported", run["id"]
            n_imp += 1
        imported.append(entry)

    # present steps in pipeline order (topological, stable)
    pos = {r["name"]: n for n, r in enumerate(out_steps)}
    indeg = {r["name"]: 0 for r in out_steps}
    succ: dict[str, list[str]] = {}
    for e in edges:
        if e["source"] in indeg and e["target"] in indeg:
            indeg[e["target"]] += 1
            succ.setdefault(e["source"], []).append(e["target"])
    ready = sorted((n for n, d in indeg.items() if d == 0), key=pos.get)
    topo: list[str] = []
    while ready:
        n = ready.pop(0)
        topo.append(n)
        for m in succ.get(n, []):
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
        ready.sort(key=pos.get)
    topo += [n for n in pos if n not in topo]
    out_steps.sort(key=lambda r: topo.index(r["name"]))

    counts = {
        "steps": len(out_steps),
        **{f"steps_{a}": sum(1 for s in out_steps if s["action"] == a) for a in ("new", "updated", "merged", "unchanged")},
        "edges_total": len(edges), "edges_new": sum(1 for e in edges if e["action"] == "new"),
        "utilities": len(utils), "unresolved_inputs": len(unresolved_inputs), "unconsumed_outputs": len(unconsumed_outputs),
        "skipped": len(skipped), "results_found": len(found), "runs_imported": n_imp,
    }
    summary: dict[str, Any] = {
        "project": str(root), "scanned": base.relative_to(root).as_posix() or ".", "dry_run": dry_run,
        "files_scanned": len(files), "counts": counts, "steps": out_steps, "edges": edges,
        "utilities": [{"path": u.rel, "reason": u.util_reason} for u in utils],
        "unresolved_inputs": unresolved_inputs[:100], "unconsumed_outputs": unconsumed_outputs[:100],
        "skipped": [s for s in skipped if s["reason"] != "excluded"][:50] + [s for s in skipped if s["reason"] == "excluded"][:20],
        "runs_imported": n_imp, "imported": imported[:100], "results_notes": result_notes,
    }
    summary["hint"] = _hint(summary)
    return summary


def _hint(s: dict[str, Any]) -> str:
    c = s["counts"]
    out = [("DRY RUN: nothing was written. " if s["dry_run"] else "")
           + f"Static analysis found {c['steps']} steps and {c['edges_total']} edges; no code was executed."]
    if c["unresolved_inputs"]:
        out.append(f"{c['unresolved_inputs']} inputs are not produced by any scanned step. Raw data is expected here, "
                   "but if one comes from another script, call connect_steps(source, target).")
    weak = sum(1 for e in s["edges"] if e["evidence"] in ("file order", "import"))
    if weak:
        out.append(f"{weak} edges are guesses from file numbering or imports; verify them with get_pipeline.")
    out.append("Review kinds, descriptions and params, and fix anything wrong with add_step (later scans keep your edits). "
               "Then call open_map to show the user the map.")
    if s["dry_run"]:
        out.append("Call scan_project with dry_run=false to write this map.")
    return " ".join(out)


def format_summary(s: dict[str, Any]) -> str:
    c = s["counts"]
    L = [f"mltree scan of {s['project']} ({'dry run, nothing written' if s['dry_run'] else 'written to .mltree/'})",
         f"  {s['files_scanned']} files, {c['steps']} steps ({c['steps_new']} new, {c['steps_updated']} updated, "
         f"{c['steps_merged']} merged, {c['steps_unchanged']} unchanged), {c['edges_total']} edges ({c['edges_new']} new), "
         f"{c['utilities']} utility modules, {c['runs_imported']} runs imported", ""]
    for st in s["steps"]:
        L.append(f"  [{st['action']:>9}] {st['name']:<26} {st['kind']:<9} {st['script_path']}")
        if st["inputs"]:
            L.append(f"      in : {', '.join(st['inputs'][:6])}")
        if st["outputs"]:
            L.append(f"      out: {', '.join(st['outputs'][:6])}")
        if st["params"]:
            L.append(f"      params: {json.dumps(st['params'])[:140]}")
    if s["edges"]:
        L += ["", "  edges:"]
        L += [f"    {e['source']} -> {e['target']}  ({e['evidence']}{': ' + e['label'] if e['label'] else ''}) [{e['action']}]" for e in s["edges"]]
    if s["utilities"]:
        L += ["", "  utilities (not steps): " + ", ".join(u["path"] for u in s["utilities"])]
    if s["unresolved_inputs"]:
        L += ["", "  inputs not produced by any step:"]
        L += [f"    {u['step']}: {u['path']}{'' if u['exists_on_disk'] else '  (not on disk)'}" for u in s["unresolved_inputs"][:20]]
    if s["imported"]:
        L += ["", "  existing results:"] + [f"    {r['source']:<12} {r['origin']} -> {r['step']} [{r['action']}]" for r in s["imported"][:25]]
    L += [f"    note: {n['path']}: {n['reason']}" for n in s["results_notes"]]
    other = [x for x in s["skipped"] if x["reason"] != "excluded"]
    if other:
        L += ["", "  skipped:"] + [f"    {x['path']}: {x['reason']}" for x in other[:10]]
    L += ["", "  " + s["hint"]]
    return "\n".join(L)
