"""GNN views: static architecture parsing, dataset detection, the stats probe and gated introspection."""
import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mltree_mcp import Store, _gnn_probe, gnn
from mltree_mcp.store import MLTreeError
from mltree_mcp.viewer import create_app

from test_tools import call_err, call_ok

DEMO = Path(__file__).resolve().parents[1] / "examples" / "gnn_demo"
TG_PY = os.environ.get("MLTREE_TEST_TORCH_PYTHON", "/tmp/tg/bin/python")

GAT_SRC = '''
import torch
from torch import nn
from torch_geometric.nn import GATConv, GINConv, global_add_pool
from torch_geometric.datasets import TUDataset

HEADS = 4

class GAT(nn.Module):
    def __init__(self, in_dim, hidden=32, out_dim=2):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=HEADS)
        self.c2 = GATConv(hidden * HEADS, hidden, heads=1)
        self.gin = GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.c1(x, edge_index))
        x = self.c2(x, edge_index)
        x = self.gin(x, edge_index)
        return self.out(global_add_pool(x, batch))

ds = TUDataset(root="data/TU", name="MUTAG")
model = GAT(ds.num_features, hidden=16)
raise SystemExit("must never be executed by mltree")
'''


@pytest.fixture
def demo(tmp_path, monkeypatch):
    work = tmp_path / "gnn_demo"
    shutil.copytree(DEMO, work, ignore=shutil.ignore_patterns(".mltree", "processed", "__pycache__"))
    spec = importlib.util.spec_from_file_location("register_demo", work / "register_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    store = mod.build(root=work, reset=True)
    monkeypatch.setenv("MLTREE_PROJECT", str(work))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    return store


@pytest.fixture
def snip(store):
    (store.root / "steps").mkdir()
    (store.root / "steps" / "gat.py").write_text(GAT_SRC)
    (store.root / "steps" / "pre.py").write_text(
        "from torch_geometric.nn.models import GCN\nmodel = GCN(9, 128, 4, out_channels=1, dropout=0.2)\n")
    (store.root / "steps" / "plain.py").write_text("import sklearn\nprint('no graphs here')\n")
    store.add_step("gat", "steps/gat.py")
    store.add_step("pre", "steps/pre.py")
    store.add_step("plain", "steps/plain.py")
    return store


def test_demo_model_parsed_across_files(demo):
    a = gnn.analyze_step(demo, "train")
    (m,) = a["models"]
    assert m["name"] == "MolGCN" and m["file"] == "steps/gcn_model.py" and m["kind"] == "class"
    assert m["summary"]["label"] == "GCNConv ×3 · hidden 64 · global_mean_pool"
    assert m["summary"]["num_conv_layers"] == 3 and m["summary"]["head"] == "Linear(64 → 1)"
    convs = [la for la in m["layers"] if la["type"] == "GCNConv"]
    assert convs[0]["in"] == {"text": "in_channels", "value": 7}  # default from a module constant
    assert convs[1]["repeat"] == 2  # range(num_layers - 1) with num_layers=3 from the call site
    assert not any(la["family"] == "container" for la in m["layers"])  # ModuleList implied by its elements
    assert m["forward"][0] == "(convs[i] → relu) × len(convs)" and m["forward"][1] == "global_mean_pool"
    assert gnn.badge(a) == {"label": m["summary"]["label"], "model": "MolGCN"}
    node = next(n for n in demo.get_pipeline()["nodes"] if n["name"] == "train")
    assert node["gnn"]["model"] == "MolGCN"
    assert next(n for n in demo.get_pipeline()["nodes"] if n["name"] == "load_smiles")["gnn"] is None


def test_snippets(snip):
    a = gnn.analyze_step(snip, "gat")
    (m,) = a["models"]
    s = m["summary"]
    assert s["conv"] == {"GATConv": 2, "GINConv": 1} and s["heads"] == ["4", "1"] and s["pooling"] == ["global_add_pool"]
    c2 = next(la for la in m["layers"] if la["attr"] == "c2")
    assert c2["in"] == {"text": "hidden * HEADS", "value": 64}  # hidden=16 from the instantiation
    assert next(la for la in m["layers"] if la["attr"] == "gin")["mlp"]
    assert m["forward"] == ["c1", "relu", "c2", "gin", "global_add_pool", "out"]
    assert a["datasets"][0]["class"] == "TUDataset" and a["datasets"][0]["name"] == "MUTAG"
    assert gnn.badge(a)["datasets"] == ["TUDataset(MUTAG)"]
    p = gnn.analyze_step(snip, "pre")["models"][0]
    assert p["kind"] == "prebuilt" and p["summary"]["label"] == "GCN · 4 layers · hidden 128 · out 1"
    assert p["layers"][0]["repeat"] == 4
    assert gnn.badge(gnn.analyze_step(snip, "plain")) is None
    assert set(gnn.summary(snip)["steps"]) == {"gat", "pre"}


def test_broken_script_never_breaks_the_map(snip):
    (snip.root / "steps" / "gat.py").write_text("class (:\n")
    node = next(n for n in snip.get_pipeline()["nodes"] if n["name"] == "gat")
    assert node["gnn"] is None


class FakeTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    dtype = "float"

    def tolist(self):
        return self.data


class FakeData:
    """Duck-typed like torch_geometric.data.Data."""

    def __init__(self, n, edges, y):
        self.x = FakeTensor([[1.0, 0.0, 0.0]] * n)
        self.edge_index = FakeTensor([[a for a, _ in edges], [b for _, b in edges]])
        self.y = FakeTensor([y])
        self.num_nodes = n


def test_probe_summarize_graph_list():
    graphs = [FakeData(3, [(0, 1), (1, 0), (1, 2), (2, 1)], 0), FakeData(4, [(0, 1), (1, 0), (2, 2)], 1),
              FakeData(2, [(0, 1), (1, 0)], 1)]
    s = _gnn_probe.summarize(graphs)
    assert s["num_graphs"] == 3 and s["nodes"]["min"] == 2 and s["nodes"]["max"] == 4 and s["nodes"]["total"] == 9
    assert s["edges"]["total"] == 9 and s["node_features"] == 3
    assert s["self_loops"] == 1 and s["isolated_nodes"] == 1  # node 3 of graph 2
    assert s["labels"]["task"] == "classification" and s["labels"]["num_classes"] == 2
    with pytest.raises(ValueError):
        _gnn_probe.summarize({"not": "graphs"})


def _fake_python(tmp_path, payload):
    """An 'interpreter' that ignores the probe and prints a canned result (no torch needed)."""
    f = tmp_path / "fakepy"
    f.write_text("#!/bin/sh\necho noise\necho 'MLTREE_GNN_STATS " + json.dumps(payload) + "'\n")
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return str(f)


def test_introspect_caches_stats_and_detects_staleness(demo, tmp_path):
    graphs = demo.root / "data" / "processed" / "graphs.pt"
    graphs.parent.mkdir(parents=True, exist_ok=True)
    graphs.write_bytes(b"fake")
    py = _fake_python(tmp_path, {"python": "3.12", "versions": {"torch": "9"},
                                 "files": {str(graphs): {"format": "pyg-collated", "num_graphs": 20}}})
    res = gnn.introspect(demo, "featurize", python=py)
    assert res["files"]["data/processed/graphs.pt"]["num_graphs"] == 20
    v = gnn.step_view(demo, "featurize")
    (row,) = [r for r in v["dataset_files"] if r["path"] == "data/processed/graphs.pt"]
    assert row["stats"]["versions"] == {"torch": "9"} and row["stale"] is False and not v["can_introspect"]
    assert next(n for n in demo.get_pipeline()["nodes"] if n["name"] == "featurize")["gnn"] == {"graphs": 20}
    graphs.write_bytes(b"changed")
    (row,) = [r for r in gnn.step_view(demo, "featurize")["dataset_files"] if r["path"] == "data/processed/graphs.pt"]
    assert row["stale"] is True
    with pytest.raises(MLTreeError, match="no .pt"):
        gnn.introspect(demo, "load_smiles", python=py)
    bad = tmp_path / "badpy"
    bad.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    bad.chmod(0o755)
    with pytest.raises(MLTreeError, match="boom"):
        gnn.introspect(demo, "featurize", python=str(bad))


def test_introspection_is_gated(demo, tmp_path, monkeypatch):
    c = TestClient(create_app(demo.root))
    v = c.get("/api/steps/featurize/gnn").json()
    assert v["can_introspect"] is False and "Unpickling" in v["introspect_note"]
    assert c.post("/api/steps/featurize/gnn/introspect", json={}, headers={"X-MLTree": "1"}).status_code == 403
    assert c.get("/api/steps/train/gnn").json()["models"][0]["name"] == "MolGCN"
    assert c.get("/api/steps/nope/gnn").status_code == 404


async def test_mcp_gnn_summary(demo, monkeypatch):
    out = await call_ok("gnn_summary", {"step": "train"})
    assert out["models"][0]["summary"]["conv"] == {"GCNConv": 3}
    assert "train" in (await call_ok("gnn_summary"))["steps"]
    monkeypatch.setenv("MLTREE_ALLOW_RUN", "0")
    assert "MLTREE_ALLOW_RUN" in str(await call_err("gnn_summary", {"step": "featurize", "introspect": True}))


def test_scan_never_runs_code(snip):
    from mltree_mcp.scan import scan_project

    scan_project(snip, dry_run=True)  # GAT_SRC raises SystemExit if executed
    gnn.analyze_step(snip, "gat")


@pytest.mark.skipif(not Path(TG_PY).exists(), reason="needs a Python with torch + torch_geometric")
def test_real_introspection(demo):
    import subprocess

    for script in ("steps/load_smiles.py", "steps/featurize.py"):
        subprocess.run([TG_PY, script], cwd=demo.root, check=True, capture_output=True)
    res = gnn.introspect(demo, "featurize", python=TG_PY)
    s = res["files"]["data/processed/graphs.pt"]
    assert s["format"] == "pyg-collated" and s["num_graphs"] == 20 and s["node_features"] == 7
    assert s["labels"]["task"] == "regression" and s["loaded_with"] == "torch.load(weights_only=True)"
