"""Multi-project viewer (0.8.0): ~/.mltree/projects.json, `view --projects`, /p/<slug>/ routing, open_map."""
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from mltree_mcp import Store, server
from mltree_mcp import projects as reg
from mltree_mcp.store import MLTreeError
from mltree_mcp.viewer import STATIC_DIR, create_multi_app


def _proj(root: Path, name: str, steps: int) -> Path:
    root.mkdir(parents=True)
    s = Store(root)
    s.init(name)
    for i in range(steps):
        s.add_step(f"s{i}")
    s.log_run("s0", {"a": 1}, {"m": 0.5})
    return root


@pytest.fixture
def two(tmp_path, monkeypatch):
    monkeypatch.setenv("MLTREE_HOME", str(tmp_path / "home"))
    a = _proj(tmp_path / "qsar", "qsar", 2)
    b = _proj(tmp_path / "other" / "qsar", "tox", 3)  # same folder name -> second slug gets a suffix
    return a, b


def test_registry_roundtrip(two, tmp_path):
    a, b = two
    assert reg.load_registry() == []
    reg.add_project(a)
    res = reg.add_project(b, name="Tox21 GNN")
    assert [p["path"] for p in res["projects"]] == [str(a), str(b)]
    reg.add_project(a)  # re-adding moves it, no duplicate
    assert [p["path"] for p in reg.load_registry()] == [str(b), str(a)]
    raw = json.loads(reg.registry_path().read_text())
    assert raw["projects"][0] == {"path": str(b), "name": "Tox21 GNN"}
    reg.remove_project("Tox21 GNN")
    assert [p["path"] for p in reg.load_registry()] == [str(a)]
    with pytest.raises(MLTreeError):
        reg.remove_project("nope")
    with pytest.raises(MLTreeError):
        reg.add_project(tmp_path / "missing")
    reg.registry_path().write_text(json.dumps([str(a), str(b)]))  # plain list of paths is fine too
    assert len(reg.load_registry()) == 2
    reg.registry_path().write_text("{oops")
    with pytest.raises(MLTreeError, match="Can't read"):
        reg.load_registry()


def test_resolve_projects_slugs(two, tmp_path):
    a, b = two
    items = reg.resolve_projects([str(a), {"path": str(b)}, str(a)])
    assert [i["slug"] for i in items] == ["qsar", "qsar-2"]
    assert reg.resolve_projects([{"path": str(b), "name": "Tox 21!"}])[0]["slug"] == "tox-21"
    with pytest.raises(MLTreeError, match="not found"):
        reg.resolve_projects([str(tmp_path / "nope")])
    with pytest.raises(MLTreeError, match="No projects"):
        reg.resolve_projects([])


def test_multi_app_routes(two):
    a, b = two
    c = TestClient(create_multi_app([str(a), str(b)]), follow_redirects=False)
    ps = c.get("/api/projects").json()["projects"]
    assert [(p["slug"], p["name"], p["steps"], p["runs"]) for p in ps] == [("qsar", "qsar", 2, 1), ("qsar-2", "tox", 3, 1)]
    assert ps[0]["url"] == "/p/qsar/"
    h = c.get("/api/health").json()
    assert h["multi"] and [p["root"] for p in h["projects"]] == [str(a), str(b)]
    assert c.get("/").headers["location"] == "/p/qsar/"
    assert c.get("/p/qsar-2").headers["location"] == "/p/qsar-2/"
    assert c.get("/p/nope").status_code == 404
    # each project has its own full API under /p/<slug>/
    assert c.get("/p/qsar/api/pipeline").json()["project"] == "qsar"
    assert len(c.get("/p/qsar-2/api/pipeline").json()["nodes"]) == 3
    assert c.get("/p/qsar-2/api/health").json()["root"] == str(b)
    assert c.get("/api/pipeline").status_code == 404  # no default project at the root
    # edits go to the right project, with the same guards
    r = c.put("/p/qsar-2/api/steps/s1/notes", json={"notes": "hi"}, headers={"X-MLTree": "1"})
    assert r.status_code == 200 and Store(b).get_step("s1")["notes"] == "hi"
    assert c.put("/p/qsar/api/steps/s1/notes", json={"notes": "x"}).status_code == 403
    assert c.post("/p/qsar/api/steps/s0/run", json={}, headers={"X-MLTree": "1"}).status_code == 403
    named = TestClient(create_multi_app([{"path": str(b), "name": "Tox21 GNN"}])).get("/api/projects").json()
    assert named["projects"][0]["name"] == "Tox21 GNN"  # a registry name wins over the store's project name
    if (STATIC_DIR / "index.html").is_file():
        page = c.get("/p/qsar/#/map")
        assert page.status_code == 200 and "<div id=\"root\">" in page.text
        asset = next((STATIC_DIR / "assets").glob("*.js")).name
        assert c.get(f"/assets/{asset}").status_code == 200


def test_uninitialized_project_is_listed(tmp_path):
    (tmp_path / "fresh").mkdir()
    c = TestClient(create_multi_app([str(tmp_path / "fresh")]))
    p = c.get("/api/projects").json()["projects"][0]
    assert p["initialized"] is False and "steps" not in p


def test_cli_projects_and_view_args(two, capsys, monkeypatch):
    a, b = two
    server.main(["projects", "add", str(a)])
    server.main(["projects", "add", str(b), "--name", "tox"])
    server.main(["projects", "list"])
    out = capsys.readouterr().out
    assert "2 project(s)" in out and str(b) in out and "tox" in out
    seen = {}
    monkeypatch.setattr("mltree_mcp.viewer.serve", lambda **kw: seen.update(kw))
    server.main(["view", "--projects"])
    assert [p["path"] for p in seen["projects"]] == [str(a), str(b)]
    server.main(["view", "--projects", str(b)])
    assert seen["projects"] == [str(b)]
    server.main(["view"])
    assert seen["projects"] is None
    server.main(["projects", "remove", str(a)])
    server.main(["projects", "remove", "tox"])
    with pytest.raises(SystemExit):
        server.main(["view", "--projects"])  # empty registry
    assert "projects add" in capsys.readouterr().err


def test_open_map_on_multi_viewer(two, monkeypatch):
    a, b = two
    monkeypatch.setenv("MLTREE_PROJECT", str(b))
    health = {"ok": True, "multi": True, "projects": [{"slug": "qsar", "root": str(a)}, {"slug": "qsar-2", "root": str(b)}]}
    monkeypatch.setattr(server, "_viewer_health", lambda url: health)
    r = server._open_map(start=False)
    assert r["url"].endswith("/p/qsar-2/") and "warning" not in r
    health["projects"] = health["projects"][:1]
    assert "projects add" in server._open_map(start=False)["warning"]
