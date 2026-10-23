"""Shareable single-file HTML export (export_map / `mltree-mcp export` / GET /api/export)."""
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

from mltree_mcp import export as export_mod
from mltree_mcp import server
from mltree_mcp.export import build_snapshot, export_html

from test_tools import call_ok

DEMO = Path(__file__).resolve().parents[1] / "examples" / "gnn_demo"
needs_bundle = pytest.mark.skipif(not (export_mod.EXPORT_DIR / "export.js").is_file(),
                                  reason="export bundle not built (npm run build in frontend/)")


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


def embedded(html: str) -> dict:
    m = re.search(r'<script id="mltree-data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "snapshot block missing"
    return json.loads(m.group(1))


def test_snapshot_contents(demo):
    snap = build_snapshot(demo)
    assert snap["format"] == "mltree-export/1" and snap["include_scripts"] is True
    assert [n["name"] for n in snap["pipeline"]["nodes"]] == ["load_smiles", "featurize", "split", "train", "evaluate"]
    assert len(snap["runs"]) == len(demo.list_runs(limit=1000)) and snap["metrics"] == demo.metric_names()
    cur = snap["scripts"]["train"]["current"]
    assert cur["path"] == "steps/train.py"
    assert snap["blobs"][cur["sha256"]] == (demo.root / "steps/train.py").read_text()
    for rid, ref in snap["run_scripts"].items():  # every run snapshot resolves to a blob
        assert snap["blobs"][ref["sha256"]] == demo.script(demo.get_run(rid)["step"], rid)["content"]
    assert snap["run_scripts"]
    assert str(demo.root) not in json.dumps(snap["pipeline"])  # no absolute server path in the map


@needs_bundle
def test_export_file(demo, tmp_path):
    res = export_html(demo, path=tmp_path / "out" / "demo.html")
    p = Path(res["path"])
    html = p.read_text()
    assert res["bytes"] == p.stat().st_size < 1_500_000
    assert res["steps"] == 5 and res["scripts_included"] and "source code" in res["note"]
    snap = embedded(html)
    assert snap["pipeline"]["project"] == "gnn_demo"
    assert "def " in "".join(snap["blobs"].values())
    # self-contained: no external scripts, styles or fetches
    assert not re.search(r'<script[^>]+src=|<link[^>]+stylesheet|https?://(?!www\.w3\.org)[^"\')\s]+\.(js|css)', html)
    assert html.lower().count("</script") == 2  # nothing inside can close the inline blocks early
    assert "MonacoEnvironment" not in html  # the lightweight code view replaces Monaco


@needs_bundle
def test_export_without_scripts(demo, tmp_path):
    res = export_html(demo, path=tmp_path, include_scripts=False)
    assert Path(res["path"]).name == "gnn_demo-mltree.html"
    snap = embedded(Path(res["path"]).read_text())
    assert snap["include_scripts"] is False and snap["blobs"] == {} and snap["run_scripts"] == {}
    assert "random.Random(seed)" not in Path(res["path"]).read_text()


@needs_bundle
def test_script_breakout_is_escaped(demo, tmp_path):
    (demo.root / "steps/train.py").write_text('X = "</script><script>alert(1)</script><!--"\n')
    html = Path(export_html(demo, path=tmp_path / "x.html")["path"]).read_text()
    assert html.lower().count("</script") == 2
    assert any("</script><script>alert(1)" in b for b in embedded(html)["blobs"].values())


def test_missing_bundle(demo, tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "EXPORT_DIR", tmp_path / "nope")
    with pytest.raises(Exception, match="not built"):
        export_html(demo, path=tmp_path / "x.html")


@needs_bundle
async def test_mcp_tool(demo):
    res = await call_ok("export_map", {})
    assert Path(res["path"]) == demo.root / "gnn_demo-mltree.html" and Path(res["path"]).is_file()
    res = await call_ok("export_map", {"path": "share/", "include_scripts": False})
    assert Path(res["path"]) == demo.root / "share" / "gnn_demo-mltree.html" and res["scripts_included"] is False
    assert "export_map" in server.INSTRUCTIONS


@needs_bundle
def test_cli(demo, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server.main(["export", "--project", str(demo.root)])
    assert (tmp_path / "gnn_demo-mltree.html").is_file()
    server.main(["export", "--project", str(demo.root), "-o", "a.html", "--no-scripts"])
    out = capsys.readouterr().out
    assert "no scripts" in out and embedded((tmp_path / "a.html").read_text())["blobs"] == {}


@needs_bundle
def test_viewer_download(demo):
    from starlette.testclient import TestClient

    from mltree_mcp.viewer import create_app

    r = TestClient(create_app(demo.root)).get("/api/export?scripts=0")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'filename="gnn_demo-mltree.html"' in r.headers["content-disposition"]
    assert embedded(r.text)["include_scripts"] is False


def _browser():
    pw = pytest.importorskip("playwright.sync_api")
    exe = next((p for p in ("/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser") if Path(p).exists()), None)
    return pw, exe


@needs_bundle
def test_renders_offline_in_headless_chrome(demo, tmp_path):
    pw, exe = _browser()
    path = Path(export_html(demo, path=tmp_path / "demo.html")["path"])
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no headless chrome: {exc}")
        page = browser.new_page(viewport={"width": 1400, "height": 850})
        problems, network = [], []
        page.on("console", lambda m: m.type == "error" and problems.append(m.text))
        page.on("pageerror", lambda e: problems.append(str(e)))
        page.on("request", lambda r: not r.url.startswith(("file:", "data:", "blob:")) and network.append(r.url))
        page.goto(path.as_uri())
        page.wait_for_selector(".react-flow__node", timeout=15000)
        assert page.locator(".react-flow__node").count() == 5
        assert page.title() == "gnn_demo · mltree"
        assert "snapshot" in page.locator(".live").inner_text()
        # step panel with code (lightweight view, no Monaco)
        page.locator(".react-flow__node[data-id='train']").click()
        page.wait_for_selector(".code-lite .src")
        assert "def " in page.locator(".code-lite .src").inner_text()
        assert page.locator("select[aria-label='Script version'] option").count() >= 2
        page.select_option("select[aria-label='Script version']", index=1)
        page.wait_for_selector(".note")
        # experiment tree
        page.click("text=Experiment tree")
        page.wait_for_timeout(500)
        assert page.locator(".react-flow__node").count() == len(demo.list_runs(limit=1000))
        # compare
        ids = [r["id"] for r in demo.list_runs(step="train", limit=3)]
        page.goto(path.as_uri() + "#/compare?ids=" + ",".join(ids))
        page.wait_for_selector("table.compare")
        page.wait_for_selector(".plot-lite svg rect")
        # PNG / SVG of the map from the Export menu
        page.click("text=Workflow map")
        page.wait_for_selector(".react-flow__node")
        for fmt in ("PNG", "SVG"):
            page.click("text=Export ▾")
            assert page.locator("text=Shareable HTML").count() == 0  # not offered inside an export
            with page.expect_download() as dl:
                page.click(f"text=Map as {fmt}")
            assert dl.value.suggested_filename == f"gnn_demo-map.{fmt.lower()}"
        browser.close()
    assert problems == []
    assert network == []


def test_runs_are_portable_and_keep_sweeps(demo):
    rid = demo.log_run("train", {"alpha": 0.5}, {"val_rmse": 0.8}, tags=["mltree-run"],
                       command=["/opt/venv/bin/python", str(demo.root / "steps" / "train.py"), "--alpha", "0.5"])["id"]
    snap = build_snapshot(demo)
    run = next(r for r in snap["runs"] if r["id"] == rid)
    assert run["command"] == ["python", "steps/train.py", "--alpha", "0.5"]
    assert str(demo.root) not in json.dumps(snap["runs"])
    sweeps = [r for r in snap["runs"] if r["kind"] == "sweep"]
    assert len(sweeps) == 1 and sweeps[0]["sweep"]["metric"] == "val_rmse"
    assert sum(r["sweep_id"] == sweeps[0]["id"] for r in snap["runs"]) == 3
