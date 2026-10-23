"""Screenshots of the 0.7.0 features (notes/tags, GNN model + dataset views) for the README.

    python examples/gnn_demo/register_demo.py --reset
    mltree-mcp --project examples/gnn_demo view --allow-run --python /path/to/python-with-torch_geometric &
    python scripts/screenshots_gnn.py [--url http://localhost:8765] [--chrome /usr/bin/google-chrome]

Runs load_smiles + featurize through the viewer (so data/processed/graphs.pt exists), then asks the viewer to
inspect it; that is the only code execution and it needs --allow-run.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"


def api(url: str, path: str, body: dict | None = None, method: str | None = None) -> dict:
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"X-MLTree": "1", "Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req) as r:  # noqa: S310 - local viewer
        return json.loads(r.read().decode())


def wait_idle(url: str, timeout: float = 180) -> None:
    end = time.time() + timeout
    while time.time() < end:
        a = api(url, "/api/active")
        if not a["runs"] and not a["sweeps"]:
            return
        time.sleep(0.3)
    raise TimeoutError("runs still active")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--theme", default="light")
    ap.add_argument("--chrome", default=None)
    a = ap.parse_args()
    url = a.url
    for step in ("load_smiles", "featurize"):
        api(url, f"/api/steps/{step}/run", {})
        wait_idle(url)
    api(url, "/api/steps/featurize/gnn/introspect", {})
    api(url, "/api/steps/train/notes", {"notes": "GCN baseline. Try hidden=128 + dropout 0.2 next."}, "PUT")
    runs = api(url, "/api/runs?step=train&limit=50")
    runs = runs["runs"] if isinstance(runs, dict) else runs
    for i, r in enumerate(runs[:4]):
        api(url, f"/api/runs/{r['id']}/tags", {"add": ["baseline"] if i == 0 else ["alpha-sweep"]}, "PUT")
    if runs:
        api(url, f"/api/runs/{runs[0]['id']}/notes", {"notes": "Reference run for the paper table."}, "PUT")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome) if a.chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1000}, device_scale_factor=2)
        page.goto(f"{url}/?theme={a.theme}#/map?step=featurize")
        page.wait_for_selector(".react-flow__node")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(DOCS / "screenshot-gnn-data.png"))
        page.goto(f"{url}/?theme={a.theme}#/map?step=train")
        page.reload()
        page.wait_for_selector(".react-flow__node")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(DOCS / "screenshot-gnn.png"))
        page.goto(f"{url}/?theme={a.theme}#/tree?metric=val_rmse&mode=min&tag=baseline")
        page.reload()
        page.wait_for_selector(".react-flow__node")
        page.wait_for_timeout(1200)
        if runs:
            page.locator(".run-node:not(.dim)").filter(has_text="baseline").first.click()
            page.wait_for_timeout(800)
        page.screenshot(path=str(DOCS / "screenshot-notes.png"))
        browser.close()
    for n in ("gnn-data", "gnn", "notes"):
        print("wrote", DOCS / f"screenshot-{n}.png")


if __name__ == "__main__":
    main()
