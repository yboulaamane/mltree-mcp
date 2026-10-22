"""Headless screenshots of the viewer for the README.

    python examples/gnn_demo/register_demo.py --reset
    mltree-mcp view --project examples/gnn_demo &
    python scripts/screenshots.py [--url http://localhost:8765] [--theme light|dark]

Needs `pip install playwright` and either `playwright install chromium` or a local Chrome.
"""
import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--theme", default="light")
    ap.add_argument("--chrome", default=None, help="path to a Chrome/Chromium binary (else Playwright's)")
    a = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    runs = json.load(urllib.request.urlopen(a.url + "/api/runs?step=train"))["runs"]
    ids = ",".join([r["id"] for r in runs if r["status"] == "succeeded"][:4])
    shots = {
        "screenshot-map.png": "/#/map?step=train",
        "screenshot-tree.png": "/#/tree?metric=val_rmse&mode=min",
        "screenshot-compare.png": f"/#/compare?ids={ids}",
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome) if a.chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 920}, device_scale_factor=2)
        for name, path in shots.items():
            page.goto(f"{a.url}/?theme={a.theme}{path.lstrip('/')}")
            page.reload()
            page.wait_for_selector(".react-flow__node, table.compare", timeout=15000)
            if "map" in name:
                page.wait_for_selector(".monaco-editor .view-lines", timeout=15000)
            if "compare" in name:
                page.wait_for_selector(".plot .main-svg", timeout=15000)
            page.wait_for_timeout(900)
            page.screenshot(path=str(DOCS / name))
            print("wrote", DOCS / name)
        browser.close()


if __name__ == "__main__":
    main()
