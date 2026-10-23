"""Screenshots of the 0.6.0 features (data versions, compare curves, code diff) for the README.

    cp -r examples/gnn_demo /tmp/demo && python /tmp/demo/register_demo.py --reset
    mltree-mcp --project /tmp/demo view --allow-run &
    python scripts/screenshots_data.py --project /tmp/demo [--chrome /usr/bin/google-chrome]

Runs the demo's stdlib-only load_smiles..split steps (so their outputs get versions), then edits
data/molecules.csv in that copy so the map flags the changed input. Use a copy, not the repo demo.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

from screenshots_run import DOCS, api, wait_idle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--project", required=True, help="the demo copy the viewer is serving")
    ap.add_argument("--theme", default="light")
    ap.add_argument("--chrome", default=None)
    a = ap.parse_args()
    url, root = a.url, Path(a.project)
    for step in ("load_smiles", "featurize", "split"):
        api(url, f"/api/steps/{step}/run", {})
        wait_idle(url)
    csv = root / "data" / "molecules.csv"
    csv.write_text(csv.read_text() + "CCO,0.12\n")
    runs = api(url, "/api/runs?step=train&limit=100")
    runs = runs["runs"] if isinstance(runs, dict) else runs
    base = next(r["id"] for r in runs if "baseline" in r["tags"])
    deeper = next(r["id"] for r in runs if "gcn-2layer" in r["tags"])
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome) if a.chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 920}, device_scale_factor=2)
        page.goto(f"{url}/?theme={a.theme}#/map?step=load_smiles")
        page.wait_for_selector(".data-table")
        page.wait_for_timeout(800)
        page.screenshot(path=str(DOCS / "screenshot-data.png"))
        page.goto(f"{url}/?theme={a.theme}#/compare?ids={base},{deeper}")
        page.wait_for_selector(".linechart svg polyline")
        page.wait_for_selector("table.diff tr.add")
        page.wait_for_timeout(600)
        page.locator(".linechart").first.scroll_into_view_if_needed()
        page.screenshot(path=str(DOCS / "screenshot-curves.png"))
        page.get_by_role("button", name="Side by side").click()
        page.wait_for_selector("table.diff.split")
        page.locator(".diffcard").scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(path=str(DOCS / "screenshot-diff.png"))
        browser.close()
    for n in ("data", "curves", "diff"):
        print("wrote", DOCS / f"screenshot-{n}.png")


if __name__ == "__main__":
    main()
