"""Screenshots of the 0.5.0 run features (Run dialog, live log panel, sweep fan) for the README.

    python examples/gnn_demo/register_demo.py --reset
    mltree-mcp --project examples/gnn_demo view --allow-run &
    python scripts/screenshots_run.py [--url http://localhost:8765] [--chrome /usr/bin/google-chrome]

This really executes the demo's stdlib-only scripts (the point is to show live runs).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"


def api(url: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"X-MLTree": "1", "Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:  # noqa: S310 - local viewer
        return json.loads(r.read().decode())


def wait_idle(url: str, timeout: float = 120) -> None:
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
    # upstream steps produce the files evaluate/train read
    for step in ("load_smiles", "featurize", "split", "train"):
        api(url, f"/api/steps/{step}/run", {})
        wait_idle(url)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome) if a.chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 920}, device_scale_factor=2)
        page.goto(f"{url}/?theme={a.theme}#/map?step=evaluate")
        page.wait_for_selector(".react-flow__node")
        page.get_by_role("button", name="▶ Run…").click()
        page.wait_for_selector(".modal pre.cmd")
        page.wait_for_timeout(600)
        page.screenshot(path=str(DOCS / "screenshot-run-dialog.png"))
        page.locator(".modal-foot input[type=checkbox]").check()
        page.locator(".modal-foot button.primary").click()
        page.wait_for_selector(".watch .badge.status-succeeded", timeout=60000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(DOCS / "screenshot-run-log.png"))

        page.goto(f"{url}/?theme={a.theme}#/map?step=train")
        page.reload()
        page.wait_for_selector(".react-flow__node")
        page.get_by_role("button", name="Grid sweep…").click()
        page.wait_for_selector(".modal .form-table")
        row = page.locator(".modal .form-table tbody tr").filter(has=page.locator("input[value=alpha]"))
        row.locator("td").nth(2).locator("input").fill("0.05, 0.2, 0.5, 1, 2, 4")
        page.locator(".sweep-opts input").fill("train_rmse")
        page.get_by_role("button", name="lower is better").click()
        page.wait_for_timeout(400)
        page.screenshot(path=str(DOCS / "screenshot-sweep-dialog.png"))
        page.locator(".modal-foot input[type=checkbox]").check()
        page.locator(".modal-foot button.primary").click()
        page.wait_for_selector(".sweep-watch .badge.status-succeeded", timeout=120000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(DOCS / "screenshot-sweep-log.png"))

        page.goto(f"{url}/?theme={a.theme}#/tree?metric=val_rmse&mode=min")
        page.reload()
        page.wait_for_selector(".react-flow__node")
        page.wait_for_timeout(1200)
        page.screenshot(path=str(DOCS / "screenshot-sweep-tree.png"))
        browser.close()
    for n in ("run-dialog", "run-log", "sweep-dialog", "sweep-log", "sweep-tree"):
        print("wrote", DOCS / f"screenshot-{n}.png")


if __name__ == "__main__":
    main()
