"""Multi-project screenshot + the README demo GIF (0.8.0).

    mltree-mcp projects add /tmp/qsar-egfr --name "QSAR EGFR"     # two registered copies of examples/gnn_demo
    mltree-mcp projects add /tmp/tox21-gnn --name "Tox21 GNN"
    mltree-mcp view --projects &
    python scripts/screenshots_projects.py [--url http://localhost:8765] [--chrome /usr/bin/google-chrome]

Writes docs/screenshot-projects.png and docs/demo.gif (needs ffmpeg on PATH). Read-only: runs nothing.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

DOCS = Path(__file__).resolve().parents[1] / "docs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--theme", default="light")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--width", type=int, default=960, help="GIF width in px")
    a = ap.parse_args()
    url = a.url
    projects = json.load(urllib.request.urlopen(url + "/api/projects"))["projects"]  # noqa: S310 - local viewer
    if len(projects) < 2:
        raise SystemExit("start the viewer with `view --projects` and at least two projects")
    first, second, second_slug = projects[0]["url"], projects[1]["url"], projects[1]["slug"]
    runs = json.load(urllib.request.urlopen(f"{url}{first}api/runs?step=train"))["runs"]  # noqa: S310
    best = [r["id"] for r in runs if r["status"] == "succeeded"][:4]
    tmp = Path(tempfile.mkdtemp())
    frames: list[tuple[Path, float]] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=a.chrome) if a.chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 860}, device_scale_factor=1)

        def frame(hold: float) -> None:
            page.wait_for_timeout(700)
            f = tmp / f"f{len(frames):03d}.png"
            page.screenshot(path=str(f))
            frames.append((f, hold))

        # 1. workflow map, then the step panel
        page.goto(f"{url}{first}?theme={a.theme}#/map")
        page.wait_for_selector(".react-flow__node")
        frame(1.6)
        page.locator(".react-flow__node[data-id='train']").click()
        page.wait_for_selector(".monaco-editor .view-lines", timeout=15000)
        frame(2.4)
        # 2. experiment tree, ranked by val_rmse, pick the two best runs
        page.goto(f"{url}{first}?theme={a.theme}#/tree?metric=val_rmse&mode=min")
        page.reload()
        page.wait_for_selector(".run-node")
        frame(1.6)
        page.locator(".run-node.best").first.click()
        frame(2.0)
        # 3. compare the best runs of train
        page.goto(f"{url}{first}?theme={a.theme}#/compare?ids={','.join(best)}")
        page.reload()
        page.wait_for_selector(".plot .main-svg", timeout=15000)
        frame(2.6)
        # 4. switch project from the header
        page.goto(f"{url}{first}?theme={a.theme}#/map")
        page.wait_for_selector(".react-flow__node")
        page.wait_for_timeout(600)
        page.screenshot(path=str(DOCS / "screenshot-projects.png"))
        frame(1.2)
        page.select_option("select.project-switch", value=second_slug)
        page.wait_for_url(f"**{second}**")
        page.wait_for_selector(".react-flow__node")
        frame(2.0)
        browser.close()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(f"wrote {DOCS / 'screenshot-projects.png'}; install ffmpeg for the GIF (frames in {tmp})")
    concat = tmp / "frames.txt"
    lines = [f"file '{f}'\nduration {d}" for f, d in frames] + [f"file '{frames[-1][0]}'"]
    concat.write_text("\n".join(lines) + "\n")
    vf = f"scale={a.width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", vf, "-loop", "0", str(DOCS / "demo.gif")], check=True)
    print("wrote", DOCS / "screenshot-projects.png", "and", DOCS / "demo.gif")


if __name__ == "__main__":
    main()
