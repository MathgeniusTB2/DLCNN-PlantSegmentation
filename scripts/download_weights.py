#!/usr/bin/env python3
"""Download the trained 115-class weights for the web app.

Fetches ``best.pt`` from the project's GitHub Release and places it at
``PlantDiseaseAPP/plant_disease/best.pt`` so the Django app serves the full
115-class PlantSeg detector instead of falling back to a pretrained COCO model.

Usage::

    python scripts/download_weights.py                 # latest release
    python scripts/download_weights.py --version v1.0  # specific tag

Uses only the standard library. If the asset has not been uploaded yet the
script explains what to do instead of failing silently.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = "MathgeniusTB2/DLCNN-PlantSegmentation"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases"
APP_WEIGHTS = Path(__file__).resolve().parent.parent / "PlantDiseaseAPP" / "plant_disease" / "best.pt"

# GitHub's API rejects requests without a User-Agent header.
HEADERS = {"User-Agent": "DLCNN-PlantSegmentation-download-weights/1.0"}

UPLOAD_HINT = (
    "Upload your trained weights before running this again:\n"
    f"  1. git clone https://github.com/{REPO}\n"
    "  2. gh release create v1.0 --title 'Trained weights' \\\n"
    "        PlantDiseaseAPP/plant_disease/best.pt#best.pt\n"
)


def latest_release(version: str | None) -> dict:
    req = urllib.request.Request(RELEASES_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        releases = json.load(resp)
    if not releases:
        sys.exit(f"No releases found for {REPO}.\n\n{UPLOAD_HINT}")
    if version:
        for rel in releases:
            if rel["tag_name"] == version:
                return rel
        sys.exit(f"Release tag '{version}' not found. Available: "
                 f"{[r['tag_name'] for r in releases]}")
    return releases[0]


def download(rel: dict) -> None:
    asset = next((a for a in rel.get("assets", []) if a["name"] == "best.pt"), None)
    if asset is None:
        sys.exit(
            f"Release '{rel['tag_name']}' has no 'best.pt' asset.\n\n{UPLOAD_HINT}"
        )
    url, size = asset["browser_download_url"], asset["size"]
    APP_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {size / 1e6:.1f} MB from {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as resp, \
            open(APP_WEIGHTS, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    print(f"Saved weights to {APP_WEIGHTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=None,
                        help="Release tag to fetch (default: latest)")
    args = parser.parse_args()
    download(latest_release(args.version))


if __name__ == "__main__":
    main()
