"""Fetch the three labeled datasets into data/. Usage: python scripts/download_data.py [--trucks N]"""
import argparse
import json
import re
import subprocess
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
REPOS = {
    "EngineFaultDB": "https://github.com/Leo-Thomas/EngineFaultDB.git",
    "OBD-Dataset": "https://github.com/AbouAbdallah-Lounis/OBD-Dataset.git",
}
ENGINEAD_API = "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi:10.5683/SP3/TX13P1"
ENGINEAD_FILE = "https://borealisdata.ca/api/access/datafile/{}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trucks", type=int, default=25, help="EngineAD trucks to fetch, lowest numbers first (25 = all, ~6.4 GB)")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        if not (DATA / name).exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(DATA / name)], check=True)

    # curl, not urllib: it uses the system CA store (python.org builds on macOS ship without one)
    listing = subprocess.run(["curl", "-sL", "--fail", ENGINEAD_API], check=True, capture_output=True).stdout
    files = [f["dataFile"] for f in json.loads(listing)["data"]["latestVersion"]["files"]]
    files.sort(key=lambda d: int(re.search(r"truck_(\d+)", d["filename"]).group(1)))
    out = DATA / "EngineAD"
    out.mkdir(exist_ok=True)
    for d in files[: args.trucks]:
        dest = out / d["filename"]
        if dest.exists():
            continue
        part = dest.with_suffix(".part")
        subprocess.run(["curl", "-sL", "--fail", "-o", str(part), ENGINEAD_FILE.format(d["id"])], check=True)
        part.rename(dest)
        print("downloaded", dest.name)


if __name__ == "__main__":
    main()
