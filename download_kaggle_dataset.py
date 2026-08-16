"""
Dataset download helper for Breast Cancer Wisconsin dataset from Kaggle.

Recommended setup:
1. Go to Kaggle -> Account -> Create New API Token.
2. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json.
3. Run: chmod 600 ~/.kaggle/kaggle.json
4. Run: python download_kaggle_dataset.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "uciml/breast-cancer-wisconsin-data"
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(DATA_DIR),
        "--unzip",
    ]
    print(f"Downloading Kaggle dataset: {DATASET_SLUG}")
    print("Destination:", DATA_DIR)
    subprocess.run(command, check=True)
    print("Download complete. Expected file:", DATA_DIR / "data.csv")


if __name__ == "__main__":
    main()

