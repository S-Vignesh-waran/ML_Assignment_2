"""
Dataset download helper for Breast Cancer Wisconsin dataset from Kaggle.

Before running:
1. Download kaggle.json from Kaggle account.
2. Place kaggle.json in the project root directory.
3. Run this script from the project root.
"""

import os
import zipfile


KAGGLE_DATASET = "uciml/breast-cancer-wisconsin-data"
ZIP_FILE = "breast-cancer-wisconsin-data.zip"
DATA_DIR = "data"


def prepare_kaggle_config():
    os.system("mkdir -p ~/.kaggle")
    os.system("cp kaggle.json ~/.kaggle/")
    os.system("chmod 600 ~/.kaggle/kaggle.json")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.system(f"kaggle datasets download -d {KAGGLE_DATASET}")


def extract_dataset():
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print(f"Dataset extracted to {DATA_DIR}")


if __name__ == "__main__":
    prepare_kaggle_config()
    download_dataset()
    extract_dataset()