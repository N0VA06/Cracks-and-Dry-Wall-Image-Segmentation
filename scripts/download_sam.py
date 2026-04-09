"""
scripts/download_sam.py
Downloads a SAM checkpoint from Meta's official URLs.

Usage:
  python scripts/download_sam.py              # default: vit_b (~375 MB)
  python scripts/download_sam.py --type vit_l # large  (~1.2 GB)
  python scripts/download_sam.py --type vit_h # huge   (~2.4 GB)
"""

import os
import sys
import argparse
import urllib.request

CHECKPOINTS = {
    "vit_b": {
        "url" : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "file": "sam_vit_b_01ec64.pth",
        "size": "375 MB",
    },
    "vit_l": {
        "url" : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "file": "sam_vit_l_0b3195.pth",
        "size": "1.2 GB",
    },
    "vit_h": {
        "url" : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "file": "sam_vit_h_4b8939.pth",
        "size": "2.4 GB",
    },
}

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
    bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
    sys.stdout.write(f"\r  [{bar}] {pct:3d}%  ({downloaded // 1_048_576} MB)")
    sys.stdout.flush()


def download_sam(model_type: str = "vit_b"):
    if model_type not in CHECKPOINTS:
        print(f"Unknown model type: {model_type}. Choose from {list(CHECKPOINTS)}")
        sys.exit(1)

    info = CHECKPOINTS[model_type]
    os.makedirs(SAVE_DIR, exist_ok=True)
    dest = os.path.join(SAVE_DIR, info["file"])

    if os.path.isfile(dest):
        print(f"Checkpoint already exists: {dest}")
        return dest

    print(f"Downloading SAM {model_type} ({info['size']}) ...")
    print(f"  URL  : {info['url']}")
    print(f"  Dest : {dest}")
    urllib.request.urlretrieve(info["url"], dest, reporthook)
    print(f"\nDone → {dest}")
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--type", default="vit_b", choices=list(CHECKPOINTS))
    args = p.parse_args()
    download_sam(args.type)
