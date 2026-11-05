# -*- coding: utf-8 -*-
"""按掩膜文件名中的 ISIC ID，从 S3 并行下载 256px 缩略图。

输出：data/train_thumb/*.jpg、data/val_thumb/*.jpg
用法：python scripts/download_thumbs.py
"""
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TPL = "https://isic-archive.s3.amazonaws.com/thumbnails/{}_thumbnail.jpg"


def ids_from_zip(zip_name: str, suffix: str = "_segmentation.png"):
    with zipfile.ZipFile(DATA / zip_name) as z:
        return [Path(n).stem.replace("_segmentation", "")
                for n in z.namelist() if n.endswith(suffix)]


def fetch(isic_id: str, out_dir: Path, retries: int = 3):
    out = out_dir / f"{isic_id}.jpg"
    if out.exists() and out.stat().st_size > 0:
        return isic_id, True
    for _ in range(retries):
        try:
            with urllib.request.urlopen(TPL.format(isic_id), timeout=30) as r:
                content = r.read()
            if len(content) > 500:
                out.write_bytes(content)
                return isic_id, True
        except Exception:
            pass
    return isic_id, False


def run(ids, out_dir: Path, workers: int = 16):
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, i, out_dir): i for i in ids}
        for n, fut in enumerate(as_completed(futs), 1):
            _, success = fut.result()
            ok, fail = ok + success, fail + (not success)
            if n % 200 == 0 or n == len(ids):
                print(f"{out_dir.name}: {n}/{len(ids)} 完成 (失败 {fail})")
    return fail


def main():
    fail = 0
    fail += run(ids_from_zip("train_gt.zip"), DATA / "train_thumb")
    fail += run(ids_from_zip("val_gt.zip"), DATA / "val_thumb")
    print(f"全部完成，失败 {fail} 个")


if __name__ == "__main__":
    main()
