# -*- coding: utf-8 -*-
"""数据准备：由 256px 缩略图 + 官方掩膜 zip 构建 ISIC 2018 Task 1 数据集。

输入：
  data/train_thumb/ISIC_*.jpg   训练图像缩略图（256px，download_thumbs.py 下载）
  data/val_thumb/ISIC_*.jpg     验证图像缩略图
  data/train_gt.zip             官方训练掩膜（原始分辨率 PNG）
  data/val_gt.zip               官方验证掩膜

输出（data/ISIC2018/，统一 256×256，掩膜 {0,255}）：
  train/, train_gt/          训练集 90%
  inner_val/, inner_val_gt/  训练集 10%（内部验证、早停用）
  val/, val_gt/              官方验证集 100 张（最终评估用）

用法：python scripts/prepare_data.py
"""
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "ISIC2018"
SIZE = (256, 256)
INNER_VAL_RATIO = 0.1
SEED = 42


def load_masks(zip_path):
    """从官方 GT zip 读取 {ISIC_id: PIL 掩膜}。"""
    masks = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("_segmentation.png"):
                isic_id = Path(name).stem.replace("_segmentation", "")
                masks[isic_id] = Image.open(io.BytesIO(z.read(name)))
    return masks


def save_sample(img, mask, name, split):
    img_dir, mask_dir = OUT / split, OUT / f"{split}_gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    img = img.convert("RGB").resize(SIZE, Image.BILINEAR)
    mask = mask.convert("L").resize(SIZE, Image.NEAREST).point(lambda v: 255 if v >= 128 else 0)
    img.save(img_dir / f"{name}.jpg", quality=95)
    mask.save(mask_dir / f"{name}_segmentation.png")


def build(thumb_dir, gt_zip, split_names):
    """split_names: dict，isic_id -> split 名。返回处理/缺失数量。"""
    masks = load_masks(gt_zip)
    ok, missing = 0, []
    for isic_id, split in split_names.items():
        thumb = thumb_dir / f"{isic_id}.jpg"
        if not thumb.exists() or isic_id not in masks:
            missing.append(isic_id)
            continue
        with Image.open(thumb) as im:
            img = im.copy()
        save_sample(img, masks[isic_id], isic_id, split)
        ok += 1
    return ok, missing


def main():
    random.seed(SEED)

    # 训练集：按 9:1 随机切内部验证
    train_ids = sorted(p.stem for p in (DATA / "train_thumb").glob("*.jpg"))
    random.shuffle(train_ids)
    n_inner = int(len(train_ids) * INNER_VAL_RATIO)
    split_map = {i: ("inner_val" if idx < n_inner else "train") for idx, i in enumerate(train_ids)}
    ok, miss = build(DATA / "train_thumb", DATA / "train_gt.zip", split_map)
    print(f"train+inner_val: {ok} 张，缺失 {len(miss)} 个 {miss[:5] if miss else ''}")

    # 官方验证集
    val_ids = sorted(p.stem for p in (DATA / "val_thumb").glob("*.jpg"))
    ok, miss = build(DATA / "val_thumb", DATA / "val_gt.zip", {i: "val" for i in val_ids})
    print(f"val: {ok} 张，缺失 {len(miss)} 个")

    # 校验数量一致性
    for split in ["train", "inner_val", "val"]:
        n_img = len(list((OUT / split).glob("*.jpg")))
        n_msk = len(list((OUT / f"{split}_gt").glob("*.png")))
        assert n_img == n_msk and n_img > 0, f"{split} 数量异常: img={n_img} mask={n_msk}"
        print(f"{split}: {n_img} 对 图像/掩膜 ✓")

    # 掩膜对齐抽查：保存 3 组叠加图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = sorted(p.stem for p in (OUT / "val").glob("*.jpg"))[:3]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, ids):
        img = np.array(Image.open(OUT / "val" / f"{name}.jpg"))
        msk = np.array(Image.open(OUT / "val_gt" / f"{name}_segmentation.png"))
        over = img.copy()
        edge = np.zeros_like(msk)
        edge[1:, :] = (np.abs(msk[1:, :].astype(int) - msk[:-1, :].astype(int)) > 0)
        over[edge > 0] = [255, 0, 0]
        ax.imshow(over); ax.set_title(name); ax.axis("off")
    fig.tight_layout()
    qc = ROOT / "experiments" / "figures"
    qc.mkdir(parents=True, exist_ok=True)
    fig.savefig(qc / "qc_overlay.png", dpi=120)
    print(f"抽查叠加图: {qc / 'qc_overlay.png'}")


if __name__ == "__main__":
    main()
