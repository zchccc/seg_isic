# -*- coding: utf-8 -*-
"""批量评估：4 种传统分割方法在官方验证集（data/ISIC2018/val, 100 张）上运行。

输出：
  experiments/traditional_metrics.csv  每张图像 × 每种方法的 Dice/IoU
  experiments/figures/traditional_samples.png  前若干张的可视化叠加对比
用法：python -m traditional.run_traditional
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from traditional import canny_edge, kmeans_seg, metrics, otsu, region_growing
from common_io import imread

DATA = ROOT / "data" / "ISIC2018"
EXP = ROOT / "experiments"
METHODS = {
    "Otsu": otsu.segment,
    "RegionGrowing": region_growing.segment,
    "KMeans": kmeans_seg.segment,
    "Canny": canny_edge.segment,
}
N_VIS = 5  # 可视化样张数量


def overlay(image_rgb, mask, color=(255, 0, 0), alpha=0.4):
    """在 RGB 图上以半透明色叠加掩膜轮廓。"""
    out = image_rgb.copy()
    m = mask > 0
    out[m] = (np.array(color) * alpha + out[m] * (1 - alpha)).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def main():
    img_dir, gt_dir = DATA / "val", DATA / "val_gt"
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    assert ids, "未找到验证集图像，请先运行 scripts/prepare_data.py"
    EXP.mkdir(exist_ok=True)
    (EXP / "figures").mkdir(exist_ok=True)

    rows, vis = [], []
    for i, name in enumerate(ids):
        bgr = imread(img_dir / f"{name}.jpg")
        gt = (imread(gt_dir / f"{name}_segmentation.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        entry = {"id": name}
        preds = {}
        for mname, fn in METHODS.items():
            t0 = time.time()
            pred = fn(bgr)
            entry[f"{mname}_dice"] = round(metrics.dice_coeff(pred, gt), 4)
            entry[f"{mname}_iou"] = round(metrics.iou_score(pred, gt), 4)
            entry[f"{mname}_ms"] = round((time.time() - t0) * 1000, 1)
            preds[mname] = pred
        rows.append(entry)
        if i < N_VIS:
            vis.append((rgb, gt, preds))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(ids)}")

    # 写 CSV
    out_csv = EXP / "traditional_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 汇总打印
    print("\n=== 传统方法在官方验证集（%d 张）上的平均指标 ===" % len(ids))
    for mname in METHODS:
        d = np.mean([r[f"{mname}_dice"] for r in rows])
        j = np.mean([r[f"{mname}_iou"] for r in rows])
        ms = np.mean([r[f"{mname}_ms"] for r in rows])
        print(f"{mname:15s} Dice={d:.4f}  IoU={j:.4f}  平均耗时={ms:.0f}ms")
    print(f"CSV 已保存: {out_csv}")

    # 可视化：每行 = 原图/GT叠加/各方法叠加
    n = len(vis)
    fig, axes = plt.subplots(n, 2 + len(METHODS), figsize=(3 * (2 + len(METHODS)), 3 * n))
    for r, (rgb, gt, preds) in enumerate(vis):
        panels = [("Image", rgb), ("GT", overlay(rgb, gt, (0, 255, 0)))] + [
            (m, overlay(rgb, preds[m], (255, 0, 0))) for m in METHODS
        ]
        for c, (title, panel) in enumerate(panels):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig_path = EXP / "figures" / "traditional_samples.png"
    fig.savefig(fig_path, dpi=120)
    print(f"可视化已保存: {fig_path}")


if __name__ == "__main__":
    main()
