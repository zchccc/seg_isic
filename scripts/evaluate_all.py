# -*- coding: utf-8 -*-
"""修改点4：统一评估脚本。加载训练好的 U-Net 权重，在官方验证集上评估，
指标使用 traditional/metrics.py 的同一实现，保证与传统方法口径一致。

用法（在 seg_isic/ 下）：python scripts/evaluate_all.py --tag baseline
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unet"))

from traditional.metrics import evaluate_pair  # 同一指标实现
from common_io import imread, imwrite
from unet import UNet

DATA = ROOT / "data" / "ISIC2018"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="val", choices=["val", "inner_val"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    model.load_state_dict(torch.load(ROOT / "experiments" / "checkpoints" / f"best_{args.tag}.pth",
                                     map_location=device))
    model.eval()

    img_dir, gt_dir = DATA / args.split, DATA / f"{args.split}_gt"
    rows = []
    vis_dir = ROOT / "experiments" / f"unet_vis_{args.tag}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, ip in enumerate(sorted(img_dir.glob("*.jpg"))):
        gt = imread(gt_dir / f"{ip.stem}_segmentation.png", cv2.IMREAD_GRAYSCALE)
        img = imread(ip)
        if gt is None or img is None:
            continue
        x = torch.from_numpy(img[:, :, ::-1].copy().transpose(2, 0, 1)).float().div_(255).unsqueeze(0).to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
        gt01 = (gt > 127).astype(np.uint8)
        m = evaluate_pair(pred, gt01)
        rows.append({"image": ip.stem, "dice": round(m["dice"], 4), "iou": round(m["iou"], 4)})
        if i < 10:
            vis_dir_img = np.hstack([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), np.repeat((pred * 255)[:, :, None], 3, 2)])
            imwrite(vis_dir / f"{ip.stem}.png", cv2.cvtColor(vis_dir_img, cv2.COLOR_RGB2BGR))

    mean = {"image": "__MEAN__",
            "dice": round(float(np.mean([r["dice"] for r in rows])), 4),
            "iou": round(float(np.mean([r["iou"] for r in rows])), 4)}
    rows.append(mean)
    out = ROOT / "experiments" / f"unet_metrics_{args.tag}_{args.split}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "dice", "iou"])
        w.writeheader()
        w.writerows(rows)
    print(f"[{args.tag} @ {args.split}] Dice={mean['dice']:.4f} IoU={mean['iou']:.4f}  (n={len(rows)-1})")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
