# -*- coding: utf-8 -*-
"""生成报告用图（experiments/figures/）：

1. methods_bar.png      各方法 Dice/IoU 条形对比图（传统4法 + U-Net基线/改进）
2. compare_grid.png     示例图像六宫格：原图 / GT / Otsu / KMeans / 基线U-Net / 改进U-Net
3. train_curves.png     U-Net 训练损失与验证 Dice 曲线（读取 train_log_*.csv）

依赖各阶段产物齐全后运行：python scripts/make_report_figs.py
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unet"))
EXP = ROOT / "experiments"
FIG = EXP / "figures"
DATA = ROOT / "data" / "ISIC2018"

from common_io import imread

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TRAD_METHODS = ["Otsu", "RegionGrowing", "KMeans", "Canny"]
TRAD_LABELS = {"Otsu": "Otsu阈值", "RegionGrowing": "区域生长", "KMeans": "K-means聚类", "Canny": "Canny边缘"}


def load_traditional():
    """从 traditional_metrics.csv 汇总传统方法均值。"""
    path = EXP / "traditional_metrics.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for m in TRAD_METHODS:
        out[m] = {
            "dice": float(np.mean([float(r[f"{m}_dice"]) for r in rows])),
            "iou": float(np.mean([float(r[f"{m}_iou"]) for r in rows])),
        }
    return out


def load_unet_summaries():
    """读取 unet_metrics_<tag>_val.csv（含 __MEAN__ 行），汇总各实验组均值。"""
    out = {}
    for p in EXP.glob("unet_metrics_*_val.csv"):
        tag = p.stem.replace("unet_metrics_", "").replace("_val", "")
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("image") == "__MEAN__":
                    out[tag] = {"dice_mean": float(row["dice"]), "iou_mean": float(row["iou"])}
    return out


def fig_bar(trad, unet):
    labels, dices, ious = [], [], []
    for m in TRAD_METHODS:
        if m in trad:
            labels.append(TRAD_LABELS[m])
            dices.append(trad[m]["dice"]); ious.append(trad[m]["iou"])
    tag_labels = {"baseline": "U-Net基线", "improved": "U-Net改进"}
    for tag in ["baseline", "improved"]:
        if tag in unet:
            labels.append(tag_labels[tag])
            dices.append(unet[tag]["dice_mean"]); ious.append(unet[tag]["iou_mean"])
    if not labels:
        print("缺少指标数据，跳过条形图"); return

    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, dices, w, label="Dice")
    b2 = ax.bar(x + w / 2, ious, w, label="IoU")
    ax.bar_label(b1, fmt="%.3f", fontsize=8); ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("得分"); ax.legend()
    ax.set_title("ISIC 2018 验证集（100张）各分割方法指标对比")
    fig.tight_layout(); fig.savefig(FIG / "methods_bar.png", dpi=150); plt.close(fig)
    print(f"已生成 {FIG / 'methods_bar.png'}")


def overlay(rgb, mask, color=(255, 0, 0), alpha=0.35):
    out = rgb.copy(); m = mask > 0
    out[m] = (np.array(color) * alpha + out[m] * (1 - alpha)).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 1)
    return out


def fig_grid(unet_tags):
    """六宫格：原图/GT/Otsu/KMeans/基线/改进。U-Net 预测直接从权重实时计算。"""
    import torch
    from unet import UNet
    from traditional import otsu, kmeans_seg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for t in unet_tags:
        ckpt = EXP / "checkpoints" / f"best_{t}.pth"
        if ckpt.exists():
            m = UNet(n_channels=3, n_classes=1, bilinear=False)
            m.load_state_dict(torch.load(ckpt, map_location=device))
            models[t] = m.to(device).eval()
    ids = sorted(p.stem for p in (DATA / "val").glob("*.jpg"))[:3]
    if not ids:
        print("缺少验证集图像，跳过对比网格"); return
    tag_labels = {"baseline": "U-Net基线", "improved": "U-Net改进"}
    cols = ["原图", "GT", "Otsu阈值", "K-means聚类"] + [tag_labels.get(t, t) for t in models]
    fig, axes = plt.subplots(len(ids), len(cols), figsize=(3 * len(cols), 3 * len(ids)))
    for r, name in enumerate(ids):
        bgr = imread(DATA / "val" / f"{name}.jpg")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gt = imread(DATA / "val_gt" / f"{name}_segmentation.png", cv2.IMREAD_GRAYSCALE)
        panels = [rgb, overlay(rgb, gt, (0, 200, 0)),
                  overlay(rgb, otsu.segment(bgr)), overlay(rgb, kmeans_seg.segment(bgr))]
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255).unsqueeze(0).to(device)
        for t in models:
            with torch.no_grad():
                pred = (torch.sigmoid(models[t](x)) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
            panels.append(overlay(rgb, pred, (255, 128, 0)))
        for c in range(len(cols)):
            ax = axes[r, c] if len(ids) > 1 else axes[c]
            if c < len(panels):
                ax.imshow(panels[c])
            ax.set_title(cols[c], fontsize=10); ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / "compare_grid.png", dpi=150); plt.close(fig)
    print(f"已生成 {FIG / 'compare_grid.png'}")


def fig_curves():
    logs = {p.stem.replace("train_log_", ""): p for p in EXP.glob("train_log_*.csv")}
    if not logs:
        print("缺少训练日志，跳过训练曲线"); return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag, path in logs.items():
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ep = [int(r["epoch"]) for r in rows]
        ax1.plot(ep, [float(r["train_loss"]) for r in rows], marker="o", ms=3, label=tag)
        ax2.plot(ep, [float(r["val_dice"]) for r in rows], marker="o", ms=3, label=tag)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("训练损失"); ax1.legend(); ax1.set_title("训练损失曲线")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("内部验证 Dice"); ax2.legend(); ax2.set_title("验证 Dice 曲线")
    fig.tight_layout(); fig.savefig(FIG / "train_curves.png", dpi=150); plt.close(fig)
    print(f"已生成 {FIG / 'train_curves.png'}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    trad = load_traditional()
    unet = load_unet_summaries()
    fig_bar(trad, unet)
    fig_grid([t for t in ["baseline", "improved"] if t in unet])
    fig_curves()
    print("全部图表生成完毕。")


if __name__ == "__main__":
    main()
