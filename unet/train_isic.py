# -*- coding: utf-8 -*-
"""修改点2：ISIC 训练脚本（基于原 train.py 改造）。

主要修改：
- 移除 wandb 依赖，改为 CSV 训练日志（epochs, loss, val_dice），供报告绘制曲线；
- 优化器 RMSprop -> Adam(lr=1e-4)；n_classes=1 二分割；
- --loss bce|composite：baseline 用纯 BCE，改进组用 0.5*BCE+0.5*Dice（修改点：复合损失）；
- --augment：开启数据增强（改进组）；
- 内部验证集早停 + 保存最佳权重。

用法（在 seg_isic/ 下）：
  python unet/train_isic.py --tag baseline  --loss bce
  python unet/train_isic.py --tag improved  --loss composite --augment
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unet import UNet
from utils.isic_dataset import ISICDataset
from utils.dice_score import dice_loss, dice_coeff

ROOT = Path(__file__).resolve().parent.parent  # seg_isic/
DATA = ROOT / "data" / "ISIC2018"
CKPT = ROOT / "experiments" / "checkpoints"
LOG = ROOT / "experiments"


def evaluate_dice(model, loader, device):
    """在整个验证集上的平均 Dice（硬阈值 0.5）。"""
    model.eval()
    score, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, dtype=torch.float32)
            mask = batch["mask"].to(device, dtype=torch.float32)
            pred = (torch.sigmoid(model(img).squeeze(1)) > 0.5).float()
            score += dice_coeff(pred, mask, reduce_batch_first=True).item()
            n += 1
    model.train()
    return score / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="实验名，用于日志/权重命名")
    ap.add_argument("--loss", choices=["bce", "composite"], default="bce")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6, help="早停容忍轮数")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device={device}, tag={args.tag}, loss={args.loss}, augment={args.augment}")

    train_set = ISICDataset(DATA / "train", DATA / "train_gt", augment=args.augment)
    val_set = ISICDataset(DATA / "inner_val", DATA / "inner_val_gt", augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    logging.info(f"train={len(train_set)}, inner_val={len(val_set)}")

    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3, factor=0.5)
    bce = nn.BCEWithLogitsLoss()

    CKPT.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"train_log_{args.tag}.csv"
    best_dice, bad_epochs = -1.0, 0
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "lr", "seconds"])

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            epoch_loss, nb = 0.0, 0
            for batch in train_loader:
                img = batch["image"].to(device, dtype=torch.float32)
                mask = batch["mask"].to(device, dtype=torch.float32)
                pred = model(img).squeeze(1)
                loss = bce(pred, mask)
                if args.loss == "composite":  # 修改点：复合损失
                    loss = 0.5 * loss + 0.5 * dice_loss(torch.sigmoid(pred), mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                nb += 1

            val_dice = evaluate_dice(model, val_loader, device)
            scheduler.step(val_dice)
            lr = optimizer.param_groups[0]["lr"]
            epoch_loss /= max(nb, 1)
            writer.writerow([epoch, round(epoch_loss, 5), round(val_dice, 5), lr, round(time.time() - t0, 1)])
            f.flush()
            logging.info(f"epoch {epoch}: loss={epoch_loss:.4f} val_dice={val_dice:.4f} lr={lr:g}")

            if val_dice > best_dice:
                best_dice, bad_epochs = val_dice, 0
                torch.save(model.state_dict(), CKPT / f"best_{args.tag}.pth")
                logging.info(f"  保存最佳权重 (dice={best_dice:.4f})")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    logging.info(f"早停于 epoch {epoch}，最佳 val_dice={best_dice:.4f}")
                    break

    logging.info(f"完成。最佳 inner_val Dice = {best_dice:.4f}，权重: {CKPT}/best_{args.tag}.pth")


if __name__ == "__main__":
    main()
