# ISIC 2018 皮肤病变分割：传统方法 vs 改进 U-Net

图像处理课程大作业。在 ISIC 2018 Task 1（皮肤镜病灶边界分割）数据集上：

- **传统方法（手写，对照组）**：Otsu 最大类间方差阈值、区域生长、K-means 聚类（Lab 空间）、Canny 边缘 + 形态学闭合
- **深度方法（基于 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) 修改）**：
  - 修改点1 `unet/utils/isic_dataset.py`：ISIC JPG+PNG 掩膜数据加载（替代原 .tif 加载）
  - 修改点2 `unet/train_isic.py`：训练脚本改造——去 wandb 改 CSV 日志、Adam 优化器、复合损失 `0.5×BCE+0.5×Dice`、数据增强、内部验证早停
  - 修改点3 数据增强：随机翻转 / ±15° 旋转 / 0.9–1.1 缩放 / 亮度对比度扰动（图像与掩膜同步几何变换）
  - 修改点4 `scripts/evaluate_all.py`：统一评估脚本，与传统方法共用同一 Dice/IoU 实现

## 环境

- Python 3.14 + PyTorch 2.14 (CUDA 可用) / OpenCV / NumPy / Matplotlib / Pillow
- `pip install -r requirements.txt`

## 数据

ISIC 2018 Task 1（训练 2594 张 + 验证 100 张，含分割掩膜）。官方源 `https://challenge.isic-archive.com/data/#2018`。
为规避官方 S3 直连限速，本项目经两条轻量通道获取：
- 掩膜：官方小体积 GT zip（训练 26 MB / 验证 0.7 MB），保存为 `data/train_gt.zip`、`data/val_gt.zip`
- 图像：ISIC Archive 官方 256px 缩略图接口逐张并行下载（`scripts/download_thumbs.py`）

```bash
python scripts/download_thumbs.py   # 下载 256px 缩略图到 data/train_thumb、data/val_thumb
python scripts/prepare_data.py      # 掩膜 zip 解压 + 统一 256×256 + 9:1 切内部验证
```

生成：`data/ISIC2018/{train,train_gt,inner_val,inner_val_gt,val,val_gt}`（2335 / 259 / 100 张）

## 复现

```bash
# 传统对照组（官方验证集 100 张）
python -m traditional.run_traditional        # → experiments/traditional_metrics.csv + figures/traditional_samples.png

# U-Net 基线（纯 BCE、无增强）
python unet/train_isic.py --tag baseline --loss bce
# U-Net 改进组（复合损失 + 数据增强）
python unet/train_isic.py --tag improved --loss composite --augment

# 统一评估（输出逐图 CSV 与均值，指标与传统方法同一实现）
python scripts/evaluate_all.py --tag baseline
python scripts/evaluate_all.py --tag improved

# 报告图表：条形图 / 六宫格对比 / 训练曲线
python scripts/make_report_figs.py           # → experiments/figures/
```

## 目录

```
seg_isic/
├── data/ISIC2018/          # 数据集（不进库）
├── traditional/            # 手写传统算法 + 批量评估
├── unet/                   # Pytorch-UNet + 上述修改点
├── scripts/                # 数据准备 / 报告图表
└── experiments/            # 日志、CSV、图表、权重（权重不进库）
```
