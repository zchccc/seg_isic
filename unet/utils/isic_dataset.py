# -*- coding: utf-8 -*-
"""修改点1：ISIC 2018 数据加载（替代原项目 BasicDataset 的 .tif/Carvana 逻辑）。

- 读取 JPG 图像 + {id}_segmentation.png 二值掩膜，掩膜统一为 {0,1} float。
- augment=True 时做随机增强：水平/垂直翻转、±15° 旋转、0.9–1.1 缩放、亮度/对比度扰动。
  图像与掩膜使用同一组随机参数，几何变换同步施加。
"""
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, images_dir, mask_dir, augment: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.ids = sorted(p.stem for p in self.images_dir.glob("*.jpg"))
        assert self.ids, f"在 {images_dir} 未找到 jpg 图像"

    def __len__(self):
        return len(self.ids)

    def _augment(self, img: Image.Image, mask: Image.Image):
        # 几何变换：图与掩膜同参数
        if random.random() < 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() < 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.9, 1.1)
        img = TF.affine(img, angle=angle, translate=(0, 0), scale=scale, shear=0,
                        interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, angle=angle, translate=(0, 0), scale=scale, shear=0,
                         interpolation=TF.InterpolationMode.NEAREST, fill=0)
        # 颜色扰动：仅图像
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        return img, mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(self.images_dir / f"{name}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{name}_segmentation.png").convert("L")

        if self.augment:
            img, mask = self._augment(img, mask)

        img = TF.to_tensor(img)                                   # [3,H,W] float 0-1
        mask = (TF.to_tensor(mask) > 0.5).float().squeeze(0)      # [H,W] {0,1}
        return {"image": img, "mask": mask}
