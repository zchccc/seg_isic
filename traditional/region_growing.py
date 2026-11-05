# -*- coding: utf-8 -*-
"""区域生长分割（课件第4章）。

种子点：图像中心 1/9 区域内灰度最低（最暗）的像素——皮肤镜病灶一般居中且偏暗。
生长准则：邻域像素与当前区域均值的灰度差 < 阈值 T 则并入（8 邻域）。
停止：无新像素加入。之后复用 otsu.postprocess 做形态学清理。
"""
import cv2
import numpy as np

from . import otsu


def region_growing(gray, seed=None, T=15, max_iter=100000):
    """对灰度图做区域生长，返回 {0,1} 掩膜。"""
    h, w = gray.shape
    if seed is None:
        cy, cx = h // 3, w // 3
        center = gray[cy: h - cy, cx: w - cx]
        idx = np.unravel_index(np.argmin(center), center.shape)
        seed = (idx[0] + cy, idx[1] + cx)

    mask = np.zeros((h, w), np.uint8)
    mask[seed] = 1
    region_sum = float(gray[seed])
    region_n = 1
    frontier = [seed]
    for _ in range(max_iter):
        if not frontier:
            break
        y, x = frontier.pop()
        mean = region_sum / region_n
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                    if abs(int(gray[ny, nx]) - mean) < T:
                        mask[ny, nx] = 1
                        region_sum += int(gray[ny, nx])
                        region_n += 1
                        frontier.append((ny, nx))
    return mask


def segment(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = region_growing(gray)
    if mask.sum() < 100:          # 生长失败（种子落在正常皮肤）时放宽阈值重试
        mask = region_growing(gray, T=25)
    return otsu.postprocess(mask)
