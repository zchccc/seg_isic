# -*- coding: utf-8 -*-
"""Otsu 最大类间方差阈值分割（课件第4章，自实现公式，不调用 cv2.threshold 的 Otsu 模式）。

思路：遍历灰度阈值 T ∈ [0,255]，最大化类间方差 σ_b² = ω0·ω1·(μ0-μ1)²。
皮肤病灶通常比周围皮肤暗，取灰度 < T 作为前景；
再经形态学开运算去噪点、保留最大连通域、孔洞填充得到最终结果。
"""
import cv2
import numpy as np


def otsu_threshold(gray):
    """返回使类间方差最大的阈值（0-255）。"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = gray.size
    p = hist / total
    omega = np.cumsum(p)                       # ω0(T)
    mu_t = np.cumsum(p * np.arange(256))       # 累积均值
    mu_all = mu_t[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_all * omega - mu_t) ** 2 / (omega * (1.0 - omega) + 1e-12)
    sigma_b2[(omega == 0) | (omega == 1)] = -1
    return int(np.argmax(sigma_b2))


def postprocess(mask):
    """开运算去噪 + 最大连通域 + 孔洞填充。输入输出均为 {0,1}。"""
    mask = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # 最大连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    # 孔洞填充（泛洪法）
    h, w = mask.shape
    flood = mask.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 2)
    mask = ((mask == 1) | (flood == 0)).astype(np.uint8)
    return mask


def segment(image_bgr):
    """输入 BGR 彩色图，输出 {0,1} 分割掩膜。"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)   # 预平滑抑制毛发/噪声
    t = otsu_threshold(gray)
    mask = (gray < t).astype(np.uint8)         # 病灶偏暗
    return postprocess(mask)
