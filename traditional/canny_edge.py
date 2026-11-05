# -*- coding: utf-8 -*-
"""Canny 边缘检测 + 形态学闭合分割（课件第4章边缘检测：一阶/二阶导数与 Canny 步骤）。

流程：高斯平滑 → Canny 双阈值边缘 → 形态学闭运算连接断裂边缘
     → 轮廓填充取最大轮廓 → 孔洞填充。
"""
import cv2
import numpy as np

from . import otsu


def segment(image_bgr, low=40, high=120):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 从边缘重建区域：取最大外轮廓并填充
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, np.uint8)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 100:
            cv2.drawContours(mask, [biggest], -1, 1, thickness=cv2.FILLED)
    if mask.sum() < 100:      # 边缘重建失败时回退为阈值法，保证可比性
        t = otsu.otsu_threshold(gray)
        mask = (gray < t).astype(np.uint8)
    return otsu.postprocess(mask)
