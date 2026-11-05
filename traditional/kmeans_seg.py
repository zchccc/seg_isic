# -*- coding: utf-8 -*-
"""K-means 聚类分割（课件第4章）。

在 CIE Lab 色彩空间对像素做 K=2 聚类（L、a、b 三通道特征），
取 L 均值较低（更暗）的簇为病灶前景，再做形态学清理。
"""
import cv2
import numpy as np

from . import otsu


def segment(image_bgr, K=2, attempts=3):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    z = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(z, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # 前景 = L 通道均值更暗的簇
    fg = int(np.argmin(centers[:, 0]))
    mask = (labels.ravel() == fg).astype(np.uint8).reshape(lab.shape[:2])
    return otsu.postprocess(mask)
