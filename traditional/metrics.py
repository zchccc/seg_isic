# -*- coding: utf-8 -*-
"""统一评估指标：Dice 系数与 IoU（Jaccard）。

传统方法与 U-Net 共用本实现，保证指标口径一致。
预测与真值均为 bool/0-1 二值数组，前景=1。
"""
import numpy as np


def _check(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    assert pred.shape == gt.shape
    return pred, gt


def dice_coeff(pred, gt):
    """Dice = 2|A∩B| / (|A|+|B|)，两者全空时定义为 1。"""
    pred, gt = _check(pred, gt)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def iou_score(pred, gt):
    """IoU = |A∩B| / |A∪B|，两者全空时定义为 1。"""
    pred, gt = _check(pred, gt)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def evaluate_pair(pred, gt):
    """一次计算 Dice 与 IoU，返回 dict（供统一评估脚本使用）。"""
    return {"dice": dice_coeff(pred, gt), "iou": iou_score(pred, gt)}
