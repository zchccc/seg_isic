# -*- coding: utf-8 -*-
"""跨平台图像读写：解决 Windows 中文路径下 cv2.imread/imwrite 失败的问题。

cv2.imread 对含非 ASCII 字符的路径（如本机 "管理员"）会返回 None，
因此统一改用 np.fromfile + cv2.imdecode / cv2.imencode + tofile。
项目内所有图像读写都应经过本模块。
"""
from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """中文路径安全的 imread。失败返回 None。"""
    path = str(Path(path))
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite(path, img, params=None):
    """中文路径安全的 imwrite。返回是否成功。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(str(path))
    return True
