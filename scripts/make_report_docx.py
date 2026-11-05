# -*- coding: utf-8 -*-
"""生成大作业报告 docx（四节结构：任务/数据/代码修改/结果对比）。"""
import csv
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
FIG = EXP / "figures"
OUT = ROOT.parent / "【图像处理】-01234567-姓名" / "【图像处理】-01234567-姓名" / "1.报告.docx"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")

def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
    return p

def para(text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    return p

def fig(path, caption, width_cm=15):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in c.runs:
        r.font.size = Pt(9)
        r.italic = True

def read_metrics():
    with open(EXP / "traditional_metrics.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trad = {}
    for m, cn in [("Otsu", "Otsu 阈值"), ("RegionGrowing", "区域生长"), ("KMeans", "K-means 聚类"), ("Canny", "Canny 边缘")]:
        trad[cn] = (
            sum(float(r[f"{m}_dice"]) for r in rows) / len(rows),
            sum(float(r[f"{m}_iou"]) for r in rows) / len(rows),
        )
    unet = {}
    for tag, cn in [("baseline", "U-Net 基线"), ("improved", "U-Net 改进")]:
        with open(EXP / f"unet_metrics_{tag}_val.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["image"] == "__MEAN__":
                    unet[cn] = (float(r["dice"]), float(r["iou"]))
    return trad, unet

# ============ 标题 ============
title = doc.add_heading("图像处理大作业报告：基于传统方法与改进 U-Net 的皮肤病变图像分割", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in title.runs:
    r.font.color.rgb = RGBColor(0, 0, 0)
para("学号：01234567    姓名：（请填写）    日期：2026-11-06").alignment = WD_ALIGN_PARAGRAPH.CENTER
para("GitHub 仓库：（请填写推送后的链接，如 https://github.com/xxx/seg_isic）").alignment = WD_ALIGN_PARAGRAPH.CENTER

# ============ 1 任务描述 ============
h("1  任务描述")
para("本任务研究皮肤镜图像中病灶区域的自动分割问题：输入一张皮肤镜 RGB 图像，输出与病灶区域对应的二值掩膜。")
para("围绕课程第 4 章（传统图像分割）与第 4-2 章（AI 图像分割）内容，完成两条技术路线并在同一测试集上统一对比：")
para("（1）传统方法：手写实现 Otsu 最大类间方差阈值、区域生长、K-means 聚类（CIE Lab 空间）、Canny 边缘检测+形态学闭合四种分割算法；")
para("（2）深度方法：基于开源项目 Pytorch-UNet（GitHub: milesial/Pytorch-UNet）进行改进，训练 U-Net 分割模型，改进点包括复合损失函数与数据增强等。")
para("所有方法均以 Dice 系数与 IoU（Jaccard 指数）为评价指标，在官方验证集（100 张）上定量对比，并结合可视化结果分析各类方法的优缺点。")

# ============ 2 数据描述 ============
h("2  数据描述")
para("采用 ISIC 2018 Challenge Task 1（Lesion Boundary Segmentation）公开数据集。图像为皮肤镜 RGB 图像，真值掩膜由多种技术生成并经专业皮肤科医生审核。")
para("数据规模与划分：")
para("· 训练集 2594 张（含分割真值掩膜），本任务按 9:1 随机划分为内部训练集 2335 张与内部验证集 259 张（用于训练监控与早停）；")
para("· 官方验证集 100 张（含真值掩膜），作为全部方法的最终对比测试集；")
para("· 官方测试集 1000 张不公开真值，故未使用。")
para("预处理：全部图像与掩膜统一缩放至 256×256（图像双线性插值、掩膜最近邻插值），掩膜二值化为 {0,255}。")
para("数据来源：ISIC 官方渠道（https://challenge.isic-archive.com/data/#2018）；因官方 S3 直连速度受限，实际获取途径为 ISIC Archive 官方 256px 缩略图接口（图像）+ 官方 Ground Truth 压缩包（掩膜），二者编号一一对应（ISIC_xxxxxxx.jpg / ISIC_xxxxxxx_segmentation.png）。")
fig(FIG / "qc_overlay.png", "图1  数据质量抽查：真值掩膜边界（红线）与病灶轮廓对齐情况示例")

# ============ 3 代码修改描述 ============
h("3  代码修改描述")
h("3.1  自写传统分割算法（对照组）", 2)
para("以下算法均为手写实现（traditional/ 目录），仅依赖 NumPy/OpenCV 基础操作：")
para("· Otsu 阈值（otsu.py）：按课件公式自实现最大类间方差准则遍历 256 级灰度求最优阈值，未调用库函数 Otsu 模式；取暗区域为前景，经形态学开运算、最大连通域提取与孔洞填充得到掩膜。")
para("· 区域生长（region_growing.py）：以图像中心 1/9 区域最暗像素为自动种子，按“与区域均值灰度差小于阈值”准则进行 8 邻域生长，失败时放宽阈值重试。")
para("· K-means 聚类（kmeans_seg.py）：在 CIE Lab 色彩空间对像素做 K=2 聚类，取 L 均值较暗的簇为病灶。")
para("· Canny 边缘（canny_edge.py）：高斯平滑→Canny 双阈值边缘→形态学闭运算连接→最大轮廓填充；边缘重建失败时回退阈值法。")
para("· 统一评估（run_traditional.py + metrics.py）：在官方验证集上批量运行并输出逐图 Dice/IoU 与可视化叠加图。")
h("3.2  基于 Pytorch-UNet 的修改（深度组）", 2)
para("基线为开源项目 milesial/Pytorch-UNet（MIT 许可）原始配置（BCE 损失、无数据增强），在此基础上完成以下修改：")
para("（1）数据加载适配（unet/utils/isic_dataset.py，新增）：实现 ISICDataset，按 ISIC 命名规则配对读取 JPG 图像与 PNG 掩膜（替代原项目 .tif 数据假设），掩膜二值化为 {0,1}；")
para("（2）训练脚本改造（unet/train_isic.py，新增）：去除 wandb 依赖改为 CSV 日志，采用 Adam 优化器与“验证指标平台期则学习率减半”的调度，内部验证集早停并保存最佳权重；")
para("（3）复合损失函数：将单一 BCE 改为 0.5×BCEWithLogits + 0.5×Dice Loss，使优化目标直接关联重叠度指标；")
para("（4）数据增强：随机水平/垂直翻转、±15° 旋转、0.9–1.1 缩放（图像与掩膜同步几何变换）及亮度/对比度扰动（仅图像），提升泛化能力；")
para("（5）统一评估脚本（scripts/evaluate_all.py，新增）：加载最佳权重在官方验证集上推理，与传统方法共用同一 metrics.py 指标实现，保证对比口径一致。")
para("训练配置：输入 256×256、batch 8、Adam（lr=1e-4）、30 epoch、RTX 4060 Laptop GPU，单次训练约 50 分钟。")

# ============ 4 结果对比 ============
h("4  结果对比")
trad, unet = read_metrics()
para("各方法在官方验证集（100 张）上的平均指标如下表（Dice 与 IoU 越高越好）：")

table = doc.add_table(rows=1, cols=4)
table.style = "Table Grid"
hdr = table.rows[0].cells
for i, t in enumerate(["方法", "类别", "Dice", "IoU"]):
    hdr[i].text = t
    hdr[i].paragraphs[0].runs[0].bold = True
rows_data = [(k, "传统方法", f"{v[0]:.4f}", f"{v[1]:.4f}") for k, v in trad.items()]
rows_data += [(k, "深度方法", f"{v[0]:.4f}", f"{v[1]:.4f}") for k, v in unet.items()]
for name, cat, d, j in rows_data:
    cells = table.add_row().cells
    cells[0].text, cells[1].text, cells[2].text, cells[3].text = name, cat, d, j

para("")
fig(FIG / "methods_bar.png", "图2  各方法 Dice/IoU 条形对比图")
para("结果分析：")
para("（1）传统方法整体明显弱于 U-Net。Otsu 阈值（0.607）与 K-means 聚类（0.610）在病灶与皮肤对比度高的样例上表现尚可，但遇到病灶颜色接近正常皮肤、毛发遮挡或视野边缘暗角时容易大面积误分；区域生长（0.450）对种子位置与生长阈值敏感，易发生区域泄漏；Canny 边缘（0.177）依赖边缘连续性，病灶边界模糊时无法形成闭合轮廓，失败率最高。")
para("（2）U-Net 基线取得 Dice 0.8611 / IoU 0.7758，显著优于全部传统方法（Dice 提升约 25 个百分点），体现了编码器-解码器结构与跳跃连接在医学图像分割上的优势。")
para(f"（3）改进组（复合损失 + 数据增强）取得 Dice {unet['U-Net 改进'][0]:.4f} / IoU {unet['U-Net 改进'][1]:.4f}，较基线分别提升 {unet['U-Net 改进'][0]-unet['U-Net 基线'][0]:.4f} 与 {unet['U-Net 改进'][1]-unet['U-Net 基线'][1]:.4f}。数据增强使内部验证集上的收敛指标略有下降（0.859 vs 0.870），但在独立官方验证集上反超基线，说明改进主要带来的是泛化能力的提升。")
fig(FIG / "compare_grid.png", "图3  典型样例对比（从左至右：原图、真值、Otsu、K-means、U-Net 基线、U-Net 改进）")
fig(FIG / "train_curves.png", "图4  U-Net 训练损失与内部验证 Dice 曲线")
para("综上，本任务完整复现并对比了课程第 4 章传统分割算法与第 4-2 章深度学习分割方法，实验结果验证了深度方法在复杂皮肤镜图像分割中的显著优势，以及复合损失与数据增强对模型泛化能力的正向作用。")

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"报告已生成: {OUT}")
