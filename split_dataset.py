import os
import random
from shutil import copy2

# 类别名称（与需求一致）
classes = [
    "Daffodil", "Snowdrop", "Lily Valley", "Bluebell",
    "Crocus", "Iris", "Tigerlily", "Tulip",
    "Fritillary", "Sunflower", "Daisy", "Colts' Foot",
    "Dandelion", "Cowslip", "Buttercup", "Windflower", "Pansy"
]
# 数据集根路径（根据实际修改）
data_root = r"F:\A.机器学习\课设\17flowers"
source_dir = data_root  # 原始图片路径

# 按类别序号划分图片（每类80张，共17类）
for class_idx, class_name in enumerate(classes, 1):
    # 计算每类图片的起始/结束编号
    start_idx = (class_idx - 1) * 80 + 1  # 第1类从1开始，第2类从81开始，依此类推
    end_idx = class_idx * 80  # 第1类到80，第2类到160，依此类推

    # 生成该类别所有图片名列表（如 image_0001.jpg）
    images = [f"image_{str(idx).zfill(4)}.jpg" for idx in range(start_idx, end_idx + 1)]

    # 创建训练集和验证集的类别子文件夹
    train_dir = os.path.join(data_root, "train", class_name)
    val_dir = os.path.join(data_root, "val", class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 随机打乱图片顺序并划分训练集（80%）和验证集（20%）
    random.shuffle(images)
    train_split = images[:64]  # 64张训练图片
    val_split = images[64:]  # 16张验证图片

    # 复制图片到对应文件夹
    for img in train_split:
        copy2(os.path.join(source_dir, img), train_dir)
    for img in val_split:
        copy2(os.path.join(source_dir, img), val_dir)

print("✅ 数据集按类别正确划分完成！train/val 文件夹已创建")
