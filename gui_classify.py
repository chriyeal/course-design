import sys
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette
import torchvision.transforms.functional as F
from PIL import Image

# 从训练脚本导入 build_model 和 classes（确保类别顺序一致）
from train_model import build_model, classes

# 类别名称（带中文翻译，顺序与训练脚本一致）
class_names = [
    "Daffodil(黄水仙)", "Snowdrop(雪花莲)", "Lily Valley(铃兰)", "Bluebell(风铃草)",
    "Crocus(番红花)", "Iris(鸢尾)", "Tigerlily(虎皮百合)", "Tulip(郁金香)",
    "Fritillary(贝母)", "Sunflower(向日葵)", "Daisy(雏菊)", "Colts' Foot(款冬花)",
    "Dandelion(蒲公英)", "Cowslip(黄花九轮草)", "Buttercup(毛茛)", "Windflower(银莲花)", "Pansy(三色堇)"
]

# 初始化模型
model = build_model()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))  # 兼容 CPU/GPU
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def set_label_color(label, rank):
    palette = label.palette()
    if rank == 0:
        # 第一名设置为黄色
        palette.setColor(QPalette.Window, QColor('yellow'))
    elif 1 <= rank <= 4:
        # 第二至第五名设置为浅绿色
        palette.setColor(QPalette.Window, QColor('#CCFFCC'))  # 浅绿色
    else:
        # 其他不设置颜色
        return
    label.setAutoFillBackground(True)
    label.setPalette(palette)


# -------------------------- PyQt5 GUI 类 --------------------------
class FlowerClassifierGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 主布局：左右结构
        main_layout = QHBoxLayout()

        # 左侧布局
        left_layout = QVBoxLayout()

        # 按钮：选择图片
        self.btn_select = QPushButton("选择花卉图片", self)
        self.btn_select.setStyleSheet("font-size: 18px; padding: 8px;")
        self.btn_select.clicked.connect(self.select_image)
        left_layout.addWidget(self.btn_select)

        # 显示图片
        self.label_image = QLabel(self)
        self.label_image.setFixedSize(300, 300)  # 固定显示大小
        self.label_image.setStyleSheet("border: 1px solid #ccc;")
        left_layout.addWidget(self.label_image)

        # 显示预测结果
        self.label_result = QLabel("等待预测...", self)
        self.label_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; padding: 10px;")
        left_layout.addWidget(self.label_result)

        # 显示类别索引映射
        self.label_mapping = QLabel(self)
        self.label_mapping.setStyleSheet("font-size: 14px; color: #666;")
        mapping_text = "类别索引映射:\n" + "\n".join([f"{i}: {name}" for i, name in enumerate(class_names)])
        self.label_mapping.setText(mapping_text)
        left_layout.addWidget(self.label_mapping)

        # 添加到主布局左边
        main_layout.addLayout(left_layout)

        # 右侧：概率列表
        self.prob_labels = []
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)   # 设置行间距为 10 像素
        right_layout.setContentsMargins(10, 20, 10, 20)

        title = QLabel("各花卉种类可能性:")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(title)

        for _ in range(len(class_names)):  # 预留位置用于显示所有类别的概率
            label = QLabel("--")
            label.setStyleSheet("font-size: 16px;font-weight: bold;")
            self.prob_labels.append(label)
            right_layout.addWidget(label)

        # 留点空间
        right_layout.addStretch()

        # 添加到主布局右边
        main_layout.addLayout(right_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("花卉分类器")
        self.resize(900, 600)  # 调整窗口宽度以适应两列内容

    def select_image(self):
        # 打开文件对话框，仅允许选择图片
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png)"
        )
        if not file_path:
            return

        # 加载并显示图片（缩放适配）
        pixmap = QPixmap(file_path)
        pixmap = pixmap.scaled(self.label_image.width(), self.label_image.height(), aspectRatioMode=1)
        self.label_image.setPixmap(pixmap)

        # 预处理：与训练一致
        image = Image.open(file_path).convert("RGB")
        image = F.resize(image, 256)
        image = F.center_crop(image, 224)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.unsqueeze(0).to(device)  # 增加 batch 维度

        # 模型预测
        with torch.no_grad():
            outputs = model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            pred_class = class_names[pred_idx.item()]

        # 显示结果（带置信度）
        self.label_result.setText(f"预测类别：{pred_class}\n置信度：{conf.item() * 100:.2f}%")

        # 更新右侧每个类别的概率 - 排序后显示
        probs_list = probs[0].cpu().numpy()
        sorted_indices = probs_list.argsort()[::-1]  # 获取按概率降序排列的索引
        sorted_probs = probs_list[sorted_indices]
        sorted_class_names = [class_names[i] for i in sorted_indices]

        for i, (name, prob) in enumerate(zip(sorted_class_names, sorted_probs)):
            label = self.prob_labels[i]
            label.setText(f"{name}: {prob * 100:.2f}%")
            if i < 5:
                set_label_color(label, i)
            else:
                set_label_color(label, -1)  # 不设置颜色


# -------------------------- 启动 GUI --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = FlowerClassifierGUI()
    gui.show()
    sys.exit(app.exec_())