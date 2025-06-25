import os
import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image

# 类别名称（与需求一致，定义为全局变量）
classes = [
    "Daffodil", "Snowdrop", "Lily Valley", "Bluebell",
    "Crocus", "Iris", "Tigerlily", "Tulip",
    "Fritillary", "Sunflower", "Daisy", "Colts' Foot",
    "Dandelion", "Cowslip", "Buttercup", "Windflower", "Pansy"
]


# -------------------------- 1. 构建模型（ResNet18 + 解冻部分层） --------------------------
def build_model(num_classes=17):
    # 使用新版API加载预训练模型
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 冻结部分特征层（只解冻后几层，便于小数据集微调）
    for param in model.parameters():
        param.requires_grad = False  # 先全部冻结

    # 解冻后三层卷积块和分类头
    for param in model.layer3.parameters():
        param.requires_grad = True  # 解冻layer3
    for param in model.layer4.parameters():
        param.requires_grad = True  # 解冻layer4
    for param in model.fc.parameters():
        param.requires_grad = True  # 解冻分类头

    # 替换分类头（减少维度，防止过拟合）
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),  # 减少全连接层维度
        nn.ReLU(),
        nn.Dropout(0.3),  # 增加Dropout防止过拟合
        nn.Linear(256, num_classes)
    )
    return model


# -------------------------- 2. 自定义数据集类（修正参数名） --------------------------
class OrderedFlowerDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_order=None):
        self.root_dir = root_dir  # 正确参数名：root_dir
        self.transform = transform
        self.class_order = class_order or classes  # 使用全局classes列表

        # 创建类别到索引的映射（按class_order顺序）
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_order)}

        # 收集所有图像路径和标签
        self.samples = []
        for cls in self.class_order:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue  # 跳过不存在的类别（防止误操作）
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------- 3. 训练/验证流程 --------------------------
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def val_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


# -------------------------- 4. 主训练逻辑 --------------------------
if __name__ == "__main__":
    # 数据集路径（与划分脚本一致）
    data_root = r"F:\A.机器学习\课设\17flowers"

    # 增强数据预处理（增加旋转、颜色抖动等）
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),  # 增加随机旋转
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 使用自定义数据集类加载数据（参数名修正为root_dir）
    train_dataset = OrderedFlowerDataset(
        root_dir=os.path.join(data_root, "train"),  # 正确参数名：root_dir
        transform=data_transform["train"],
        class_order=classes
    )
    val_dataset = OrderedFlowerDataset(
        root_dir=os.path.join(data_root, "val"),  # 正确参数名：root_dir
        transform=data_transform["val"],
        class_order=classes
    )

    # 构建 DataLoader（小批次适配集成显卡）
    batch_size = 4  # 集成显卡用更小批次
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    # 仅优化解冻的层（layer3、layer4和fc）
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,  # 小学习率防止过拟合
        weight_decay=0.01  # 权重衰减正则化
    )

    # 学习率调度器（移除verbose参数，修复兼容性问题）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # 训练参数
    num_epochs = 10  # 增加训练轮数
    best_acc = 0.0
    best_epoch = 0

    print(f"🔧 训练设备: {device} | 批次大小: {batch_size} | 类别数: {len(train_dataset.class_order)}")
    print("==================== 开始训练 ====================")

    for epoch in range(num_epochs):
        # 训练1轮
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        # 调整学习率
        scheduler.step(val_loss)

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "best_model.pth")
            print(f"📦 模型更新：Epoch {epoch + 1} 验证准确率 {val_acc:.2f}%")

        # 打印日志
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}%")
        print(f"  验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}%")
        print("-" * 50)

    print(f"==================== 训练结束 ===================")
    print(f"💯 最优验证准确率: {best_acc:.2f}% (Epoch {best_epoch}) | 模型已保存为 best_model.pth")
