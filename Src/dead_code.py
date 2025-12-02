import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
# 针对 RTX 4060 8GB 显存的优化配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8  # 若显存不足 (OOM)，请降为 4
NUM_WORKERS = 4  # 数据加载线程数
LR = 1e-4  # 学习率
EPOCHS = 20  # 训练轮数 (作业演示可设为 10-20)
IMAGE_SIZE = 256  # 图像统一缩放尺寸
NUM_CLASSES = 21  # PASCAL VOC: 20 前景 + 1 背景
IGNORE_INDEX = 255  # VOC 边界标签，计算 Loss 时忽略

print(f"Running on device: {DEVICE}")


# ==========================================
# 2. 数据集处理 (Dataset & Transforms)
# ==========================================
# 说明：PASCAL VOC 的 Mask 是调色板模式 (P mode)，
# 直接 ToTensor 会归一化到 ，破坏类别索引。
# 因此需要自定义 Target Transform。

def get_transforms(train=True):
    """
    自定义转换函数，确保图像和Mask进行相同的随机变换（如翻转）。
    """

    def joint_transform(image, target):
        # 1. Resize (图像和Mask都要缩放)
        # InterpolationMode.NEAREST 用于 Mask，保证类别索引不被插值改变
        resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        image = resize(image)
        target = resize(target, interpolation=transforms.InterpolationMode.NEAREST)

        # 2. 随机水平翻转 (仅训练时)
        if train:
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)

        # 3. 图像转 Tensor 并归一化 (ImageNet 标准均值方差)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 4. Mask 转 LongTensor (不缩放，保持 0-20 整数)
        target = np.array(target)
        target = torch.as_tensor(target, dtype=torch.long)

        # 将 Mask 中 255 的区域保留，Loss 函数会自动忽略
        return image, target

    return joint_transform


class VOCDatasetWrapper(torch.utils.data.Dataset):
    """
    包装 torchvision 的 VOCSegmentation，应用自定义变换
    """

    def __init__(self, root, image_set='train', train=True):
        self.voc = datasets.VOCSegmentation(
            root=root,
            year='2012',
            image_set=image_set,
            download=True,  # 自动下载数据集
            transform=None,  # 我们在 __getitem__ 中手动处理
            target_transform=None
        )
        self.transform = get_transforms(train=train)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, target = self.voc[idx]
        img, target = self.transform(img, target)
        return img, target


# ==========================================
# 3. 模型定义 (Attention U-Net)
# ==========================================

class ConvBlock(nn.Module):
    """
    U-Net 基础卷积块：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    注意力门控模块 (Attention Gate)
    原理：利用 gating signal (g) 对 skip connection (x) 进行加权筛选
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # g 的变换 (Decoder特征)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # x 的变换 (Encoder特征)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 生成注意力系数 alpha
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 归一化到
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 逐元素相加 -> ReLU -> 1x1 Conv -> Sigmoid
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 特征重校准
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(AttentionUNet, self).__init__()

        # --- Encoder (收缩路径) ---
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.enc5 = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Decoder (扩张路径) ---
        # Up4
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionGate(F_g=1024, F_l=512, F_int=256)  # g来自Bottleneck上采样, x来自enc4
        self.dec4 = ConvBlock(1024 + 512, 512)  # 拼接后通道数 = 1024(up) + 512(skip)

        # Up3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512 + 256, 256)

        # Up2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256 + 128, 128)

        # Up1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(F_g=128, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128 + 64, 64)

        # Final Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.enc5(p4)

        # Decoder with Attention
        # D4: Upsample Bottleneck -> Attention(g=d4, x=e4) -> Concat -> Conv
        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)  # Skip connection
        d4 = self.dec4(d4)

        # D3
        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)

        # D2
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)

        # D1
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out


# ==========================================
# 4. 训练与评估工具 (Utils)
# ==========================================

def calculate_iou(pred, target, num_classes=21):
    """
    计算 mIoU
    """
    pred = torch.argmax(pred, dim=1)
    # 忽略 index=255
    keep = target != IGNORE_INDEX
    pred = pred[keep]
    target = target[keep]

    # 快速计算混淆矩阵技巧
    # target * num_classes + pred 会生成唯一的索引
    indices = target * num_classes + pred
    m = torch.bincount(indices, minlength=num_classes ** 2).view(num_classes, num_classes)

    # IoU = TP / (TP + FP + FN)
    intersection = torch.diag(m)
    union = m.sum(0) + m.sum(1) - intersection
    iou = intersection / (union + 1e-6)
    return iou.mean().item()


def train_model():
    # 1. 准备数据
    print("Loading Dataset (This may take time to download if first run)...")
    train_ds = VOCDatasetWrapper(root='../Data', image_set='train', train=True)
    val_ds = VOCDatasetWrapper(root='../Data', image_set='val', train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # 2. 初始化模型
    model = AttentionUNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)

    # 3. 损失函数与优化器
    # CrossEntropyLoss 自带 ignore_index 功能
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度Scaler

    print("Start Training...")
    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0

        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # 简单验证 mIoU (每5个epoch或最后一个epoch)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            total_miou = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(DEVICE)
                    masks = masks.to(DEVICE)
                    outputs = model(imgs)
                    total_miou += calculate_iou(outputs, masks)
            print(f"Validation mIoU: {total_miou / len(val_loader):.4f}")

    # 保存模型
    torch.save(model.state_dict(), "attention_unet_voc.pth")
    print("Model Saved!")
    return model, val_ds


# ==========================================
# 5. 可视化结果 (Visualization)
# ==========================================
def visualize_prediction(model, dataset):
    model.eval()
    # 随机取一张图
    idx = np.random.randint(0, len(dataset))
    img, mask = dataset[idx]

    input_tensor = img.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu()

    # 反归一化以便显示图片
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    img_disp = inv_normalize(img).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)

    # 绘图
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax.imshow(img_disp)
    ax.set_title("Input Image")
    ax.axis('off')

    ax.imshow(mask, cmap='jet')
    ax.set_title("Ground Truth")
    ax.axis('off')

    ax.imshow(pred, cmap='jet')
    ax.set_title("Attention U-Net Prediction")
    ax.axis('off')

    plt.show()


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 如果已有下载好的数据，会自动使用；否则会自动下载
    # 确保 data 文件夹有写入权限
    os.makedirs('../Data', exist_ok=True)

    trained_model, validation_dataset = train_model()

    # 训练结束后展示效果
    visualize_prediction(trained_model, validation_dataset)