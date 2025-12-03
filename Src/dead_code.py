# @FileName: evaluation.py
# @Author  : ChaoQiezi & Co-Pilot
# @Purpose : 加载训练好的模型，生成定量指标对比表和定性效果图

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import pre_dataloader, AttentionUNet, StandardUNet

# ================= 配置区域 =================
# 请替换为你训练好的模型路径
UNET_PATH = 'logs/StandardUNet_20251202_xxxx/best_model.pth'
ATT_UNET_PATH = 'logs/AttentionUNet_20251202_xxxx/best_model.pth'
CONFIG_PATH = '../config.yaml'
SAVE_DIR = './results_visualization'


# ===========================================

def calculate_metrics(pred, target, num_classes=21):
    """计算Batch的mIoU和Dice"""
    pred = torch.argmax(pred, dim=1)
    # 忽略背景/边界 (假设255是忽略)
    keep = target != 255
    pred = pred[keep]
    target = target[keep]

    # 混淆矩阵
    indices = target * num_classes + pred
    m = torch.bincount(indices, minlength=num_classes ** 2).view(num_classes, num_classes)

    # IoU
    intersection = torch.diag(m)
    union = m.sum(0) + m.sum(1) - intersection
    iou = intersection / (union + 1e-6)
    miou = iou.mean().item()

    # Dice
    dice = 2 * intersection / (m.sum(0) + m.sum(1) + 1e-6)
    mdice = dice.mean().item()

    return miou, mdice


def visualize_comparison(loader, model_unet, model_att, device, save_dir):
    """生成定性对比图：Input | GT | U-Net | Attention U-Net"""
    model_unet.eval()
    model_att.eval()

    # 以此颜色映射VOC类别 (简化版)
    def decode_segmap(image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
        return rgb

    images, masks = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        out_u = model_unet(images)
        out_a = model_att(images)

    pred_u = torch.argmax(out_u, dim=1).cpu().numpy()
    pred_a = torch.argmax(out_a, dim=1).cpu().numpy()
    masks = masks.numpy()

    # 反归一化图片以便显示
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    os.makedirs(save_dir, exist_ok=True)

    # 绘制前4张图
    plt.figure(figsize=(12, 10))
    for i in range(4):
        # Image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(4, 4, i * 4 + 1)
        plt.imshow(img)
        plt.title("Input Image") if i == 0 else None
        plt.axis('off')

        plt.subplot(4, 4, i * 4 + 2)
        plt.imshow(decode_segmap(masks[i]))
        plt.title("Ground Truth") if i == 0 else None
        plt.axis('off')

        plt.subplot(4, 4, i * 4 + 3)
        plt.imshow(decode_segmap(pred_u[i]))
        plt.title("Standard U-Net") if i == 0 else None
        plt.axis('off')

        plt.subplot(4, 4, i * 4 + 4)
        plt.imshow(decode_segmap(pred_a[i]))
        plt.title("Attention U-Net") if i == 0 else None
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_result.png'), dpi=300)
    print(f"Visualization saved to {os.path.join(save_dir, 'comparison_result.png')}")


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])

    # 加载数据
    val_loader = pre_dataloader(config, mode='val', batch_size=config['batch_size'], num_workers=2)

    # 加载模型
    print("Loading Models...")
    try:
        unet = StandardUNet(config['in_channels'], config['num_classes']).to(device)
        unet.load_state_dict(torch.load(UNET_PATH, map_location=device))

        att_unet = AttentionUNet(config['in_channels'], config['num_classes']).to(device)
        att_unet.load_state_dict(torch.load(ATT_UNET_PATH, map_location=device))
    except FileNotFoundError:
        print("Error: 请先修改脚本顶部的模型路径 (UNET_PATH, ATT_UNET_PATH)！")
        return

    # 1. 定量评估
    print("Running Quantitative Evaluation...")
    u_miou_list, u_dice_list = [], []
    a_miou_list, a_dice_list = [], []

    unet.eval()
    att_unet.eval()

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            # UNet
            out_u = unet(imgs)
            miou_u, dice_u = calculate_metrics(out_u, masks)
            u_miou_list.append(miou_u)
            u_dice_list.append(dice_u)

            # Att UNet
            out_a = att_unet(imgs)
            miou_a, dice_a = calculate_metrics(out_a, masks)
            a_miou_list.append(miou_a)
            a_dice_list.append(dice_a)

    print("\n" + "=" * 40)
    print(f"Standard U-Net | mIoU: {np.mean(u_miou_list):.4f} | Dice: {np.mean(u_dice_list):.4f}")
    print(f"Attention U-Net | mIoU: {np.mean(a_miou_list):.4f} | Dice: {np.mean(a_dice_list):.4f}")
    print("=" * 40 + "\n")

    # 2. 定性可视化
    print("Running Qualitative Visualization...")
    visualize_comparison(val_loader, unet, att_unet, device, SAVE_DIR)


if __name__ == '__main__':
    main()