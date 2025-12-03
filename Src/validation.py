# @Author  : ChaoQiezi
# @Time    : 2025/12/3 上午5:49
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: validation.py

"""
This script is used to 验证+可视化
"""
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

from utils import pre_dataloader


def cal_metrics(prediction, target, num_classes, ignore_class=None):
    """
    依据预测值和真实值计算mIOU和Dice系数
    :param prediction:
    :param target:
    :param num_classes:
    :param ignore_class:
    :return:
    """

    # 忽略背景值
    if ignore_class is not None:
        mask = target != ignore_class
        prediction = prediction[mask]
        target = target[mask]

    # 计算混淆矩阵
    indices = target * num_classes + prediction  # 每一个值在混淆矩阵的索引
    m = torch.bincount(indices, minlength=num_classes ** 2).view(num_classes, num_classes)

    # 计算IOU
    intersection = torch.diag(m)  # 获取混淆矩阵的对角线值, 交集部分
    union = m.sum(dim=1) + m.sum(dim=0) - intersection  # 类的真实总数 + 类的预测总数 - 交集 (因为交集被加了两次)
    iou = intersection / (union + 1e-6)
    # 计算mIOU(IOU的均值)
    miou = iou.mean().item()

    # 计算Dice系数
    dice = (intersection * 2) / (m.sum(dim=1) + m.sum(dim=0) + 1e-6)  # (2 × 交集) / (真实区域总面积 + 预测区域总面积)
    dice = dice.mean().item()

    return miou, dice


def decode_segmap(raster_arr, template_path):
    """将单波段的语义分割结果转为RGB图片"""

    # 获取调色板(mode=P才可)
    with Image.open(template_path) as img:
        palette = img.palette.getdata()[1]

    # 创建伪彩色图像
    out_img = Image.fromarray(raster_arr, mode='P')
    out_img.putpalette(palette)

    return out_img


def visualize_comparison(images, targets, att_preds, unet_preds, config):
    # # 获取target-png图片的调色板(颜色映射表)
    # with Image.open(template_path) as img:
    #     palette = img.getpalette()
    #     palette = np.array(palette).reshape(-1, 3)
    #     """
    #     第0行的三列值表示像元值为0的RGB值
    #     第1行的三列值表示像元值为1的RGB值
    #     ···
    #     """

    # 绘制对比
    for cur_image, cur_target, cur_att_pred, cur_unet_pred in zip(images, targets, att_preds, unet_preds):
        # 对cur_image反标准化
        cur_image = cur_image * config['data']['rgb_std'] + config['data']['rgb_mean']

        # 将图像分割结果(真实+预测)转换为RGB图片
        cur_target = decode_segmap(cur_target, config['data']['template_path'])
        cur_att_pred = decode_segmap(cur_att_pred, config['data']['template_path'])
        cur_unet_pred = decode_segmap(cur_unet_pred, config['data']['template_path'])

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(cur_image)
        axes[0].set_title('Image')

        axes[1].imshow(cur_target)
        axes[1].set_title('Target')

        axes[2].imshow(cur_att_pred)
        axes[2].set_title('Attention Prediction')

        axes[3].imshow(cur_unet_pred)
        axes[3].set_title('U-Net Prediction')

        # 保存
        cur_out_path = os.path.join(config['dir']['chart_dir'], 'MyTemp.png')
        plt.savefig(cur_out_path, dpi=300)
        break


# 准备
config_path = '../config.yaml'

# 加载配置文件
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
device = torch.device(config['device'])

# 验证数据集加载
val_loader = pre_dataloader(config, mode='val',
                            batch_size=config['batch_size'],
                            num_workers=config['data']['num_workers'],
                            )
# 加载训练好的模型
att_model: nn.Module = torch.load(config['path'][f'attention_unet_path'])
unet_model: nn.Module = torch.load(config['path'][f'unet_path'])

# 验证
att_model.eval()  # 评估模式
unet_model.eval()
for images, targets in val_loader:
    # 预测
    images, targets = images.to(device), targets.to(device)
    with torch.no_grad():  # 禁用梯度计算
        att_preds = att_model(images)
        unet_preds = unet_model(images)

    # 定量结果(精度指标)
    att_miou, att_dice = cal_metrics(att_preds, targets, config['num_classes'], config['ignore_class'])
    unet_miou, unet_dice = cal_metrics(unet_preds, targets, config['num_classes'], config['ignore_class'])

    # 定性结果(图像分割结果)
    visualize_comparison(images, targets, att_preds, unet_preds, config)

