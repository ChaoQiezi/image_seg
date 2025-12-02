# @Author  : ChaoQiezi
# @Time    : 2025/12/2 下午10:29
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: train.py

"""
This script is used to 训练模型
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision

from utils import pre_dataloader, AttentionUNet, model_train, VOCDataset


def main():
    # 准备
    config_path = '../config.yaml'

    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device(config['device'])  # 训练设备
    writer = SummaryWriter(config['dir']['log_dir'])  # 日志记录器

    # 数据集加载
    train_loader = pre_dataloader(config, mode='train',
                                  batch_size=config['batch_size'],
                                  num_workers=config['data']['num_workers'],
                                  )
    val_loader = pre_dataloader(config, mode='val',
                                batch_size=config['batch_size'],
                                num_workers=config['data']['num_workers'],
                                )

    # 实例化模型
    model = AttentionUNet(in_channels=config['in_channels'], out_channels=config['num_classes']).to(device)
    print('模型框架: \n', model)

    model_train(train_loader, val_loader, model, config=config, device=device, log_writer=writer)


# 模型训练
if __name__ == '__main__':
    main()
