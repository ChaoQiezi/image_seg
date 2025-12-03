# @Author  : ChaoQiezi
# @Time    : 2025/12/2 下午10:06
# @Email   : chaoqiezi.one@qq.com
# @Wechat  : GIS茄子
# @FileName: utils.py

"""
This script is used to 存放常用工具
"""

import yaml
from torch.utils.data import DataLoader, Dataset  # 数据集和加载器
from torchvision.datasets import VOCSegmentation  # VOC数据集(用于图像分割)
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms.functional as F


def load_config(config_path):
    """
    加载配置文件
    :param config_path:
    :return:
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# VOC数据集类
class VOCDataset(Dataset):

    def __init__(self, config, image_set, year='2012'):
        self.voc = VOCSegmentation(
            root=config['dir']['data_dir'],  # 数据集输出路径
            year=year,  # 数据集年份
            image_set=image_set,  # 训练数据集(train) or 验证数据集(val)
            download=False,
        )
        self.transform = ImageTransform(mode=image_set, crop_size=config['data']['image_crop_size'],
                                        rgb_mean=config['data']['rgb_mean'], rgb_std=config['data']['rgb_std'])

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, target = self.voc[idx]
        image, target = self.transform(image, target)

        return image, target


def pre_dataloader(config, mode, batch_size, num_workers=0):
    """
    返回数据集加载器
    :param config: 配置文件
    :param mode: 'train' or 'val'
    :param batch_size: 批次大小
    :param num_workers: 数据加载器线程数
    :return: 数据加载器
    """

    # 下载和预处理VOC数据集
    dataset = VOCDataset(config, image_set=mode)

    # 加载器
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(mode == 'train'), num_workers=num_workers, pin_memory=True)
    """
    pin_memory表示是否将数据放置在锁页内存(一定在物理内存)而不放置在可分页内存(可能会放在虚拟内存也就是硬盘)中
    pin_memory设置为True可以提高训练速度, 但是注意物理内存足够和使用GPU训练(使用CPU完全没有必要)
    """

    return dataloader


# transform(数据预处理)
class ImageTransform():
    def __init__(self, mode, crop_size, rgb_mean, rgb_std):
        self.mode = mode  # 训练模式 or 验证模式
        self.crop_size = crop_size  # 随机裁剪后的尺寸
        self.mean = rgb_mean  # VOC数据集三个波段的均值和标准差
        self.std = rgb_std

    def __call__(self, image, target):
        """transform处理"""

        # 原始尺寸小于crop_size则填充
        src_w, src_h = image.size
        dst_w, dst_h = self.crop_size, self.crop_size
        pad_w, pad_h = max(dst_w - src_w, 0), max(dst_h - src_h, 0)
        if (pad_w > 0) or (pad_h > 0):
            padding_tuple = [0, 0, pad_w, pad_h]  # (left, top, right, bottom), 选择往后和往下填充
            image = F.pad(image, padding=padding_tuple, fill=0)
            target = F.pad(target, padding_tuple, fill=255)  # VOC数据集target中0表示背景, 255为忽略值

        # RandomCrop(随机裁剪)
        top, left, height, width = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = F.crop(image, top, left, height, width)
        target = F.crop(target, top, left, height, width)

        # 随机翻转
        if self.mode == 'train':
            if torch.rand(1) > 0.5:
                image = F.hflip(image)  # 水平翻转
                target = F.hflip(target)
            # 一般做Data Augmentation不进行上下颠倒
            # if torch.rand(1) > 0.5:
            #     image = F.vflip(image)  # 垂直翻转
            #     target = F.vflip(target)

        # image图像归一化
        image = F.to_tensor(image)  # 完全等价于实例化的ToTensor方法的作用
        image = F.normalize(image, mean=self.mean, std=self.std)

        # target标签ToTenser
        # target = torch.from_numpy(np.array(target, dtype=np.int16))  # torch训练计算损失时host_softmax不支持int16/short: 报错
        target = torch.as_tensor(np.array(target), dtype=torch.long)

        return image, target


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

        # 模型名称
        self.name = 'attention_unet'

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

    @staticmethod  # 静态方法既可以创建实例调用例如model.cal_loss()也可以不实例化调用例如AttentionUNet.cal_loss()
    def cal_loss(prediction, target, ignore_index):
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = criterion(prediction, target)

        return loss


class StandardUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=21):
        super(StandardUNet, self).__init__()

        # --- Encoder (收缩路径) ---
        # 与 Attention U-Net 完全一致
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck (瓶颈层)
        self.enc5 = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Decoder (扩张路径) ---

        # Up4
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(1024 + 512, 512)  # 输入通道 = Bottleneck(1024) + Skip(512)

        # Up3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(512 + 256, 256)  # 输入通道 = Up4(512) + Skip(256)

        # Up2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(256 + 128, 128)  # 输入通道 = Up3(256) + Skip(128)

        # Up1
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(128 + 64, 64)  # 输入通道 = Up2(128) + Skip(64)

        # Final Output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 模型名称
        self.name = 'unet'

    def forward(self, x):
        # --- Encoder Forward ---
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.enc5(p4)

        # --- Decoder Forward ---
        # 关键区别：直接拼接 (torch.cat)，没有 AttentionGate

        # D4
        d4 = self.up4(b)
        # 此时 d4 是深层特征(Upsampled), e4 是浅层特征(Skip Connection)
        # 直接把 e4 和 d4 拼在一起
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.dec4(d4)

        # D3
        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)

        # D2
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)

        # D1
        d1 = self.up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out

    @staticmethod  # 静态方法既可以创建实例调用例如model.cal_loss()也可以不实例化调用例如AttentionUNet.cal_loss()
    def cal_loss(prediction, target, ignore_index):
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = criterion(prediction, target)

        return loss


def model_train(train_set: DataLoader, dev_set: DataLoader, model: nn.Module, config,
                device: str | torch.device = 'CPU', log_writer: SummaryWriter = None):
    """
    基于VOC数据集进行训练
    :param train_set: 训练集
    :param dev_set: 验证集
    :param model: 模型
    :param config: 配置参数
    :param device: GPU or CPU
    :param log_writer: 日志记录器
    :return:
    """

    # 训练前准备
    # criterion = nn.CrossEntropyLoss(ignore_index=config['ignore_class'])
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(),
                                                          lr=config['learning_rate'],
                                                          weight_decay=config['weight_decay'])
    epochs_num = config['epochs']  # 训练轮数
    epoch_pbar = tqdm(range(epochs_num), desc='Epoch', position=0, ncols=80, colour='blue')
    min_mse = float('inf')

    # 训练-epoch
    for cur_epoch_ix in epoch_pbar:
        model.train()  # 设置模型为训练模式
        train_loss_list = []  # 计算当前epoch的损失

        # 训练-batch
        batch_pbar = tqdm(train_set, desc='Batch', position=1, ncols=90, leave=False,
                          colour='green')  # (留着)leave=False表示结束后不显示进度条
        for images, target in batch_pbar:
            images, target = images.to(device), target.to(device)  # 将获取的数据集放置指定设备中

            optimizer.zero_grad()  # 将模型参数的梯度清零
            prediction = model(images)  # 前向传播
            loss = model.cal_loss(prediction, target, ignore_index=config['ignore_class'])  # 计算损失
            loss.backward()  # 反向传播(计算梯度)
            optimizer.step()  # 基于计算的梯度更新参数

            # 更新batch进度条
            batch_pbar.set_postfix_str('Loss: {:.6f}'.format(loss))
            # 添加当前batch的loss值
            train_loss_list.append(loss.item())

        # 计算当前epoch在训练集的平均Loss
        avg_train_loss = np.mean(train_loss_list)
        # 计算当前epoch在验证集的平均Loss
        avg_dev_loss = model_val(dev_set, model, device, config['ignore_class'])

        # 更新epoch进度条
        epoch_pbar.set_postfix_str('Loss: {:.6f}'.format(avg_train_loss))

        # 保存日志
        if log_writer is not None:
            log_writer.add_scalar('train_loss', scalar_value=avg_train_loss, global_step=cur_epoch_ix)
            log_writer.add_scalar('dev_loss', scalar_value=avg_dev_loss, global_step=cur_epoch_ix)

        # 保存模型
        if avg_dev_loss < min_mse:  # 如果当前阶段的模型的损失值比之前的都要好, 那么保存
            min_mse = avg_dev_loss
            torch.save(model, config['path'][f'{model.name}_path'])

    epoch_pbar.close()


def model_val(dev_set: DataLoader, model: nn.Module, device: str | torch.device, ignore_index: int):
    """
    基于验证集对当前model进行评估
    :param dev_set: 验证集
    :param model: 模型
    :param device: 设备
    :param ignore_index:
    :return: 平均Loss
    """

    dev_loss_list = []  # 验证集的Loss列表

    # 基于验证集对model进行评估 --> 计算loss
    model.eval()  # 设置模型为评估模式
    for images, target in dev_set:
        images, target = images.to(device), target.to(device)

        with torch.no_grad():  # 禁用梯度计算
            prediction = model(images)
            loss = model.cal_loss(prediction, target, ignore_index=ignore_index)  # 计算损失
            dev_loss_list.append(loss.item())

    # 计算model在验证集的平均loss
    avg_dev_loss = np.mean(dev_loss_list)

    return avg_dev_loss


def cal_metrics():
    pass
