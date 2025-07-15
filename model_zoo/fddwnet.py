"""
Paper:      FDDWNet: A Lightweight Convolutional Neural Network for Real-time 
            Sementic Segmentation
Url:        https://arxiv.org/abs/1911.00632
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import DWConvBNAct, ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as DownsamplingUnit

__all__ = ['FDDWNet']

def build_blocks(block, channels, num_block, kernel_size, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, kernel_size, dilations[i], act_type))
    return nn.Sequential(*layers)


class EERMUnit(nn.Module):
    def __init__(self, channels, ks, dt, act_type):
        super().__init__()
        self.conv = nn.Sequential(
                        DWConvBNAct(channels, channels, (ks, 1), act_type='none'),
                        DWConvBNAct(channels, channels, (1, ks), act_type='none'),
                        ConvBNAct(channels, channels, 1, act_type=act_type, inplace=True),
                        DWConvBNAct(channels, channels, (ks, 1), dilation=dt, act_type='none'),
                        DWConvBNAct(channels, channels, (1, ks), dilation=dt, act_type='none'),
                        ConvBNAct(channels, channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.act(x)


class FDDWNet(nn.Module):
    """
    FDDWNet: A Lightweight Convolutional Neural Network for Real-time 
    Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 ks=3, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器部分
        self.layer1 = DownsamplingUnit(input_channels, 16, act_type)
        self.layer2 = DownsamplingUnit(16, 64, act_type)
        self.layer3_7 = build_blocks(EERMUnit, 64, 5, ks, [1,1,1,1,1], act_type)
        self.layer8 = DownsamplingUnit(64, 128, act_type)
        self.layer9_16 = build_blocks(EERMUnit, 128, 8, ks, [1,2,5,9,1,2,5,9], act_type)
        self.layer17_24 = build_blocks(EERMUnit, 128, 8, ks, [2,5,9,17,2,5,9,17], act_type)
        
        # 解码器部分
        self.layer25 = DeConvBNAct(128, 64, act_type=act_type)
        self.layer26_27 = build_blocks(EERMUnit, 64, 2, ks, [1,1], act_type)
        self.layer28 = DeConvBNAct(64, 16, act_type=act_type)
        self.layer29_30 = build_blocks(EERMUnit, 16, 2, ks, [1,1], act_type)
        self.layer31 = DeConvBNAct(16, num_classes, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            # 辅助分支1 - 从中间层
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, 1, act_type=act_type),
                nn.Conv2d(32, num_classes, 1)
            )
            
            # 辅助分支2 - 从深层特征
            self.aux_head2 = nn.Sequential(
                ConvBNAct(128, 64, 3, 1, act_type=act_type),
                nn.Conv2d(64, num_classes, 1)
            )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 编码器前向传播
        x = self.layer1(x)
        x = self.layer2(x)
        residual = self.layer3_7(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(residual)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        x = self.layer8(residual)
        x = self.layer9_16(x)
        x = self.layer17_24(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器前向传播
        x = self.layer25(x)
        x = self.layer26_27(x)
        x += residual
        x = self.layer28(x)
        x = self.layer29_30(x)
        x = self.layer31(x)
        
        # 调整到输入尺寸（如果需要）
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 FDDWNet 特有的参数：

    ks: 卷积核大小
    act_type: 激活函数类型

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到第一个残差块（layer3_7）的输出
    第二个辅助头连接到最深层特征（layer17_24的输出）

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 FDDWNet 的核心架构和特性，包括：

    高效的下采样单元（从ENet借鉴）
    高效的编码-解码结构
    轻量级的深度可分离卷积
    扩展有效感受野的空洞卷积
    残差连接

依赖导入：从您现有的模块中导入所需组件，如 DWConvBNAct, ConvBNAct, DeConvBNAct, Activation 和 InitialBlock（从ENet）。
'''