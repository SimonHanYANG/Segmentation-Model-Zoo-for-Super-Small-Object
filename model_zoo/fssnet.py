"""
Paper:      Fast Semantic Segmentation for Scene Perception
Url:        https://ieeexplore.ieee.org/document/8392426
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as InitBlock

__all__ = ['FSSNet']

def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, dilations[i], act_type))
    return nn.Sequential(*layers)


class FactorizedBlock(nn.Module):
    def __init__(self, channels, dilation=1, act_type='relu'):
        super().__init__()
        hid_channels = channels // 4
        self.conv = nn.Sequential(
                        ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                        ConvBNAct(hid_channels, hid_channels, (1,3), act_type='none'),
                        ConvBNAct(hid_channels, hid_channels, (3,1), act_type=act_type),
                        ConvBNAct(hid_channels, channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.act(x)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super().__init__()
        hid_channels = channels // 4
        self.conv = nn.Sequential(
                        ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                        ConvBNAct(hid_channels, hid_channels, 3, dilation=dilation, act_type=act_type),
                        ConvBNAct(hid_channels, channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.act(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        hid_channels = out_channels // 4
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels, hid_channels, 2, 2, act_type=act_type),
                        ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type),
                        ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                    )
        self.pool = nn.Sequential(
                        nn.MaxPool2d(3, 2, 1),
                        ConvBNAct(in_channels, out_channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        x_pool = self.pool(x)
        x = self.conv(x)
        x += x_pool
        return self.act(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        hid_channels = in_channels // 4
        self.deconv = nn.Sequential(
                            ConvBNAct(in_channels, hid_channels, 1, act_type=act_type),
                            DeConvBNAct(hid_channels, hid_channels, act_type=act_type),
                            ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                        )
        self.conv = ConvBNAct(in_channels, out_channels, 1, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x, pool_feat):
        x_deconv = self.deconv(x)

        x = x + pool_feat
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x += x_deconv
        return self.act(x)


class FSSNet(nn.Module):
    """
    Fast Semantic Segmentation Network for Scene Perception
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 act_type='prelu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器部分
        self.init_block = InitBlock(input_channels, 16, act_type)
        self.down1 = DownsamplingBlock(16, 64, act_type)
        self.factorized = build_blocks(FactorizedBlock, 64, 4, act_type=act_type)
        self.down2 = DownsamplingBlock(64, 128, act_type)
        self.dilated = build_blocks(DilatedBlock, 128, 6, [2,5,9,2,5,9], act_type)
        
        # 解码器部分
        self.up2 = UpsamplingBlock(128, 64, act_type)
        self.bottleneck2 = build_blocks(DilatedBlock, 64, 2, act_type=act_type)
        self.up1 = UpsamplingBlock(64, 16, act_type)
        self.bottleneck1 = build_blocks(DilatedBlock, 16, 2, act_type=act_type)
        self.full_conv = DeConvBNAct(16, num_classes, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            # 辅助分支1 - 从第一个下采样后
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, 1, act_type=act_type),
                nn.Conv2d(32, num_classes, 1)
            )
            
            # 辅助分支2 - 从第二个下采样后
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
        x = self.init_block(x)      # 2x down
        x_d1 = self.down1(x)        # 4x down
        x = self.factorized(x_d1)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        x_d2 = self.down2(x)        # 8x down
        x = self.dilated(x_d2)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器前向传播
        x = self.up2(x, x_d2)       # 8x up
        x = self.bottleneck2(x)
        x = self.up1(x, x_d1)       # 4x up
        x = self.bottleneck1(x)
        x = self.full_conv(x)       # 2x up
        
        # 调整到输入尺寸（如果需要）
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 FSSNet 特有的参数：

    act_type: 激活函数类型，默认为 'prelu'

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到第一个下采样后的特征
    第二个辅助头连接到第二个下采样后的特征

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 FSSNet 的核心架构和特性，包括：

    初始块（从ENet借鉴）
    分解卷积块（Factorized Block）
    空洞卷积块（Dilated Block）
    下采样块（Downsampling Block）
    上采样块（Upsampling Block）

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, DeConvBNAct, Activation 和 InitialBlock（从ENet）。
'''