"""
Paper:      ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
Url:        https://ieeexplore.ieee.org/document/8063438
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as DownsamplerBlock

__all__ = ['ERFNet']

def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, dilation=dilations[i], act_type=act_type))
    return nn.Sequential(*layers)


class NonBt1DBlock(nn.Module):
    def __init__(self, channels, dilation=1, act_type='relu'):
        super().__init__()
        self.conv = nn.Sequential(
                                ConvBNAct(channels, channels, (3, 1), inplace=True),
                                ConvBNAct(channels, channels, (1, 3), inplace=True),
                                ConvBNAct(channels, channels, (3, 1), dilation=dilation, inplace=True),
                                nn.Conv2d(channels, channels, (1, 3), dilation=dilation, 
                                            padding=(0, dilation), bias=False)
                            )
        self.bn_act = nn.Sequential(
                                nn.BatchNorm2d(channels),
                                Activation(act_type, inplace=True)
                            )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        x = self.bn_act(x)
        return x


class ERFNet(nn.Module):
    """
    ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器部分
        self.layer1 = DownsamplerBlock(input_channels, 16, act_type=act_type)
        
        self.layer2 = DownsamplerBlock(16, 64, act_type=act_type)
        self.layer3_7 = build_blocks(NonBt1DBlock, 64, 5, act_type=act_type)
        
        self.layer8 = DownsamplerBlock(64, 128, act_type=act_type)
        self.layer9_16 = build_blocks(NonBt1DBlock, 128, 8, 
                                      dilations=[2,4,8,16,2,4,8,16], act_type=act_type)
        
        # 解码器部分
        self.layer17 = DeConvBNAct(128, 64, act_type=act_type)
        self.layer18_19 = build_blocks(NonBt1DBlock, 64, 2, act_type=act_type)
        
        self.layer20 = DeConvBNAct(64, 16, act_type=act_type)
        self.layer21_22 = build_blocks(NonBt1DBlock, 16, 2, act_type=act_type)
        
        self.layer23 = DeConvBNAct(16, num_classes, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, 1, act_type=act_type),
                nn.Conv2d(32, num_classes, 1)
            )
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
        
        # 编码器阶段1
        x = self.layer1(x)  # 1/2
        
        # 编码器阶段2
        x = self.layer2(x)  # 1/4
        x = self.layer3_7(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段3
        x = self.layer8(x)  # 1/8
        x = self.layer9_16(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器阶段1
        x = self.layer17(x)  # 1/4
        x = self.layer18_19(x)
        
        # 解码器阶段2
        x = self.layer20(x)  # 1/2
        x = self.layer21_22(x)
        
        # 解码器阶段3
        x = self.layer23(x)  # 1/1
        
        # 确保输出尺寸与输入一致
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ERFNet 特有的参数：

    act_type: 激活函数类型（默认为'relu'）

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的编码器阶段添加两个辅助分割头：

    第一个辅助头连接到第二阶段编码器的输出（1/4分辨率）
    第二个辅助头连接到第三阶段编码器的输出（1/8分辨率）

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。这是通过在最终输出层添加额外的插值操作实现的。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 ERFNet 的核心架构和特性，包括非瓶颈1D模块、下采样块和解码器设计。

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, DeConvBNAct, Activation 和 DownsamplerBlock。
'''