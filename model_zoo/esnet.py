"""
Paper:      ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/1906.09826
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as DownsamplingUnit

__all__ = ['ESNet']

def build_blocks(block_type, channels, num_block, K=None, r1=None, r2=None, r3=None, 
                act_type='relu'):
    layers = []
    for _ in range(num_block):
        if block_type == 'fcu':
            layers.append(FCU(channels, K, act_type))
        elif block_type == 'pfcu':
            layers.append(PFCU(channels, r1, r2, r3, act_type))
        else:
            raise NotImplementedError(f'Unsupported block type: {block_type}.\n')
    return nn.Sequential(*layers)


class FCU(nn.Module):
    def __init__(self, channels, K, act_type):
        super().__init__()
        assert K is not None, 'K should not be None.\n'
        padding = (K - 1) // 2
        self.conv = nn.Sequential(
                        nn.Conv2d(channels, channels, (K, 1), padding=(padding, 0), bias=False),
                        Activation(act_type, inplace=True),
                        ConvBNAct(channels, channels, (1, K), act_type=act_type, inplace=True),
                        nn.Conv2d(channels, channels, (K, 1), padding=(padding, 0), bias=False),
                        Activation(act_type, inplace=True),
                        ConvBNAct(channels, channels, (1, K), act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x += residual

        return self.act(x)


class PFCU(nn.Module):
    def __init__(self, channels, r1, r2, r3, act_type):
        super().__init__()
        assert (r1 is not None) and (r2 is not None) and (r3 is not None)

        self.conv0 = nn.Sequential(
                        nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False),
                        Activation(act_type, inplace=True),
                        ConvBNAct(channels, channels, (1, 3), act_type=act_type, inplace=True)
                    )
        self.conv_left = nn.Sequential(
                            nn.Conv2d(channels, channels, (3, 1), padding=(r1, 0), 
                                        dilation=r1, bias=False),
                            Activation(act_type, inplace=True),
                            ConvBNAct(channels, channels, (1, 3), dilation=r1, act_type='none')
                        )
        self.conv_mid = nn.Sequential(
                            nn.Conv2d(channels, channels, (3, 1), padding=(r2, 0), 
                                        dilation=r2, bias=False),
                            Activation(act_type, inplace=True),
                            ConvBNAct(channels, channels, (1, 3), dilation=r2, act_type='none')
                        )
        self.conv_right = nn.Sequential(
                            nn.Conv2d(channels, channels, (3, 1), padding=(r3, 0), 
                                        dilation=r3, bias=False),
                            Activation(act_type, inplace=True),
                            ConvBNAct(channels, channels, (1, 3), dilation=r3, act_type='none')
                        )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x = self.conv0(x)

        x_left = self.conv_left(x)
        x_mid = self.conv_mid(x)
        x_right = self.conv_right(x)

        x = x_left + x_mid + x_right + residual

        return self.act(x)


class ESNet(nn.Module):
    """
    ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器部分
        self.block1_down = DownsamplingUnit(input_channels, 16, act_type)
        self.block1 = build_blocks('fcu', 16, 3, K=3, act_type=act_type)
        
        self.block2_down = DownsamplingUnit(16, 64, act_type)
        self.block2 = build_blocks('fcu', 64, 2, K=5, act_type=act_type)
        
        self.block3_down = DownsamplingUnit(64, 128, act_type)
        self.block3 = build_blocks('pfcu', 128, 3, r1=2, r2=5, r3=9, act_type=act_type)
        
        # 解码器部分
        self.block4_up = DeConvBNAct(128, 64, act_type=act_type)
        self.block4 = build_blocks('fcu', 64, 2, K=5, act_type=act_type)
        
        self.block5_up = DeConvBNAct(64, 16, act_type=act_type)
        self.block5 = build_blocks('fcu', 16, 2, K=3, act_type=act_type)
        
        self.full_conv = DeConvBNAct(16, num_classes, act_type=act_type)
        
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
        x = self.block1_down(x)  # 1/2
        x = self.block1(x)
        
        # 编码器阶段2
        x = self.block2_down(x)  # 1/4
        x = self.block2(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段3
        x = self.block3_down(x)  # 1/8
        x = self.block3(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器阶段1
        x = self.block4_up(x)  # 1/4
        x = self.block4(x)
        
        # 解码器阶段2
        x = self.block5_up(x)  # 1/2
        x = self.block5(x)
        
        # 最终卷积
        x = self.full_conv(x)  # 1/1
        
        # 确保输出尺寸与输入一致
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ESNet 特有的参数：

    act_type: 激活函数类型（默认为'relu'）

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的编码器阶段添加两个辅助分割头：

    第一个辅助头连接到第二阶段编码器的输出（1/4分辨率）
    第二个辅助头连接到第三阶段编码器的输出（1/8分辨率）

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。这是通过在最终输出层添加额外的插值操作实现的。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 ESNet 的核心架构和特性，包括：

    对称的编码器-解码器结构
    分解卷积单元（FCU）
    并行分解卷积单元（PFCU）
    下采样和上采样单元

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, DeConvBNAct, Activation 和 DownsamplingUnit。

'''