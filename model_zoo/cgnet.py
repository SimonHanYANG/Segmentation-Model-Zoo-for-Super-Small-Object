"""
Paper:      CGNet: A Light-weight Context Guided Network for Semantic Segmentation
Url:        https://arxiv.org/abs/1811.08201
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct, Activation

__all__ = ['CGNet']

class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.conv0 = ConvBNAct(in_channels, out_channels, stride=2, act_type=act_type)
        self.conv1 = ConvBNAct(out_channels, out_channels, act_type=act_type)
        self.conv2 = ConvBNAct(out_channels, out_channels, act_type=act_type)

    def forward(self, x):
        x0 = self.conv0(x)
        x = self.conv1(x0)
        x = self.conv2(x)
        return x, x0


def build_blocks(block, in_channels, out_channels, dilation, num_block, act_type):
    layers = []
    for _ in range(num_block):
        layers.append(block(in_channels, out_channels, 1, dilation, act_type=act_type))
        in_channels = out_channels
    return nn.Sequential(*layers)


class CGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, res_type='GRL', act_type='prelu'):
        super().__init__()
        if res_type not in ['GRL', 'LRL']:
            raise ValueError('Residual learning only support GRL and LRL type.\n')
        self.res_type = res_type
        self.use_skip = (stride == 1) and (in_channels == out_channels)
        self.conv = conv1x1(in_channels, out_channels//2)
        self.loc = nn.Conv2d(out_channels//2, out_channels//2, 3, stride, padding=1, 
                                groups=out_channels//2, bias=False)
        self.sur = nn.Conv2d(out_channels//2, out_channels//2, 3, stride, padding=dilation, 
                                dilation=dilation, groups=out_channels//2, bias=False)
        self.joi = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                            )
        self.glo = nn.Sequential(
                                nn.Linear(out_channels, out_channels//8),
                                nn.Linear(out_channels//8, out_channels)
                            )

    def forward(self, x):
        residual = x
        x = self.conv(x)

        x_loc = self.loc(x)
        x_sur = self.sur(x)

        x = torch.cat([x_loc, x_sur], dim=1)
        x = self.joi(x)

        if self.use_skip and self.res_type == 'LRL':
            x += residual

        x_glo = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x_glo = torch.sigmoid(self.glo(x_glo))
        x_glo = x_glo.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = x * x_glo

        if self.use_skip and self.res_type == 'GRL':
            x += residual

        return x


class CGNet(nn.Module):
    """
    CGNet: A Light-weight Context Guided Network for Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 M=3, N=15, act_type='prelu', res_type='GRL', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 主要网络结构
        self.stage1 = InitBlock(input_channels, 32, act_type=act_type)
        self.stage2_down = CGBlock(64, 64, 2, 2, res_type=res_type, act_type=act_type)
        self.stage2 = build_blocks(CGBlock, 64+3, 64, 2, M-1, act_type)
        self.stage3_down = CGBlock(128, 128, 2, 4, res_type=res_type, act_type=act_type)
        self.stage3 = build_blocks(CGBlock, 128+3, 128, 4, N-1, act_type)
        self.seg_head = conv1x1(128*2, num_classes)
        
        # 深度监督分支
        if deep_supervision:
            self.dsv1 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.dsv2 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.dsv3 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 降采样输入用于输入注入
        x_d4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        x_d8 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=True)
        
        # 阶段1
        x, x1 = self.stage1(x)
        
        if self.deep_supervision:
            dsv1 = self.dsv1(x)
        
        # 阶段2
        x = torch.cat([x, x1], dim=1)
        x2 = self.stage2_down(x)
        
        if self.deep_supervision:
            dsv2 = self.dsv2(x2)
        
        x = torch.cat([x2, x_d4], dim=1)  # 输入注入
        x = self.stage2(x)
        
        # 阶段3
        x = torch.cat([x, x2], dim=1)
        x3 = self.stage3_down(x)
        
        if self.deep_supervision:
            dsv3 = self.dsv3(x3)
        
        x = torch.cat([x3, x_d8], dim=1)  # 输入注入
        x = self.stage3(x)
        
        # 最终输出
        x = torch.cat([x, x3], dim=1)
        x = self.seg_head(x)
        
        # 上采样到原始尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 深度监督
        if self.deep_supervision:
            dsv1 = F.interpolate(dsv1, input_size, mode='bilinear', align_corners=True)
            dsv2 = F.interpolate(dsv2, input_size, mode='bilinear', align_corners=True)
            dsv3 = F.interpolate(dsv3, input_size, mode='bilinear', align_corners=True)
            return [dsv1, dsv2, dsv3, x]
        
        return x
    

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 CGNet 特有的参数：M, N, act_type, res_type。

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的三个主要阶段添加辅助分割头，并返回所有输出。

参数扩展：

    M: 控制第二阶段的 CGBlock 数量（默认为3）
    N: 控制第三阶段的 CGBlock 数量（默认为15）
    act_type: 激活函数类型（默认为'prelu'）
    res_type: 残差学习类型，可选 'GRL' 或 'LRL'（默认为'GRL'）
    
输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，包括卷积层、批归一化层和线性层的初始化，提高模型收敛性。
'''