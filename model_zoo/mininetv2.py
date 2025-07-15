"""
Paper:      MiniNet: An Efficient Semantic Segmentation ConvNet for Real-Time Robotic Applications
Url:        https://ieeexplore.ieee.org/abstract/document/9023474
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import DWConvBNAct, PWConvBNAct, ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as DownsamplingUnit

__all__ = ['MiniNetv2']

def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, channels, 3, 1, dilations[i], act_type))
    return nn.Sequential(*layers)


class MultiDilationDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act_type='relu'):
        super().__init__()
        self.dilated = dilation > 1
        self.dw_conv = DWConvBNAct(in_channels, in_channels, kernel_size, stride, 1, act_type)
        self.pw_conv = PWConvBNAct(in_channels, out_channels, act_type, inplace=True)
        if self.dilated:
            self.ddw_conv = DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, inplace=True)

    def forward(self, x):
        x_dw = self.dw_conv(x)
        if self.dilated:
            x_ddw = self.ddw_conv(x)
            x_dw += x_ddw
        x = self.pw_conv(x_dw)

        return x


class MiniNetv2(nn.Module):
    """
    MiniNetv2: An Efficient Semantic Segmentation ConvNet for Real-Time Robotic Applications
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 feat_dt=[1,2,1,4,1,8,1,16,1,1,1,2,1,4,1,8], act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 主干网络
        self.d1_2 = nn.Sequential(
                        DownsamplingUnit(input_channels, 16, act_type),
                        DownsamplingUnit(16, 64, act_type),
                    )
        self.ref = nn.Sequential(
                        DownsamplingUnit(input_channels, 16, act_type),
                        DownsamplingUnit(16, 64, act_type)
                    )
        self.m1_10 = build_blocks(MultiDilationDSConv, 64, 10, act_type=act_type)
        self.d3 = DownsamplingUnit(64, 128, act_type)
        self.feature_extractor = build_blocks(MultiDilationDSConv, 128, len(feat_dt), feat_dt, act_type)
        self.up1 = DeConvBNAct(128, 64, act_type=act_type)
        self.m26_29 = build_blocks(MultiDilationDSConv, 64, 4, act_type=act_type)
        self.output = DeConvBNAct(64, num_classes, act_type=act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从m1_10输出
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, act_type=act_type),
                DeConvBNAct(32, num_classes, act_type=act_type)
            )
            
            # 辅助分支2 - 从feature_extractor输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(128, 64, 3, act_type=act_type),
                DeConvBNAct(64, 32, act_type=act_type),
                DeConvBNAct(32, num_classes, act_type=act_type)
            )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 参考分支
        x_ref = self.ref(x)
        
        # 主干网络
        x = self.d1_2(x)
        x = self.m1_10(x)
        
        # 第一个辅助输出
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 继续主干网络
        x = self.d3(x)
        x = self.feature_extractor(x)
        
        # 第二个辅助输出
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        # 上采样和残差连接
        x = self.up1(x)
        x += x_ref
        
        # 最终处理
        x = self.m26_29(x)
        x = self.output(x)
        
        # 调整到输入尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 MiniNetv2 特有的参数：

    feat_dt: 特征提取器中的扩张率列表，默认为 [1,2,1,4,1,8,1,16,1,1,1,2,1,4,1,8]
    act_type: 激活函数类型，默认为 'relu'

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到 m1_10 的输出
    第二个辅助头连接到 feature_extractor 的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 MiniNetv2 的核心架构和特性，包括：

    轻量级设计
    多扩张率卷积
    残差连接
    高效的下采样单元（从 ENet 借鉴）

依赖导入：从您现有的模块中导入所需组件，如 DWConvBNAct, PWConvBNAct, ConvBNAct, DeConvBNAct, Activation 和 InitialBlock。
'''