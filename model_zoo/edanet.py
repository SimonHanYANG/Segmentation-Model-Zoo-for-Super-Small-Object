"""
Paper:      Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1809.06323
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, conv3x3, ConvBNAct, Activation

__all__ = ['EDANet']

class EDAModule(nn.Module):
    def __init__(self, in_channels, k, dilation=1, act_type='relu'):
        super().__init__()
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels, k, 1),
                        nn.Conv2d(k, k, (3, 1), padding=(1, 0), bias=False),
                        ConvBNAct(k, k, (1, 3), act_type=act_type),
                        nn.Conv2d(k, k, (3, 1), dilation=dilation, 
                                    padding=(dilation, 0), bias=False),
                        ConvBNAct(k, k, (1, 3), dilation=dilation, act_type=act_type)
                    )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = torch.cat([x, residual], dim=1)
        return x


class EDABlock(nn.Module):
    def __init__(self, in_channels, k, num_block, dilations, act_type):
        super().__init__()
        assert len(dilations) == num_block, 'number of dilation rate should be equal to number of block'

        layers = []
        for i in range(num_block):
            dt = dilations[i]
            layers.append(EDAModule(in_channels, k, dt, act_type))
            in_channels += k
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels - in_channels, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_act = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                            )

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return self.bn_act(x)


class EDANet(nn.Module):
    """
    EDANet: Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 k=40, num_b1=5, num_b2=8, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 第一阶段：下采样块
        self.stage1 = DownsamplingBlock(input_channels, 15, act_type)
        
        # 第二阶段：下采样块 + EDA块
        self.stage2_d = DownsamplingBlock(15, 60, act_type)
        self.stage2 = EDABlock(60, k, num_b1, [1,1,1,2,2], act_type)
        
        # 计算第二阶段输出通道数
        stage2_out_channels = 60 + k * num_b1
        
        # 第三阶段：下采样块 + EDA块
        self.stage3_d = ConvBNAct(stage2_out_channels, 130, 3, 2, act_type=act_type)
        self.stage3 = EDABlock(130, k, num_b2, [2,2,4,4,8,8,16,16], act_type)
        
        # 计算第三阶段输出通道数
        stage3_out_channels = 130 + k * num_b2
        
        # 主分割头
        self.project = conv1x1(stage3_out_channels, num_classes)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(stage2_out_channels, 128, 3, 1, act_type=act_type),
                conv1x1(128, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(130, 128, 3, 1, act_type=act_type),
                conv1x1(128, num_classes)
            )
        
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

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 第一阶段
        x = self.stage1(x)
        
        # 第二阶段
        x = self.stage2_d(x)
        stage2_features = self.stage2(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(stage2_features)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 第三阶段第一部分
        x = self.stage3_d(stage2_features)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        # 第三阶段第二部分
        x = self.stage3(x)
        
        # 主分割头
        x = self.project(x)
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x
    
'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 EDANet 特有的参数：

    k: EDA模块的增长率（默认为40）
    num_b1: 第一个EDA块中的模块数量（默认为5）
    num_b2: 第二个EDA块中的模块数量（默认为8）
    act_type: 激活函数类型（默认为'relu'）

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间层添加两个辅助分割头：

    第一个辅助头连接到第二阶段的输出
    第二个辅助头连接到第三阶段的第一部分输出
    
通道数计算：动态计算每个阶段的输出通道数，以适应不同的 k 和 num_b 参数配置。

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。
'''