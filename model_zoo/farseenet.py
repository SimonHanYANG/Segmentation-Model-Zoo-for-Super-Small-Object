"""
Paper:      FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale 
            Context Aggregation and Feature Space Super-resolution
Url:        https://arxiv.org/abs/2003.03913
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, DWConvBNAct, ConvBNAct
from backbone import ResNet

__all__ = ['FarSeeNet']

class FASPP(nn.Module):
    def __init__(self, high_channels, low_channels, num_class, act_type, 
                    dilations=[6,12,18], hid_channels=256):
        super().__init__()
        # High level convolutions
        self.conv_high = nn.ModuleList([
                                ConvBNAct(high_channels, hid_channels, 1, act_type=act_type)
                            ])
        for dt in dilations:
            self.conv_high.append(
                nn.Sequential(
                    ConvBNAct(high_channels, hid_channels, 1, act_type=act_type),
                    DWConvBNAct(hid_channels, hid_channels, 3, dilation=dt, act_type=act_type)
                )
            )

        self.sub_pixel_high = nn.Sequential(
                                    conv1x1(hid_channels*4, hid_channels*2*(2**2)),
                                    nn.PixelShuffle(2)
                                )

        # Low level convolutions
        self.conv_low_init = ConvBNAct(low_channels, 48, 1, act_type=act_type)
        self.conv_low = nn.ModuleList([
                            ConvBNAct(hid_channels*2+48, hid_channels//2, 1, act_type=act_type)
                        ])
        for dt in dilations[:-1]:
            self.conv_low.append(
                nn.Sequential(
                    ConvBNAct(hid_channels*2+48, hid_channels//2, 1, act_type=act_type),
                    DWConvBNAct(hid_channels//2, hid_channels//2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_low_last = nn.Sequential(
                                ConvBNAct(hid_channels//2*3, hid_channels*2, 1, act_type=act_type),
                                ConvBNAct(hid_channels*2, hid_channels*2, act_type=act_type)
                            )

        self.sub_pixel_low = nn.Sequential(
                                conv1x1(hid_channels*2, num_class*(4**2)),
                                nn.PixelShuffle(4)
                            )

    def forward(self, x_high, x_low):
        # High level features
        high_feats = []
        for conv_high in self.conv_high:
            high_feats.append(conv_high(x_high))

        x = torch.cat(high_feats, dim=1)
        x = self.sub_pixel_high(x)

        # Low level features
        x_low = self.conv_low_init(x_low)
        x = torch.cat([x, x_low], dim=1)

        low_feats = []
        for conv_low in self.conv_low:
            low_feats.append(conv_low(x))

        x = torch.cat(low_feats, dim=1)
        x = self.conv_low_last(x)
        x = self.sub_pixel_low(x)

        return x


class FarSeeNet(nn.Module):
    """
    FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale 
    Context Aggregation and Feature Space Super-resolution
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 backbone_type='resnet18', act_type='relu', dilations=[6,12,18], 
                 hid_channels=256, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 前端网络（骨干网络）
        if 'resnet' in backbone_type:
            self.frontend_network = ResNet(backbone_type)
            high_channels = 512 if backbone_type in ['resnet18', 'resnet34'] else 2048
            low_channels = 256 if backbone_type in ['resnet18', 'resnet34'] else 1024
        else:
            raise NotImplementedError("Only ResNet backbones are supported")

        # 后端网络（FASPP）
        self.backend_network = FASPP(high_channels, low_channels, num_classes, 
                                    act_type, dilations, hid_channels)
        
        # 深度监督分支
        if deep_supervision:
            # 辅助分支1 - 从低级特征
            self.aux_head1 = nn.Sequential(
                ConvBNAct(low_channels, 128, 3, 1, act_type=act_type),
                ConvBNAct(128, 64, 3, 1, act_type=act_type),
                nn.Conv2d(64, num_classes, 1)
            )
            
            # 辅助分支2 - 从高级特征
            self.aux_head2 = nn.Sequential(
                ConvBNAct(high_channels, 256, 3, 1, act_type=act_type),
                ConvBNAct(256, 128, 3, 1, act_type=act_type),
                nn.Conv2d(128, num_classes, 1)
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
        
        # 提取骨干网络特征
        _, _, x_low, x_high = self.frontend_network(x)
        
        # 辅助分支（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x_low)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
            
            aux2 = self.aux_head2(x_high)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 主分支 - FASPP
        x = self.backend_network(x_high, x_low)
        
        # 调整到输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x
    

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 FarSeeNet 特有的参数：

    backbone_type: 骨干网络类型（如'resnet18', 'resnet34', 'resnet50'等）
    act_type: 激活函数类型
    dilations: FASPP模块中使用的空洞率
    hid_channels: 隐藏层通道数

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到骨干网络的低级特征
    第二个辅助头连接到骨干网络的高级特征

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 FarSeeNet 的核心架构和特性，包括：

    基于ResNet的前端网络
    FASPP（Fast Atrous Spatial Pyramid Pooling）后端网络
    特征空间超分辨率（通过PixelShuffle实现）
    
依赖导入：从您现有的模块中导入所需组件，如 conv1x1, DWConvBNAct, ConvBNAct 和 ResNet。
'''