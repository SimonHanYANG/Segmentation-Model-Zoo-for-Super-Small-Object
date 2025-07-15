"""
Paper:      Cross Attention Network for Semantic Segmentation
Url:        https://arxiv.org/abs/1907.10958
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, DeConvBNAct, Activation
from backbone import ResNet, Mobilenetv2

__all__ = ['CANet']

class SpatialBranch(nn.Sequential):
    def __init__(self, n_channel, channels, act_type):
        super().__init__(
            ConvBNAct(n_channel, channels, 3, 2, act_type=act_type, inplace=True),
            ConvBNAct(channels, channels*2, 3, 2, act_type=act_type, inplace=True),
            ConvBNAct(channels*2, channels*4, 3, 2, act_type=act_type, inplace=True),
        )


class ContextBranch(nn.Module):
    def __init__(self, out_channels, backbone_type, hid_channels=192):
        super().__init__()
        if 'mobilenet' in backbone_type:
            self.backbone = Mobilenetv2()
            channels = [320, 96]
        elif 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [512, 256] if (('18' in backbone_type) or ('34' in backbone_type)) else [2048, 1024]
        else:
            raise NotImplementedError()

        self.up1 = DeConvBNAct(channels[0], hid_channels)
        self.up2 = DeConvBNAct(channels[1] + hid_channels, out_channels)

    def forward(self, x):
        _, _, x_d16, x = self.backbone(x)
        x = self.up1(x)

        x = torch.cat([x, x_d16], dim=1)
        x = self.up2(x)

        return x


class SpatialAttentionBlock(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__(
            ConvBNAct(in_channels, 1, act_type='sigmoid')
        )


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x_max = self.max_pool(x).view(-1, self.in_channels)
        x_avg = self.avg_pool(x).view(-1, self.in_channels)

        x_max = self.fc(x_max)
        x_avg = self.fc(x_avg)

        x = x_max + x_avg
        x = torch.sigmoid(x)

        return x.unsqueeze(-1).unsqueeze(-1)


class FeatureCrossAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.conv_init = ConvBNAct(2*in_channels, in_channels, act_type=act_type, inplace=True)
        self.sa = SpatialAttentionBlock(in_channels)
        self.ca = ChannelAttentionBlock(in_channels)
        self.conv_last = ConvBNAct(in_channels, out_channels, inplace=True)

    def forward(self, x_s, x_c):
        x = torch.cat([x_s, x_c], dim=1)
        x_s = self.sa(x_s)
        x_c = self.ca(x_c)

        x = self.conv_init(x)
        residual = x

        x = x * x_s
        x = x * x_c
        x += residual

        x = self.conv_last(x)

        return x


class CANet(nn.Module):
    """
    Cross Attention Network for Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 backbone_type='mobilenet_v2', act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 主要分支
        self.spatial_branch = SpatialBranch(input_channels, 64, act_type)
        self.context_branch = ContextBranch(64*4, backbone_type)
        self.fca = FeatureCrossAttentionModule(64*4, num_classes, act_type)
        self.up = DeConvBNAct(num_classes, num_classes, scale_factor=8)
        
        # 深度监督分支
        if deep_supervision:
            self.dsv1 = nn.Conv2d(64*4, num_classes, kernel_size=1)
            self.dsv2 = nn.Conv2d(64*4, num_classes, kernel_size=1)
            self.dsv3 = nn.Conv2d(num_classes, num_classes, kernel_size=1)

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
        
        # 主要分支处理
        x_s = self.spatial_branch(x)
        x_c = self.context_branch(x)
        
        # 深度监督输出
        if self.deep_supervision:
            dsv1 = self.dsv1(x_s)
            dsv2 = self.dsv2(x_c)
        
        # 特征融合
        x = self.fca(x_s, x_c)
        
        if self.deep_supervision:
            dsv3 = self.dsv3(x)
        
        # 上采样到原始尺寸
        x = self.up(x)
        
        # 确保输出大小与输入大小匹配
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 深度监督
        if self.deep_supervision:
            dsv1 = F.interpolate(dsv1, size=input_size, mode='bilinear', align_corners=True)
            dsv2 = F.interpolate(dsv2, size=input_size, mode='bilinear', align_corners=True)
            dsv3 = F.interpolate(dsv3, size=input_size, mode='bilinear', align_corners=True)
            return [dsv1, dsv2, dsv3, x]
        
        return x