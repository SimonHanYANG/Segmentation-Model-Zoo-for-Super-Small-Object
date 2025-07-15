"""
Paper:      ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time
Url:        https://arxiv.org/abs/1805.04554
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, DSConvBNAct, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation

__all__ = ['ContextNet']

class Branch_1(nn.Sequential):
    def __init__(self, in_channels, hid_channels, out_channels, act_type='relu'):
        assert len(hid_channels) == 3
        super().__init__(
                ConvBNAct(in_channels, hid_channels[0], 3, 2, act_type=act_type),
                DWConvBNAct(hid_channels[0], hid_channels[0], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[0], hid_channels[1], act_type=act_type),
                DWConvBNAct(hid_channels[1], hid_channels[1], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[1], hid_channels[2], act_type=act_type),
                DWConvBNAct(hid_channels[2], hid_channels[2], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[2], out_channels, act_type=act_type)
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6, act_type='relu'):
        super().__init__()
        hid_channels = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
                        PWConvBNAct(in_channels, hid_channels, act_type=act_type),
                        DWConvBNAct(hid_channels, hid_channels, 3, stride, act_type=act_type),
                        ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                    )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Branch_4(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.conv_init = ConvBNAct(in_channels, 32, 3, 2, act_type=act_type)
        inverted_residual_setting = [
                # t, c, n, s
                [1, 32, 1, 1],
                [6, 32, 1, 1],
                [6, 48, 3, 2],
                [6, 64, 3, 2],
                [6, 96, 2, 1],
                [6, 128, 2, 1],
            ]

        # Building inverted residual blocks
        features = []
        in_channels = 32
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
        self.bottlenecks = nn.Sequential(*features)
        self.conv_last = ConvBNAct(128, out_channels, 3, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)

        return x


class FeatureFusion(nn.Module):
    def __init__(self, branch_1_channels, branch_4_channels, out_channels, act_type='relu'):
        super().__init__()
        self.branch_1_conv = conv1x1(branch_1_channels, out_channels)
        self.branch_4_conv = nn.Sequential(
                                DSConvBNAct(branch_4_channels, out_channels, 3, dilation=4, act_type='none'),
                                conv1x1(out_channels, out_channels)
                                )
        self.act = Activation(act_type=act_type)                                 

    def forward(self, branch_1_feat, branch_4_feat):
        size = branch_1_feat.size()[2:]

        branch_1_feat = self.branch_1_conv(branch_1_feat)

        branch_4_feat = F.interpolate(branch_4_feat, size, mode='bilinear', align_corners=True)
        branch_4_feat = self.branch_4_conv(branch_4_feat)

        res = branch_1_feat + branch_4_feat
        res = self.act(res)

        return res


class ContextNet(nn.Module):
    """
    ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 主要网络结构
        self.full_res_branch = Branch_1(input_channels, [32, 64, 128], 128, act_type=act_type)
        self.lower_res_branch = Branch_4(input_channels, 128, act_type=act_type)
        self.feature_fusion = FeatureFusion(128, 128, 128, act_type=act_type)
        self.classifier = ConvBNAct(128, num_classes, 1, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.dsv1 = nn.Conv2d(128, num_classes, kernel_size=1)
            self.dsv2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
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
        
        # 降采样输入用于低分辨率分支
        x_lower = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        
        # 全分辨率分支
        full_res_feat = self.full_res_branch(x)
        
        # 低分辨率分支
        lower_res_feat = self.lower_res_branch(x_lower)
        
        # 深度监督输出
        if self.deep_supervision:
            # 全分辨率分支的深度监督
            dsv1 = self.dsv1(full_res_feat)
            dsv1 = F.interpolate(dsv1, input_size, mode='bilinear', align_corners=True)
            
            # 低分辨率分支的深度监督
            dsv2 = self.dsv2(lower_res_feat)
            dsv2 = F.interpolate(dsv2, input_size, mode='bilinear', align_corners=True)
        
        # 特征融合
        x = self.feature_fusion(full_res_feat, lower_res_feat)
        
        # 分类器
        x = self.classifier(x)
        
        # 上采样到原始尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [dsv1, dsv2, x]
        
        return x
    
'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ContextNet 特有的参数：act_type。

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在全分辨率分支和低分辨率分支添加辅助分割头，并返回所有输出。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，包括卷积层和批归一化层的初始化，提高模型收敛性。

双分支结构保留：保留了 ContextNet 的核心双分支结构，包括全分辨率分支和低分辨率分支，以及它们之间的特征融合机制。
'''