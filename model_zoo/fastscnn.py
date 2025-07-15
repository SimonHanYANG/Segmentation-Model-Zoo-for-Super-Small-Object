"""
Paper:      Fast-SCNN: Fast Semantic Segmentation Network
Url:        https://arxiv.org/abs/1902.04502
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, DSConvBNAct, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation, PyramidPoolingModule

__all__ = ['FastSCNN']

class LearningToDownsample(nn.Sequential):
    def __init__(self, in_channels, out_channels, hid_channels=[32, 48], act_type='relu'):
        super().__init__(
            ConvBNAct(in_channels, hid_channels[0], 3, 2, act_type=act_type),
            DSConvBNAct(hid_channels[0], hid_channels[1], 3, 2, act_type=act_type),
            DSConvBNAct(hid_channels[1], out_channels, 3, 2, act_type=act_type),
        )


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        inverted_residual_setting = [
                # t, c, n, s
                [6, 64, 3, 2],
                [6, 96, 2, 2],
                [6, 128, 3, 1],
            ]

        # Building inverted residual blocks
        features = []
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
        self.bottlenecks = nn.Sequential(*features)

        self.ppm = PyramidPoolingModule(in_channels, out_channels, act_type=act_type, bias=True)

    def forward(self, x):
        x = self.bottlenecks(x)
        x = self.ppm(x)

        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, higher_channels, lower_channels, out_channels, act_type='relu'):
        super().__init__()
        self.higher_res_conv = conv1x1(higher_channels, out_channels)
        self.lower_res_conv = nn.Sequential(
                                DWConvBNAct(lower_channels, lower_channels, 3, 1, act_type=act_type),
                                conv1x1(lower_channels, out_channels)
                            )
        self.non_linear = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                        )

    def forward(self, higher_res_feat, lower_res_feat):
        size = higher_res_feat.size()[2:]
        higher_res_feat = self.higher_res_conv(higher_res_feat)
        lower_res_feat = F.interpolate(lower_res_feat, size, mode='bilinear', align_corners=True)
        lower_res_feat = self.lower_res_conv(lower_res_feat)
        x = self.non_linear(higher_res_feat + lower_res_feat)

        return x


class Classifier(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type='relu'):
        super().__init__(
            DSConvBNAct(in_channels, in_channels, 3, 1, act_type=act_type),
            DSConvBNAct(in_channels, in_channels, 3, 1, act_type=act_type),
            PWConvBNAct(in_channels, num_class, act_type=act_type),
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


class FastSCNN(nn.Module):
    """
    Fast-SCNN: Fast Semantic Segmentation Network
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 主要网络组件
        self.learning_to_downsample = LearningToDownsample(input_channels, 64, act_type=act_type)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128, act_type=act_type)
        self.feature_fusion = FeatureFusionModule(64, 128, 128, act_type=act_type)
        self.classifier = Classifier(128, num_classes, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            # 辅助分支1 - 从下采样模块
            self.aux_head1 = nn.Sequential(
                DSConvBNAct(64, 64, 3, 1, act_type=act_type),
                PWConvBNAct(64, num_classes, act_type=act_type)
            )
            
            # 辅助分支2 - 从全局特征提取器
            self.aux_head2 = nn.Sequential(
                DSConvBNAct(128, 64, 3, 1, act_type=act_type),
                PWConvBNAct(64, num_classes, act_type=act_type)
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
        
        # 学习下采样模块
        higher_res_feat = self.learning_to_downsample(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(higher_res_feat)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 全局特征提取器
        lower_res_feat = self.global_feature_extractor(higher_res_feat)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(lower_res_feat)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 特征融合和分类
        x = self.feature_fusion(higher_res_feat, lower_res_feat)
        x = self.classifier(x)
        
        # 调整到输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 FastSCNN 特有的参数：

    act_type: 激活函数类型  

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到学习下采样模块的输出
    第二个辅助头连接到全局特征提取器的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 FastSCNN 的核心架构和特性，包括：

    学习下采样模块（Learning to Downsample）
    全局特征提取器（Global Feature Extractor）
    特征融合模块（Feature Fusion Module）
    分类器（Classifier）

依赖导入：从您现有的模块中导入所需组件，如 conv1x1, DSConvBNAct, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation 和 PyramidPoolingModule。
'''