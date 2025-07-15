"""
Paper:      LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation
Url:        https://arxiv.org/abs/1912.06683
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct
from backbone import ResNet, Mobilenetv2

__all__ = ['LiteSeg']

class DASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        hid_channels = in_channels // 5
        last_channels = in_channels - hid_channels * 4
        self.stage1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.stage2 = ConvBNAct(in_channels, hid_channels, 3, dilation=3, act_type=act_type)
        self.stage3 = ConvBNAct(in_channels, hid_channels, 3, dilation=6, act_type=act_type)
        self.stage4 = ConvBNAct(in_channels, hid_channels, 3, dilation=9, act_type=act_type)
        self.stage5 = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(in_channels, last_channels)
                        )
        self.conv = ConvBNAct(2*in_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x)
        x5 = self.stage5(x)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=True)

        x = self.conv(torch.cat([x, x1, x2, x3, x4, x5], dim=1))
        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=256):
        super().__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            ConvBNAct(hid_channels, hid_channels//2, 3, act_type=act_type),
            conv1x1(hid_channels//2, num_class)
        )


class LiteSeg(nn.Module):
    """
    LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 backbone_type='mobilenet_v2', act_type='relu', pretrained=True, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 骨干网络
        if backbone_type == 'mobilenet_v2':
            self.backbone = Mobilenetv2(pretrained=pretrained)
            channels = [320, 32]  # 最后一层和低层特征的通道数
        elif 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type, pretrained=pretrained)
            channels = [512, 128] if backbone_type in ['resnet18', 'resnet34'] else [2048, 512]
        else:
            raise NotImplementedError("Only MobileNetV2 and ResNet backbones are supported")

        # DASPP模块
        self.daspp = DASPPModule(channels[0], 512, act_type)
        
        # 分割头
        self.seg_head = SegHead(512 + channels[1], num_classes, act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从骨干网络的中间层
            self.aux_head1 = nn.Sequential(
                ConvBNAct(channels[1], 128, 3, act_type=act_type),
                conv1x1(128, num_classes)
            )
            
            # 辅助分支2 - 从DASPP输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(512, 256, 3, act_type=act_type),
                conv1x1(256, num_classes)
            )
        
        # 初始化权重（除了预训练的backbone部分）
        self._init_weights()

    def _init_weights(self):
        """初始化非骨干网络部分的权重"""
        for m in [self.daspp, self.seg_head]:
            if isinstance(m, nn.Sequential):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
            elif isinstance(m, nn.Module):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
        
        if self.deep_supervision:
            for m in [self.aux_head1, self.aux_head2]:
                for module in m.modules():
                    if isinstance(module, nn.Conv2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 骨干网络前向传播
        _, x1, _, x = self.backbone(x)
        size1 = x1.size()[2:]
        
        # 第一个辅助输出 - 从低层特征
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x1)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # DASPP模块
        x_daspp = self.daspp(x)
        
        # 第二个辅助输出 - 从DASPP输出
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x_daspp)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        # 上采样DASPP输出到低层特征尺寸
        x_daspp = F.interpolate(x_daspp, size1, mode='bilinear', align_corners=True)
        
        # 特征融合
        x = torch.cat([x_daspp, x1], dim=1)
        
        # 分割头
        x = self.seg_head(x)
        
        # 上采样到输入尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 LiteSeg 特有的参数：

    backbone_type: 骨干网络类型，支持 'mobilenet_v2' 和各种 'resnet' 变体，默认为 'mobilenet_v2'
    act_type: 激活函数类型，默认为 'relu'
    pretrained: 是否使用预训练的骨干网络，默认为 True

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到骨干网络的低层特征
    第二个辅助头连接到DASPP模块的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，对非预训练部分进行初始化，提高模型收敛性。

结构保留：保留了原始 LiteSeg 的核心架构和特性，包括：

    轻量级骨干网络（MobileNetV2或ResNet）
    密集自适应空间金字塔池化(DASPP)模块
    特征融合和高效分割头

依赖导入：从您现有的模块中导入所需组件，如 conv1x1, ConvBNAct, ResNet 和 Mobilenetv2。
'''