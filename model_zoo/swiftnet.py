"""
Paper:      In Defense of Pre-trained ImageNet Architectures for Real-time 
            Semantic Segmentation of Road-driving Images
Url:        https://arxiv.org/abs/1903.08469
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, PWConvBNAct, ConvBNAct, PyramidPoolingModule
from backbone import ResNet, Mobilenetv2

__all__ = ['SwiftNet']

class Decoder(nn.Module):
    def __init__(self, channels, num_class, act_type):
        super().__init__()
        self.up_stage3 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.up_stage2 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.up_stage1 = ConvBNAct(channels, num_class, 3, act_type=act_type)

    def forward(self, x, x1, x2, x3):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x3
        x = self.up_stage3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x2
        x = self.up_stage2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x1
        x = self.up_stage1(x)

        return x


class SwiftNet(nn.Module):
    """
    SwiftNet: In Defense of Pre-trained ImageNet Architectures for Real-time 
              Semantic Segmentation of Road-driving Images
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 backbone_type='resnet18', up_channels=128, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 骨干网络
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        elif backbone_type == 'mobilenet_v2':
            self.backbone = Mobilenetv2()
            channels = [24, 32, 96, 320]
        else:
            raise NotImplementedError("目前仅支持ResNet系列和MobileNetV2骨干网络")

        # 连接层
        self.connection1 = ConvBNAct(channels[0], up_channels, 1, act_type=act_type)
        self.connection2 = ConvBNAct(channels[1], up_channels, 1, act_type=act_type)
        self.connection3 = ConvBNAct(channels[2], up_channels, 1, act_type=act_type)
        
        # 空间金字塔池化模块
        self.spp = PyramidPoolingModule(channels[3], up_channels, act_type, bias=True)
        
        # 解码器
        self.decoder = Decoder(up_channels, num_classes, act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从stage3输出
            self.aux_head1 = nn.Sequential(
                ConvBNAct(up_channels, up_channels//2, 3, act_type=act_type),
                conv1x1(up_channels//2, num_classes)
            )
            
            # 辅助分支2 - 从stage2输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(up_channels, up_channels//2, 3, act_type=act_type),
                conv1x1(up_channels//2, num_classes)
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

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        size = x.size()[2:]
        
        # 骨干网络特征提取
        x1, x2, x3, x4 = self.backbone(x)
        
        # 特征转换
        x1_c = self.connection1(x1)
        x2_c = self.connection2(x2)
        x3_c = self.connection3(x3)
        x4_c = self.spp(x4)
        
        # 解码过程
        # 第一次上采样和融合
        x_up3 = F.interpolate(x4_c, scale_factor=2, mode='bilinear', align_corners=True)
        x_up3 = x_up3 + x3_c
        x_up3 = self.decoder.up_stage3(x_up3)
        
        # 第一个辅助输出（如果启用深度监督）
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x_up3)
            aux1 = F.interpolate(aux1, size, mode='bilinear', align_corners=True)
        
        # 第二次上采样和融合
        x_up2 = F.interpolate(x_up3, scale_factor=2, mode='bilinear', align_corners=True)
        x_up2 = x_up2 + x2_c
        x_up2 = self.decoder.up_stage2(x_up2)
        
        # 第二个辅助输出（如果启用深度监督）
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x_up2)
            aux2 = F.interpolate(aux2, size, mode='bilinear', align_corners=True)
        
        # 最终上采样和融合
        x_up1 = F.interpolate(x_up2, scale_factor=2, mode='bilinear', align_corners=True)
        x_up1 = x_up1 + x1_c
        x = self.decoder.up_stage1(x_up1)
        
        # 调整到原始大小
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x
    

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 SwiftNet 特有的参数：

    backbone_type: 骨干网络类型，可选 'resnet18', 'resnet34', 'resnet50', 'resnet101' 或 'mobilenet_v2'，默认为 'resnet18'
    up_channels: 上采样通道数，默认为 128
    act_type: 激活函数类型，默认为 'relu'
    
深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到 up_stage3 的输出
    第二个辅助头连接到 up_stage2 的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 SwiftNet 的核心架构和特性，包括：

    预训练骨干网络
    特征连接层
    空间金字塔池化模块
    轻量级解码器
    跳跃连接

依赖导入：从您现有的模块中导入所需组件，如 conv1x1, PWConvBNAct, ConvBNAct, PyramidPoolingModule, ResNet 和 Mobilenetv2。
'''
