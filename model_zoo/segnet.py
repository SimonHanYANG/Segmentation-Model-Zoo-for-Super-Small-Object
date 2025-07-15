"""
Paper:      SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
Url:        https://arxiv.org/abs/1511.00561
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct

__all__ = ['SegNet']

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu', extra_conv=False):
        super().__init__()
        layers = [ConvBNAct(in_channels, out_channels, 3, act_type=act_type, inplace=True),
                  ConvBNAct(out_channels, out_channels, 3, act_type=act_type, inplace=True)]
        if extra_conv:
            layers.append(ConvBNAct(out_channels, out_channels, 3, act_type=act_type, inplace=True))
        self.conv = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x)
        return x, indices


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu', extra_conv=False):
        super().__init__()
        self.pool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        hid_channel = in_channels if extra_conv else out_channels

        layers = [ConvBNAct(in_channels, in_channels, 3, act_type=act_type, inplace=True),
                  ConvBNAct(in_channels, hid_channel, 3, act_type=act_type, inplace=True)]

        if extra_conv:
            layers.append(ConvBNAct(hid_channel, out_channels, 3, act_type=act_type, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, indices):
        x = self.pool(x, indices)
        x = self.conv(x)

        return x


class SegNet(nn.Module):
    """
    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 hid_channel=64, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器
        self.down_stage1 = DownsampleBlock(input_channels, hid_channel, act_type, False)
        self.down_stage2 = DownsampleBlock(hid_channel, hid_channel*2, act_type, False)
        self.down_stage3 = DownsampleBlock(hid_channel*2, hid_channel*4, act_type, True)
        self.down_stage4 = DownsampleBlock(hid_channel*4, hid_channel*8, act_type, True)
        self.down_stage5 = DownsampleBlock(hid_channel*8, hid_channel*8, act_type, True)
        
        # 解码器
        self.up_stage5 = UpsampleBlock(hid_channel*8, hid_channel*8, act_type, True)
        self.up_stage4 = UpsampleBlock(hid_channel*8, hid_channel*4, act_type, True)
        self.up_stage3 = UpsampleBlock(hid_channel*4, hid_channel*2, act_type, True)
        self.up_stage2 = UpsampleBlock(hid_channel*2, hid_channel, act_type, False)
        self.up_stage1 = UpsampleBlock(hid_channel, hid_channel, act_type, False)
        self.classifier = ConvBNAct(hid_channel, num_classes, 3, act_type=act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从up_stage3输出
            self.aux_head1 = nn.Sequential(
                ConvBNAct(hid_channel*2, hid_channel, 3, act_type=act_type),
                ConvBNAct(hid_channel, num_classes, 3, act_type=act_type)
            )
            
            # 辅助分支2 - 从up_stage2输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(hid_channel, num_classes, 3, act_type=act_type)
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
        input_size = x.size()[2:]
        
        # 编码器
        x, indices1 = self.down_stage1(x)
        x, indices2 = self.down_stage2(x)
        x, indices3 = self.down_stage3(x)
        x, indices4 = self.down_stage4(x)
        x, indices5 = self.down_stage5(x)
        
        # 解码器
        x = self.up_stage5(x, indices5)
        x = self.up_stage4(x, indices4)
        x = self.up_stage3(x, indices3)
        
        # 第一个辅助输出
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 继续解码
        x = self.up_stage2(x, indices2)
        
        # 第二个辅助输出
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        # 最终解码
        x = self.up_stage1(x, indices1)
        x = self.classifier(x)
        
        # 如果输入大小与输出不同，进行插值调整
        if x.size()[2:] != input_size:
            x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 SegNet 特有的参数：

    hid_channel: 第一层的隐藏通道数，默认为 64
    act_type: 激活函数类型，默认为 'relu'

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到 up_stage3 的输出
    第二个辅助头连接到 up_stage2 的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 SegNet 的核心架构和特性，包括：

    对称的编码器-解码器结构
    最大池化索引的重用
    多层卷积块设计

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct。

修复了 UpsampleBlock 中的 Bug：原代码在 extra_conv=True 时，最后一个卷积层的输入通道数应为 hid_channel 而不是 in_channels。
'''