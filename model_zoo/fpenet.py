"""
Paper:      Feature Pyramid Encoding Network for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/1909.08599
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import DWConvBNAct, ConvBNAct

__all__ = ['FPENet']

def build_blocks(block, channels, num_block, expansion, act_type):
    layers = []
    for i in range(num_block):
        layers.append(block(channels, channels, expansion, 1, act_type=act_type))
    return nn.Sequential(*layers)


class FPEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, dilations=[1,2,4,8], 
                    act_type='relu'):
        super().__init__()
        assert len(dilations) > 0, 'Length of dilations should be larger than 0.\n'
        self.K = len(dilations)
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        expand_channels = out_channels * expansion
        self.ch = expand_channels // self.K

        self.conv_init = ConvBNAct(in_channels, expand_channels, 1, act_type=act_type, inplace=True)

        self.layers = nn.ModuleList()
        for i in range(self.K):
            self.layers.append(DWConvBNAct(self.ch, self.ch, 3, stride, dilations[i], act_type=act_type))

        self.conv_last = ConvBNAct(expand_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        if self.use_skip:
            residual = x

        x = self.conv_init(x)

        transform_feats = []
        for i in range(self.K):
            transform_feats.append(self.layers[i](x[:, i*self.ch:(i+1)*self.ch]))

        for j in range(1, self.K):
            transform_feats[j] += transform_feats[j-1]

        x = torch.cat(transform_feats, dim=1)

        x = self.conv_last(x)

        if self.use_skip:
            x += residual

        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        self.conv = ConvBNAct(1, 1, 1, act_type=act_type, inplace=True)

    def forward(self, x):
        x = self.conv(torch.mean(x, dim=1, keepdim=True))
        return x


class ChannelAttentionBlock(nn.Sequential):
    def __init__(self, channels, act_type):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(channels, channels, 1, act_type=act_type, inplace=True)
        )


class MEUModule(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, act_type):
        super().__init__()
        self.conv_low = ConvBNAct(low_channels, out_channels, 1, act_type=act_type, inplace=True)
        self.conv_high = ConvBNAct(high_channels, out_channels, 1, act_type=act_type, inplace=True)
        self.sa = SpatialAttentionBlock(act_type)
        self.ca = ChannelAttentionBlock(out_channels, act_type)

    def forward(self, x_low, x_high):
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)

        x_sa = self.sa(x_low)
        x_ca = self.ca(x_high)

        x_low = x_low * x_ca
        x_high = F.interpolate(x_high, scale_factor=2, mode='bilinear', align_corners=True)
        x_high = x_high * x_sa

        return x_low + x_high


class FPENet(nn.Module):
    """
    Feature Pyramid Encoding Network for Real-time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 p=3, q=9, k=4, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器部分
        self.stage1 = nn.Sequential(
                            ConvBNAct(input_channels, 16, 3, 2, act_type=act_type, inplace=True),
                            FPEBlock(16, 16, 1, 1, act_type=act_type)
                        )
        self.stage2_0 = FPEBlock(16, 32, k, 2, act_type=act_type)
        self.stage2 = build_blocks(FPEBlock, 32, p-1, k, act_type)
        self.stage3_0 = FPEBlock(32, 64, k, 2, act_type=act_type)
        self.stage3 = build_blocks(FPEBlock, 64, q-1, k, act_type)
        
        # 解码器部分
        self.decoder2 = MEUModule(32, 64, 64, act_type)
        self.decoder1 = MEUModule(16, 64, 32, act_type)
        self.final = ConvBNAct(32, num_classes, 1, act_type=act_type, inplace=True)
        
        # 深度监督分支
        if deep_supervision:
            # 辅助分支1 - 从stage2
            self.aux_head1 = nn.Sequential(
                ConvBNAct(32, 32, 3, 1, act_type=act_type, inplace=True),
                nn.Conv2d(32, num_classes, 1)
            )
            
            # 辅助分支2 - 从stage3
            self.aux_head2 = nn.Sequential(
                ConvBNAct(64, 64, 3, 1, act_type=act_type, inplace=True),
                nn.Conv2d(64, num_classes, 1)
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
        
        # 编码器前向传播
        x1 = self.stage1(x)
        x = self.stage2_0(x1)
        x2 = self.stage2(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x2)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        x = self.stage3_0(x2)
        x3 = self.stage3(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x3)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器前向传播
        x = self.decoder2(x2, x3)
        x = self.decoder1(x1, x)
        x = self.final(x)
        
        # 调整到输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 FPENet 特有的参数：

    p: stage2中FPEBlock的数量
    q: stage3中FPEBlock的数量
    k: 扩展因子
    act_type: 激活函数类型

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到stage2的输出
    第二个辅助头连接到stage3的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 FPENet 的核心架构和特性，包括：

    特征金字塔编码模块（FPE Block）
    多尺度特征提取
    多尺度增强单元（MEU Module）
    空间和通道注意力机制
    
依赖导入：从您现有的模块中导入所需组件，如 DWConvBNAct 和 ConvBNAct。

'''