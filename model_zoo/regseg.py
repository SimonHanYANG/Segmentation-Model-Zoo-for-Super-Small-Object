"""
Paper:      Rethinking Dilated Convolution for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/2111.09957
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct, Activation

__all__ = ['RegSeg']

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r1=None, r2=None, 
                    g=16, se_ratio=0.25, act_type='relu'):
        super().__init__()
        assert stride in [1, 2], f'Unsupported stride: {stride}'
        self.stride = stride

        self.conv1 = ConvBNAct(in_channels, out_channels, 1, act_type=act_type)
        if stride == 1:
            assert in_channels == out_channels, 'In_channels should be the same as out_channels when stride = 1'
            split_ch = out_channels // 2
            assert split_ch % g == 0, 'Group width `g` should be evenly divided by split_ch'
            groups = split_ch // g
            self.split_channels = split_ch
            self.conv_left = ConvBNAct(split_ch, split_ch, 3, dilation=r1, groups=groups, act_type=act_type)
            self.conv_right = ConvBNAct(split_ch, split_ch, 3, dilation=r2, groups=groups, act_type=act_type)
        else:   # stride == 2
            assert out_channels % g == 0, 'Group width `g` should be evenly divided by out_channels'
            groups = out_channels // g
            self.conv_left = ConvBNAct(out_channels, out_channels, 3, 2, groups=groups, act_type=act_type)
            self.conv_skip = nn.Sequential(
                                nn.AvgPool2d(2, 2, 0),
                                ConvBNAct(in_channels, out_channels, 1, act_type='none')
                            )
        self.conv2 = nn.Sequential(
                        SEBlock(out_channels, se_ratio, act_type),
                        ConvBNAct(out_channels, out_channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.stride == 1:
            x_left = self.conv_left(x[:, :self.split_channels])
            x_right = self.conv_right(x[:,self.split_channels:])
            x = torch.cat([x_left, x_right], dim=1)
        else:
            x = self.conv_left(x)
            residual = self.conv_skip(residual)

        x = self.conv2(x)
        x += residual

        return self.act(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio, act_type):
        super().__init__()
        squeeze_channels = int(channels * reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(
                            nn.Linear(channels, squeeze_channels),
                            Activation(act_type),
                            nn.Linear(squeeze_channels, channels),
                            Activation('sigmoid')
                        )

    def forward(self, x):
        residual = x
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.se_block(x).unsqueeze(-1).unsqueeze(-1)
        x = x * residual

        return x


class Decoder(nn.Module):
    def __init__(self, num_class, d4_channel, d8_channel, d16_channel, act_type):
        super().__init__()
        self.conv_d16 = ConvBNAct(d16_channel, 128, 1, act_type=act_type)
        self.conv_d8_stage1 = ConvBNAct(d8_channel, 128, 1, act_type=act_type)
        self.conv_d4_stage1 = ConvBNAct(d4_channel, 8, 1, act_type=act_type)
        self.conv_d8_stage2 = ConvBNAct(128, 64, 3, act_type=act_type)
        self.conv_d4_stage2 = nn.Sequential(
                                    ConvBNAct(64+8, 64, 3, act_type=act_type),
                                    conv1x1(64, num_class)
                                )

    def forward(self, x_d4, x_d8, x_d16):
        size_d4 = x_d4.size()[2:]
        size_d8 = x_d8.size()[2:]

        x_d16 = self.conv_d16(x_d16)
        x_d16 = F.interpolate(x_d16, size_d8, mode='bilinear', align_corners=True)

        x_d8 = self.conv_d8_stage1(x_d8)
        x_d8 += x_d16
        x_d8 = self.conv_d8_stage2(x_d8)
        x_d8 = F.interpolate(x_d8, size_d4, mode='bilinear', align_corners=True)

        x_d4 = self.conv_d4_stage1(x_d4)
        x_d4 = torch.cat([x_d4, x_d8], dim=1)
        x_d4 = self.conv_d4_stage2(x_d4)

        return x_d4


def build_blocks(block, in_channels, out_channels, stride, num_block, dilations=None, act_type='relu'):
    layers = []
    # 第一个块可能需要下采样
    layers.append(block(in_channels, out_channels, stride, 
                        r1=1 if dilations is None else dilations[0][0], 
                        r2=1 if dilations is None else dilations[0][1], 
                        act_type=act_type))
    
    # 剩余的块
    for i in range(1, num_block):
        r1 = 1 if dilations is None else dilations[i-1][0]
        r2 = 1 if dilations is None else dilations[i-1][1]
        layers.append(block(out_channels, out_channels, 1, r1=r1, r2=r2, act_type=act_type))
    
    return nn.Sequential(*layers)


class RegSeg(nn.Module):
    """
    RegSeg: Rethinking Dilated Convolution for Real-time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 dilations=None, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        if dilations is None:
            dilations = [[1,1], [1,2], [1,2], [1,3], [2,3], [2,7], [2,3],
                         [2,6], [2,5], [2,9], [2,11], [4,7], [5,14]]
        else:
            if len(dilations) != 13:
                raise ValueError("Dilation pairs' length should be 13\n")

        # Backbone-1
        self.conv_init = ConvBNAct(input_channels, 32, 3, 2, act_type=act_type)

        # Backbone-2
        self.stage_d4 = DBlock(32, 48, 2, act_type=act_type)

        # Backbone-3
        self.stage_d8 = build_blocks(DBlock, 48, 128, 2, 3, None, act_type)

        # Backbone-4 & 5
        layers = [DBlock(128, 256, 2, act_type=act_type)]
        for i in range(12):
            layers.append(DBlock(256, 256, 1, r1=dilations[i][0], r2=dilations[i][1], act_type=act_type))
        layers.append(DBlock(256, 320, 2, r1=dilations[-1][0], r2=dilations[-1][1], act_type=act_type))
        self.stage_d16 = nn.Sequential(*layers)

        # Decoder
        self.decoder = Decoder(num_classes, 48, 128, 320, act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从stage_d8输出
            self.aux_head1 = nn.Sequential(
                ConvBNAct(128, 64, 3, act_type=act_type),
                ConvBNAct(64, 32, 3, act_type=act_type),
                conv1x1(32, num_classes)
            )
            
            # 辅助分支2 - 从stage_d16中间层输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(256, 128, 3, act_type=act_type),
                ConvBNAct(128, 64, 3, act_type=act_type),
                conv1x1(64, num_classes)
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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 骨干网络
        x = self.conv_init(x)               # 2x down
        x_d4 = self.stage_d4(x)             # 4x down
        x_d8 = self.stage_d8(x_d4)          # 8x down
        
        # 第一个辅助输出 - 从x_d8
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x_d8)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 继续骨干网络
        # 获取中间特征用于第二个辅助输出
        if self.deep_supervision and self.training:
            # 提取stage_d16的中间层特征
            x_mid = self.stage_d16[:7](x_d8)  # 取前7层
            aux2 = self.aux_head2(x_mid)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
            # 继续处理剩余层
            x_d16 = self.stage_d16[7:](x_mid)
        else:
            x_d16 = self.stage_d16(x_d8)        # 16x down
        
        # 解码器
        x = self.decoder(x_d4, x_d8, x_d16)  # 4x down
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 RegSeg 特有的参数：

    dilations: 扩张卷积率列表，默认为原论文中的设置
    act_type: 激活函数类型，默认为 'relu'

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到 stage_d8 的输出
    第二个辅助头连接到 stage_d16 的中间层输出（第7层）

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 RegSeg 的核心架构和特性，包括：

    DBlock 设计（分组卷积和双路径扩张卷积）
    SE 注意力模块
    多尺度特征融合解码器
    扩张卷积率设计

依赖导入：从您现有的模块中导入所需组件，如 conv1x1, ConvBNAct, 和 Activation。

辅助函数：添加了 build_blocks 辅助函数，用于构建具有相同结构的多个模块。

'''