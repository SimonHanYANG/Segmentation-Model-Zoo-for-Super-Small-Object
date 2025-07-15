"""
Paper:      LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1905.02423
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct, Activation, channel_shuffle
from .enet import InitialBlock as DownsampleUint

__all__ = ['LEDNet']

class SSnbtUnit(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super().__init__()
        assert channels % 2 == 0, 'Input channel should be multiple of 2.\n'
        split_channels = channels // 2
        self.split_channels = split_channels
        self.left_branch = nn.Sequential(
                                nn.Conv2d(split_channels, split_channels, (3, 1), padding=(1,0)),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (1, 3), act_type=act_type),
                                nn.Conv2d(split_channels, split_channels, (3, 1), 
                                            padding=(dilation,0), dilation=dilation),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (1, 3), dilation=dilation, act_type=act_type),
                            )

        self.right_branch = nn.Sequential(
                                nn.Conv2d(split_channels, split_channels, (1, 3), padding=(0,1)),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (3, 1), act_type=act_type),
                                nn.Conv2d(split_channels, split_channels, (1, 3), 
                                            padding=(0,dilation), dilation=dilation),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (3, 1), dilation=dilation, act_type=act_type),
                            )
        self.act = Activation(act_type)

    def forward(self, x):
        x_left = x[:, :self.split_channels].clone()
        x_right = x[:, self.split_channels:].clone()
        x_left = self.left_branch(x_left)
        x_right = self.right_branch(x_right)
        x_cat = torch.cat([x_left, x_right], dim=1)
        x += x_cat
        x = self.act(x)
        x = channel_shuffle(x)
        return x


class Encoder(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__(
            DownsampleUint(in_channels, 32, act_type),
            SSnbtUnit(32, 1, act_type=act_type),
            SSnbtUnit(32, 1, act_type=act_type),
            SSnbtUnit(32, 1, act_type=act_type),
            DownsampleUint(32, 64, act_type),
            SSnbtUnit(64, 1, act_type=act_type),
            SSnbtUnit(64, 1, act_type=act_type),
            DownsampleUint(64, out_channels, act_type),
            SSnbtUnit(out_channels, 1, act_type=act_type),
            SSnbtUnit(out_channels, 2, act_type=act_type),
            SSnbtUnit(out_channels, 5, act_type=act_type),
            SSnbtUnit(out_channels, 9, act_type=act_type),
            SSnbtUnit(out_channels, 2, act_type=act_type),
            SSnbtUnit(out_channels, 5, act_type=act_type),
            SSnbtUnit(out_channels, 9, act_type=act_type),
            SSnbtUnit(out_channels, 17, act_type=act_type),
        )


class AttentionPyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.left_conv1_1 = ConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type)
        self.left_conv1_2 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.left_conv2_1 = ConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type)
        self.left_conv2_2 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.left_conv3 = nn.Sequential(
                                ConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type),
                                ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
                            )

        self.mid_branch = ConvBNAct(in_channels, out_channels, act_type=act_type)
        self.right_branch = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                ConvBNAct(in_channels, out_channels, act_type=act_type),
                            )

    def forward(self, x):
        size0 = x.size()[2:]

        x_left = self.left_conv1_1(x)
        size1 = x_left.size()[2:]

        x_left2 = self.left_conv2_1(x_left)
        size2 = x_left2.size()[2:]

        x_left3 = self.left_conv3(x_left2)
        x_left3 = F.interpolate(x_left3, size2, mode='bilinear', align_corners=True)

        x_left2 = self.left_conv2_2(x_left2)
        x_left2 += x_left3
        x_left2 = F.interpolate(x_left2, size1, mode='bilinear', align_corners=True)

        x_left = self.left_conv1_2(x_left)
        x_left += x_left2
        x_left = F.interpolate(x_left, size0, mode='bilinear', align_corners=True)

        x_mid = self.mid_branch(x)
        x_mid = torch.mul(x_left, x_mid)

        x_right = self.right_branch(x)
        x_right = F.interpolate(x_right, size0, mode='bilinear', align_corners=True)

        x_mid += x_right
        return x_mid


class LEDNet(nn.Module):
    """
    LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 编码器部分
        self.encoder = Encoder(input_channels, 128, act_type)
        
        # 解码器部分 (APN)
        self.apn = AttentionPyramidNetwork(128, num_classes, act_type)
        
        # 深度监督分支
        if deep_supervision:
            # 辅助分支1 - 从第一个下采样后
            self.aux_head1 = nn.Sequential(
                ConvBNAct(32, 32, 3, 1, act_type=act_type),
                nn.Conv2d(32, num_classes, 1)
            )
            
            # 辅助分支2 - 从第二个下采样后
            self.aux_head2 = nn.Sequential(
                ConvBNAct(64, 64, 3, 1, act_type=act_type),
                nn.Conv2d(64, num_classes, 1)
            )
            
            # 保存中间特征的钩子
            self.encoder_features = {}
            self._register_hooks()
        
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
    
    def _register_hooks(self):
        """为深度监督注册前向钩子，捕获中间特征"""
        if not self.deep_supervision:
            return
            
        def get_activation(name):
            def hook(module, input, output):
                self.encoder_features[name] = output
            return hook
            
        # 注册钩子到第一个和第二个下采样单元后
        self.encoder[3].register_forward_hook(get_activation('stage1'))  # 第一个下采样后
        self.encoder[6].register_forward_hook(get_activation('stage2'))  # 第二个下采样后

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 编码器前向传播
        x = self.encoder(x)
        
        # 解码器前向传播
        x = self.apn(x)
        
        # 调整到输入尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 深度监督
        if self.deep_supervision and self.training:
            # 处理辅助输出
            aux1 = self.aux_head1(self.encoder_features['stage1'])
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
            
            aux2 = self.aux_head2(self.encoder_features['stage2'])
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
            
            return [aux1, aux2, x]
        
        return x
    

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 LEDNet 特有的参数：

    act_type: 激活函数类型，默认为 'relu'

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到第一个下采样单元后（32通道）
    第二个辅助头连接到第二个下采样单元后（64通道）
    使用钩子机制捕获中间特征，避免修改原始编码器结构

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 LEDNet 的核心架构和特性，包括：

    轻量级编码器结构
    非对称卷积分解（Split-Shuffle-Non-bottleneck）单元
    注意力金字塔网络(APN)解码器

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, Activation, channel_shuffle 和 DownsampleUint（从ENet）。
'''