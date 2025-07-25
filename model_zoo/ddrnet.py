"""
Paper:      Deep Dual-resolution Networks for Real-time and Accurate Semantic 
            Segmentation of Road Scenes
Url:        https://arxiv.org/abs/2101.06085
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct, Activation, SegHead

__all__ = ['DDRNet']

class RB(nn.Module):
    # Building sequential residual basic blocks
    def __init__(self, in_channels, out_channels, stride=1, act_type='relu'):
        super().__init__()
        self.downsample = (stride > 1) or (in_channels != out_channels)
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride, act_type=act_type)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act_type='none')

        if self.downsample:
            self.conv_down = ConvBNAct(in_channels, out_channels, 1, stride, act_type='none')
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_down(x)
        out += identity
        out = self.act(out)

        return out


class RBB(nn.Module):
    # Building single residual bottleneck block
    def __init__(self, in_channels, out_channels, stride=1, act_type='relu'):
        super().__init__()
        self.downsample = (stride > 1) or (in_channels != out_channels)
        self.conv1 = ConvBNAct(in_channels, in_channels, 1, act_type=act_type)
        self.conv2 = ConvBNAct(in_channels, in_channels, 3, stride, act_type=act_type)
        self.conv3 = ConvBNAct(in_channels, out_channels, 1, act_type='none')

        if self.downsample:
            self.conv_down = ConvBNAct(in_channels, out_channels, 1, stride, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            identity = self.conv_down(x)
        out += identity
        out = self.act(out)

        return out


def build_blocks(block, in_channels, out_channels, stride, repeat_times, act_type):
    layers = [block(in_channels, out_channels, stride, act_type=act_type)]
    for _ in range(1, repeat_times):
        layers.append(block(out_channels, out_channels, 1, act_type=act_type))
    return nn.Sequential(*layers)


class Stage2(nn.Module):
    def __init__(self, init_channel, repeat_times, act_type='relu'):
        super().__init__()
        in_channels = init_channel
        out_channels = init_channel

        layers = [ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type)]
        for _ in range(repeat_times):
            layers.append(RB(out_channels, out_channels, 1, act_type))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Stage3(nn.Module):
    def __init__(self, init_channel, repeat_times, act_type='relu'):
        super().__init__()
        in_channels = init_channel
        out_channels = init_channel * 2

        self.conv = build_blocks(RB, in_channels, out_channels, 2, repeat_times, act_type)

    def forward(self, x):
        return self.conv(x)


class BilateralFusion(nn.Module):
    def __init__(self, low_res_channels, high_res_channels, stride, act_type='relu'):
        super().__init__()
        self.conv_low = ConvBNAct(low_res_channels, high_res_channels, 1, act_type='none')
        self.conv_high = ConvBNAct(high_res_channels, low_res_channels, 3, stride, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x_low, x_high):
        size = x_high.size()[2:]
        fuse_low = self.conv_low(x_low)
        fuse_high = self.conv_high(x_high)
        x_low = self.act(x_low + fuse_high)

        fuse_low = F.interpolate(fuse_low, size, mode='bilinear', align_corners=True)
        x_high = self.act(x_high + fuse_low)

        return x_low, x_high


class Stage4(nn.Module):
    def __init__(self, init_channel, repeat_times1, repeat_times2, act_type='relu'):
        super().__init__()
        in_channels = init_channel * 2
        low_res_channels = init_channel * 4
        high_res_channels = init_channel * 2
        if low_res_channels < high_res_channels:
            raise ValueError('Low resolution channel should be more than high resolution channel.\n')

        self.low_conv1 = build_blocks(RB, in_channels, low_res_channels, 2, repeat_times1, act_type)
        self.high_conv1 = build_blocks(RB, in_channels, high_res_channels, 1, repeat_times1, act_type)
        self.bilateral_fusion1 = BilateralFusion(low_res_channels, high_res_channels, 2)

        self.extra_conv = repeat_times2 > 0
        if self.extra_conv:
            self.low_conv2 = build_blocks(RB, low_res_channels, low_res_channels, 1, repeat_times2, act_type)
            self.high_conv2 = build_blocks(RB, high_res_channels, high_res_channels, 1, repeat_times2, act_type)
            self.bilateral_fusion2 = BilateralFusion(low_res_channels, high_res_channels, 2)

    def forward(self, x):
        x_low = self.low_conv1(x)
        x_high = self.high_conv1(x)
        x_low, x_high = self.bilateral_fusion1(x_low, x_high)

        if self.extra_conv:
            x_low = self.low_conv2(x_low)
            x_high = self.high_conv2(x_high)
            x_low, x_high = self.bilateral_fusion2(x_low, x_high)

        return x_low, x_high


class DAPPM(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        hid_channels = int(in_channels // 4)

        self.conv0 = ConvBNAct(in_channels, out_channels, 1, act_type=act_type)
        self.conv1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.pool2 = self._build_pool_layers(in_channels, hid_channels, 5, 2)
        self.conv2 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool3 = self._build_pool_layers(in_channels, hid_channels, 9, 4)
        self.conv3 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool4 = self._build_pool_layers(in_channels, hid_channels, 17, 8)
        self.conv4 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool5 = self._build_pool_layers(in_channels, hid_channels, -1, -1)
        self.conv5 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.conv_last = ConvBNAct(hid_channels*5, out_channels, 1, act_type=act_type)

    def _build_pool_layers(self, in_channels, out_channels, kernel_size, stride):
        layers = []
        if kernel_size == -1:
            layers.append(nn.AdaptiveAvgPool2d(1))
        else:
            padding = (kernel_size - 1) // 2
            layers.append(nn.AvgPool2d(kernel_size, stride, padding))
        layers.append(conv1x1(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]
        y0 = self.conv0(x)
        y1 = self.conv1(x)

        y2 = self.pool2(x)
        y2 = F.interpolate(y2, size, mode='bilinear', align_corners=True)
        y2 = self.conv2(y1 + y2)

        y3 = self.pool3(x)
        y3 = F.interpolate(y3, size, mode='bilinear', align_corners=True)
        y3 = self.conv3(y2 + y3)

        y4 = self.pool4(x)
        y4 = F.interpolate(y4, size, mode='bilinear', align_corners=True)
        y4 = self.conv4(y3 + y4)

        y5 = self.pool5(x)
        y5 = F.interpolate(y5, size, mode='bilinear', align_corners=True)
        y5 = self.conv5(y4 + y5)

        x = self.conv_last(torch.cat([y1, y2, y3, y4, y5], dim=1)) + y0

        return x


class Stage5(nn.Module):
    def __init__(self, init_channel, repeat_times1, repeat_times2, act_type='relu'):
        super().__init__()
        low_in_channels = init_channel * 4
        high_in_channels = init_channel * 2
        low_res_channels1 = init_channel * 8
        high_res_channels1 = init_channel * 2
        low_res_channels2 = init_channel * 16
        high_res_channels2 = init_channel * 4
        if (low_in_channels < high_in_channels) or (low_res_channels1 < high_res_channels1) or (low_res_channels2 < high_res_channels2):
            raise ValueError('Low resolution channel should be more than high resolution channel.\n')

        self.low_conv1 = build_blocks(RB, low_in_channels, low_res_channels1, 2, repeat_times1, act_type)
        self.high_conv1 = build_blocks(RB, high_in_channels, high_res_channels1, 1, repeat_times1, act_type)
        self.bilateral_fusion = BilateralFusion(low_res_channels1, high_res_channels1, 4)

        self.low_conv2 = build_blocks(RBB, low_res_channels1, low_res_channels2, 2, repeat_times2, act_type)
        self.high_conv2 = build_blocks(RBB, high_res_channels1, high_res_channels2, 1, repeat_times2, act_type)
        self.dappm = DAPPM(low_res_channels2, high_res_channels2)

    def forward(self, x_low, x_high):
        size = x_high.size()[2:]

        x_low = self.low_conv1(x_low)
        x_high = self.high_conv1(x_high)
        x_low, x_high = self.bilateral_fusion(x_low, x_high)

        x_low = self.low_conv2(x_low)
        x_low = self.dappm(x_low)
        x_low = F.interpolate(x_low, size, mode='bilinear', align_corners=True)

        x_high = self.high_conv2(x_high) + x_low

        return x_high


class DDRNet(nn.Module):
    """
    Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 arch_type='DDRNet-23-slim', act_type='relu', **kwargs):
        super().__init__()
        arch_hub = {
            'DDRNet-23-slim': {'init_channel': 32, 'repeat_times': [2, 2, 2, 0, 2, 1]},
            'DDRNet-23': {'init_channel': 64, 'repeat_times': [2, 2, 2, 0, 2, 1]},
            'DDRNet-39': {'init_channel': 64, 'repeat_times': [3, 4, 3, 3, 3, 1]},
        }
        
        if arch_type not in arch_hub.keys():
            raise ValueError(f'Unsupport architecture type: {arch_type}.\n')

        init_channel = arch_hub[arch_type]['init_channel']
        repeat_times = arch_hub[arch_type]['repeat_times']
        self.deep_supervision = deep_supervision

        # 主要网络结构
        self.conv1 = ConvBNAct(input_channels, init_channel, 3, 2, act_type=act_type)
        self.conv2 = Stage2(init_channel, repeat_times[0], act_type)
        self.conv3 = Stage3(init_channel, repeat_times[1], act_type)
        self.conv4 = Stage4(init_channel, repeat_times[2], repeat_times[3], act_type)
        self.conv5 = Stage5(init_channel, repeat_times[4], repeat_times[5], act_type)
        self.seg_head = SegHead(init_channel*4, num_classes, act_type)
        
        # 辅助分割头
        if self.deep_supervision:
            self.aux_head = SegHead(init_channel*2, num_classes, act_type)
        
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
        
        # 网络前向传播
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_low, x_high = self.conv4(x)
        
        # 辅助分割头
        if self.deep_supervision:
            x_aux = self.aux_head(x_high)
            x_aux = F.interpolate(x_aux, input_size, mode='bilinear', align_corners=True)
        
        # 主分割头
        x = self.conv5(x_low, x_high)
        x = self.seg_head(x)
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [x_aux, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 DDRNet 特有的参数：arch_type, act_type。

架构变体：保留了原始 DDRNet 的三种变体：

    DDRNet-23-slim：轻量级版本，初始通道数为32
    DDRNet-23：标准版本，初始通道数为64
    DDRNet-39：更深的版本，初始通道数为64，更多的重复块

深度监督：将原始实现中的 use_aux 参数替换为 UNeXt 框架中的 deep_supervision 参数，保持功能一致。当 deep_supervision=True 时，添加辅助分割头并返回所有输出。

输出格式：当 deep_supervision=True 时，模型返回一个包含辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。
'''