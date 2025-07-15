"""
Paper:      ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1606.02147
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn

from modules import conv1x1, ConvBNAct, Activation

__all__ = ['ENet']

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, kernel_size=3, **kwargs):
        super().__init__()
        assert out_channels > in_channels, 'out_channels should be larger than in_channels.\n'
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, kernel_size, 2, act_type=act_type, **kwargs)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return x        


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type, act_type='prelu', 
                    upsample_type='regular', dilation=1, drop_p=0.1, shrink_ratio=0.25):
        super().__init__()
        self.conv_type = conv_type
        hid_channels = int(in_channels * shrink_ratio)
        if conv_type == 'regular':
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    ConvBNAct(hid_channels, hid_channels),
                                )
        elif conv_type == 'downsampling':
            self.left_pool = nn.MaxPool2d(2, 2, return_indices=True)
            self.left_conv = ConvBNAct(in_channels, out_channels, 1)         
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 3, 2),
                                    ConvBNAct(hid_channels, hid_channels),
                                )
        elif conv_type == 'upsampling':
            self.left_conv = ConvBNAct(in_channels, out_channels, 1)
            self.left_pool = nn.MaxUnpool2d(2, 2)
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    Upsample(hid_channels, hid_channels, scale_factor=2,  
                                                kernel_size=3, upsample_type=upsample_type),
                                )
        elif conv_type == 'dilate':
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    ConvBNAct(hid_channels, hid_channels, dilation=dilation),
                                )
        elif conv_type == 'asymmetric':
            self.right_init_conv = nn.Sequential(
                                    ConvBNAct(in_channels, hid_channels, 1),
                                    ConvBNAct(hid_channels, hid_channels, (5,1)),
                                    ConvBNAct(hid_channels, hid_channels, (1,5)),
                                )
        else:
            raise ValueError(f'[!] Unsupport convolution type: {conv_type}')

        self.right_last_conv = nn.Sequential(
                                    conv1x1(hid_channels, out_channels),
                                    nn.Dropout(drop_p)
                            )
        self.act = Activation(act_type)

    def forward(self, x, indices=None):
        x_right = self.right_last_conv(self.right_init_conv(x))
        if self.conv_type == 'downsampling':
            x_left, indices = self.left_pool(x)
            x_left = self.left_conv(x_left)
            x = self.act(x_left + x_right)
            return x, indices

        elif self.conv_type == 'upsampling':
            if indices is None:
                raise ValueError('Upsampling-type conv needs pooling indices.')

            x_left = self.left_conv(x)
            x_left = self.left_pool(x_left, indices)
            x = self.act(x_left + x_right)

        else:
            x = self.act(x + x_right)    # shortcut

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                    upsample_type=None, act_type='relu'):
        super().__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            padding = (kernel_size - 1) // 2
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                                stride=scale_factor, padding=padding, 
                                                output_padding=1, bias=False)
        else:
            self.up_conv = nn.Sequential(
                                    ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                            )

    def forward(self, x):
        return self.up_conv(x)


class BottleNeck1(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='prelu', drop_p=0.01):
        super().__init__()
        self.conv_pool = Bottleneck(in_channels, out_channels, 'downsampling', act_type, drop_p=drop_p)
        self.conv_regular = nn.Sequential(
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
            Bottleneck(out_channels, out_channels, 'regular', act_type, drop_p=drop_p),
        )

    def forward(self, x):
        x, indices = self.conv_pool(x)
        x = self.conv_regular(x)

        return x, indices


class BottleNeck23(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='prelu', downsample=True):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.conv_pool = Bottleneck(in_channels, out_channels, 'downsampling', act_type=act_type)

        self.conv_regular = nn.Sequential(
            Bottleneck(out_channels, out_channels, 'regular', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=2),
            Bottleneck(out_channels, out_channels, 'asymmetric', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=4),
            Bottleneck(out_channels, out_channels, 'regular', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=8),
            Bottleneck(out_channels, out_channels, 'asymmetric', act_type),
            Bottleneck(out_channels, out_channels, 'dilate', act_type, dilation=16),
        )

    def forward(self, x):
        if self.downsample:
            x, indices = self.conv_pool(x)
        x = self.conv_regular(x)

        if self.downsample:
            return x, indices

        return x


class BottleNeck45(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='prelu', upsample_type=None, 
                    extra_conv=False):
        super().__init__()
        self.extra_conv = extra_conv
        self.conv_unpool = Bottleneck(in_channels, out_channels, 'upsampling', act_type, upsample_type)
        self.conv_regular = Bottleneck(out_channels, out_channels, 'regular', act_type)

        if extra_conv:
            self.conv_extra = Bottleneck(out_channels, out_channels, 'regular', act_type)

    def forward(self, x, indices):
        x = self.conv_unpool(x, indices)
        x = self.conv_regular(x)

        if self.extra_conv:
            x = self.conv_extra(x)

        return x


class ENet(nn.Module):
    """
    ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 act_type='prelu', upsample_type='deconvolution', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 初始块
        self.initial = InitialBlock(input_channels, 16, act_type)
        
        # 编码器部分
        self.bottleneck1 = BottleNeck1(16, 64, act_type)
        self.bottleneck2 = BottleNeck23(64, 128, act_type, True)
        self.bottleneck3 = BottleNeck23(128, 128, act_type, False)
        
        # 解码器部分
        self.bottleneck4 = BottleNeck45(128, 64, act_type, upsample_type, True)
        self.bottleneck5 = BottleNeck45(64, 16, act_type, upsample_type, False)
        self.fullconv = Upsample(16, num_classes, scale_factor=2, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, 1, act_type=act_type),
                conv1x1(32, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(128, 64, 3, 1, act_type=act_type),
                conv1x1(64, num_classes)
            )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 初始块
        x = self.initial(x)
        
        # 编码器阶段1
        x, indices1 = self.bottleneck1(x)  # 2x downsample
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = nn.functional.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段2
        x, indices2 = self.bottleneck2(x)  # 2x downsample
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = nn.functional.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段3
        x = self.bottleneck3(x)
        
        # 解码器阶段
        x = self.bottleneck4(x, indices2)  # 2x upsample
        x = self.bottleneck5(x, indices1)  # 2x upsample
        x = self.fullconv(x)
        
        # 确保输出尺寸与输入一致
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ENet 特有的参数：

    act_type: 激活函数类型（默认为'prelu'）
    upsample_type: 上采样方法（默认为'deconvolution'）

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的编码器阶段添加两个辅助分割头：

    第一个辅助头连接到第一个瓶颈块的输出
    第二个辅助头连接到第二个瓶颈块的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。这是通过在最终输出层添加额外的插值操作实现的，因为原始 ENet 可能不会精确恢复到输入尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 ENet 的核心架构和特性，包括瓶颈模块设计、最大池化索引重用和非对称卷积。
'''