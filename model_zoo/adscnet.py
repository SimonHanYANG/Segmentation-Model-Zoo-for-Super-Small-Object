"""
Paper:      ADSCNet: asymmetric depthwise separable convolution for semantic 
            segmentation in real-time
Url:        https://link.springer.com/article/10.1007/s10489-019-01587-1
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct, DWConvBNAct, DeConvBNAct

__all__ = ['ADSCNet']

class ADSCModule(nn.Module):
    def __init__(self, channels, stride, dilation=1, act_type='relu'):
        super().__init__()
        assert stride in [1, 2], 'Unsupported stride type.\n'
        self.use_skip = stride == 1
        self.conv = nn.Sequential(
                        DWConvBNAct(channels, channels, (3, 1), stride, dilation, act_type, inplace=True),
                        conv1x1(channels, channels),
                        DWConvBNAct(channels, channels, (1, 3), 1, dilation, act_type, inplace=True),
                        conv1x1(channels, channels)
                    )
        if not self.use_skip:
            self.pool = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        x_conv = self.conv(x)

        if self.use_skip:
            x = x + x_conv
        else:
            x_pool = self.pool(x)
            x = torch.cat([x_conv, x_pool], dim=1)

        return x


class DDCC(nn.Module):
    def __init__(self, channels, dilations, act_type):
        super().__init__()
        assert len(dilations)==4, 'Length of dilations should be 4.\n'
        self.block1 = nn.Sequential(
                            nn.AvgPool2d(dilations[0], 1, dilations[0]//2),
                            ADSCModule(channels, 1, dilations[0], act_type)
                        )

        self.block2 = nn.Sequential(
                            conv1x1(2*channels, channels),
                            nn.AvgPool2d(dilations[1], 1, dilations[1]//2),
                            ADSCModule(channels, 1, dilations[1], act_type)
                        )

        self.block3 = nn.Sequential(
                            conv1x1(3*channels, channels),
                            nn.AvgPool2d(dilations[2], 1, dilations[2]//2),
                            ADSCModule(channels, 1, dilations[2], act_type)
                        )

        self.block4 = nn.Sequential(
                            conv1x1(4*channels, channels),
                            nn.AvgPool2d(dilations[3], 1, dilations[3]//2),
                            ADSCModule(channels, 1, dilations[3], act_type)
                        )

        self.conv_last = conv1x1(5*channels, channels)

    def forward(self, x):
        x1 = self.block1(x)

        x2 = torch.cat([x, x1], dim=1)
        x2 = self.block2(x2)

        x3 = torch.cat([x, x1, x2], dim=1)
        x3 = self.block3(x3)

        x4 = torch.cat([x, x1, x2, x3], dim=1)
        x4 = self.block4(x4)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.conv_last(x)

        return x


class ADSCNet(nn.Module):
    """
    ADSCNet: Asymmetric Depthwise Separable Convolution for Semantic Segmentation in Real-Time
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, act_type='relu6', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 初始层
        self.conv0 = ConvBNAct(input_channels, 32, 3, 2, act_type=act_type, inplace=True)
        
        # 编码器部分
        self.conv1 = ADSCModule(32, 1, act_type=act_type)
        self.conv2_4 = nn.Sequential(
                            ADSCModule(32, 1, act_type=act_type),
                            ADSCModule(32, 2, act_type=act_type),
                            ADSCModule(64, 1, act_type=act_type)
                        )
        self.conv5 = ADSCModule(64, 2, act_type=act_type)
        
        # 密集扩张卷积上下文模块
        self.ddcc = DDCC(128, [3, 5, 9, 13], act_type)
        
        # 解码器部分
        self.up1 = nn.Sequential(
                        DeConvBNAct(128, 64),
                        ADSCModule(64, 1, act_type=act_type)
                    )
        self.up2 = nn.Sequential(
                        ADSCModule(64, 1, act_type=act_type),
                        DeConvBNAct(64, 32)
                    )
        self.up3 = nn.Sequential(
                        ADSCModule(32, 1, act_type=act_type),
                        DeConvBNAct(32, 16)  # 减少通道数到16以匹配UNeXt的最终层
                    )
        
        # 最终输出层
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        
        # 深度监督输出
        if deep_supervision:
            self.dsv1 = nn.Conv2d(64, num_classes, kernel_size=1)
            self.dsv2 = nn.Conv2d(32, num_classes, kernel_size=1)
            self.dsv3 = nn.Conv2d(16, num_classes, kernel_size=1)

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
        
        # 编码器路径
        x = self.conv0(x)
        x1 = self.conv1(x)
        x4 = self.conv2_4(x1)
        x = self.conv5(x4)
        
        # 密集扩张卷积上下文
        x = self.ddcc(x)
        
        # 解码器路径
        x = self.up1(x)
        d1 = x  # 用于深度监督
        
        x = x + x4  # 跳跃连接
        x = self.up2(x)
        d2 = x  # 用于深度监督
        
        x = x + x1  # 跳跃连接
        x = self.up3(x)
        d3 = x  # 用于深度监督
        
        # 最终输出
        out = self.final(x)
        
        # 确保输出大小与输入大小匹配
        if out.size()[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        # 深度监督
        if self.deep_supervision:
            dsv1 = self.dsv1(d1)
            dsv2 = self.dsv2(d2)
            dsv3 = self.dsv3(d3)
            
            # 确保所有输出大小与输入大小匹配
            dsv1 = F.interpolate(dsv1, size=input_size, mode='bilinear', align_corners=True)
            dsv2 = F.interpolate(dsv2, size=input_size, mode='bilinear', align_corners=True)
            dsv3 = F.interpolate(dsv3, size=input_size, mode='bilinear', align_corners=True)
                
            return [dsv1, dsv2, dsv3, out]
        
        return out