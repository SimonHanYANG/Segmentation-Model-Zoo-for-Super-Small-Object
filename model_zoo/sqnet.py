"""
Paper:      Speeding up Semantic Segmentation for Autonomous Driving
Url:        https://openreview.net/pdf?id=S1uHiFyyg
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, DeConvBNAct, Activation

__all__ = ['SQNet']

class FireModule(nn.Module):
    def __init__(self, in_channels, sq_channels, ex1_channels, ex3_channels, act_type):
        super().__init__()
        self.conv_squeeze = ConvBNAct(in_channels, sq_channels, 1, act_type=act_type)
        self.conv_expand1 = ConvBNAct(sq_channels, ex1_channels, 1, act_type=act_type)
        self.conv_expand3 = ConvBNAct(sq_channels, ex3_channels, 3, act_type=act_type)

    def forward(self, x):
        x = self.conv_squeeze(x)
        x1 = self.conv_expand1(x)
        x3 = self.conv_expand3(x)
        x = torch.cat([x1, x3], dim=1)

        return x


class ParallelDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, act_type):
        super().__init__()
        assert len(dilations) == 4, 'Length of dilations should be 4.\n'
        self.conv0 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[0], act_type=act_type)
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[1], act_type=act_type)
        self.conv2 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[2], act_type=act_type)
        self.conv3 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[3], act_type=act_type)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x0 + x1 + x2 + x3

        return x


class BypassRefinementModule(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, act_type):
        super().__init__()
        self.conv_low = ConvBNAct(low_channels, low_channels, 3, act_type=act_type)
        self.conv_cat = ConvBNAct(low_channels + high_channels, out_channels, 3, act_type=act_type)

    def forward(self, x_low, x_high):
        x_low = self.conv_low(x_low)
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv_cat(x)

        return x


class SQNet(nn.Module):
    """
    SQNet: Speeding up Semantic Segmentation for Autonomous Driving
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 act_type='elu', dilations=[1,2,4,8], **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Encoder, SqueezeNet-1.1
        self.conv = ConvBNAct(input_channels, 64, 3, 2, act_type=act_type)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.fire1 = nn.Sequential(
                            FireModule(64, 16, 64, 64, act_type),
                            FireModule(128, 16, 64, 64, act_type)
                        )
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.fire2 = nn.Sequential(
                            FireModule(128, 32, 128, 128, act_type),
                            FireModule(256, 32, 128, 128, act_type)
                        )
        self.pool3 = nn.MaxPool2d(3, 2, 1)
        self.fire3 = nn.Sequential(
                            FireModule(256, 48, 192, 192, act_type),
                            FireModule(384, 48, 192, 192, act_type),
                            FireModule(384, 64, 256, 256, act_type),
                            FireModule(512, 64, 256, 256, act_type)
                        )
        
        # Decoder
        self.pdc = ParallelDilatedConv(512, 128, dilations, act_type)
        self.up1 = DeConvBNAct(128, 128, act_type=act_type)
        self.refine1 = BypassRefinementModule(256, 128, 128, act_type)
        self.up2 = DeConvBNAct(128, 128, act_type=act_type)
        self.refine2 = BypassRefinementModule(128, 128, 64, act_type=act_type)
        self.up3 = DeConvBNAct(64, 64, act_type=act_type)
        self.refine3 = BypassRefinementModule(64, 64, num_classes, act_type=act_type)
        self.up4 = DeConvBNAct(num_classes, num_classes, act_type=act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从refine1输出
            self.aux_head1 = nn.Sequential(
                ConvBNAct(128, 64, 3, act_type=act_type),
                DeConvBNAct(64, 32, act_type=act_type),
                DeConvBNAct(32, num_classes, act_type=act_type)
            )
            
            # 辅助分支2 - 从refine2输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(64, 32, 3, act_type=act_type),
                DeConvBNAct(32, num_classes, act_type=act_type)
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
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 编码器
        x1 = self.conv(x)
        x = self.pool1(x1)
        x2 = self.fire1(x)
        x = self.pool2(x2)
        x3 = self.fire2(x)
        x = self.pool3(x3)
        x = self.fire3(x)
        
        # 解码器
        x = self.pdc(x)
        x = self.up1(x)
        x = self.refine1(x3, x)
        
        # 第一个辅助输出
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 继续解码
        x = self.up2(x)
        x = self.refine2(x2, x)
        
        # 第二个辅助输出
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        # 最终解码
        x = self.up3(x)
        x = self.refine3(x1, x)
        x = self.up4(x)
        
        # 如果输入大小与输出不同，进行插值调整
        if x.size()[2:] != input_size:
            x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 SQNet 特有的参数：

    act_type: 激活函数类型，默认为 'elu'（SQNet 原始论文中使用的激活函数）
    dilations: 并行扩张卷积的扩张率列表，默认为 [1,2,4,8]

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到 refine1 的输出
    第二个辅助头连接到 refine2 的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 SQNet 的核心架构和特性，包括：

    SqueezeNet-1.1 风格的编码器
    Fire 模块设计
    并行扩张卷积
    旁路精细化模块（Bypass Refinement Module）

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, DeConvBNAct, 和 Activation。
'''