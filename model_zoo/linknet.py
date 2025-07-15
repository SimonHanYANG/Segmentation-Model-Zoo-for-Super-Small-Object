"""
Paper:      LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
Url:        https://arxiv.org/abs/1707.03718
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, DeConvBNAct, Activation
from backbone import ResNet

__all__ = ['LinkNet']

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, scale_factor=2):
        super().__init__()
        hid_channels = in_channels // 4
        self.conv1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        if scale_factor > 1:
            self.full_conv = DeConvBNAct(hid_channels, hid_channels, scale_factor, act_type=act_type)
        else:
            self.full_conv = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.conv2 = ConvBNAct(hid_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.full_conv(x)
        x = self.conv2(x)

        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, scale_factor=2):
        hid_channels = in_channels // 2
        super().__init__(
                DeConvBNAct(in_channels, hid_channels, scale_factor, act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type),
                DeConvBNAct(hid_channels, num_class, scale_factor, act_type=act_type)
        )


class LinkNet(nn.Module):
    """
    LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 backbone_type='resnet18', act_type='relu', pretrained=True, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 骨干网络
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type, pretrained=pretrained)
            channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        else:
            raise NotImplementedError("Only ResNet backbones are supported")

        # 解码器块
        self.dec_block4 = DecoderBlock(channels[3], channels[2], act_type)
        self.dec_block3 = DecoderBlock(channels[2], channels[1], act_type)
        self.dec_block2 = DecoderBlock(channels[1], channels[0], act_type)
        self.dec_block1 = DecoderBlock(channels[0], channels[0], act_type, scale_factor=1)
        
        # 分割头
        self.seg_head = SegHead(channels[0], num_classes, act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从第三个解码器块输出
            self.aux_head1 = nn.Sequential(
                ConvBNAct(channels[1], channels[1]//2, 3, 1, act_type=act_type),
                nn.Conv2d(channels[1]//2, num_classes, 1)
            )
            
            # 辅助分支2 - 从第二个解码器块输出
            self.aux_head2 = nn.Sequential(
                ConvBNAct(channels[0], channels[0]//2, 3, 1, act_type=act_type),
                nn.Conv2d(channels[0]//2, num_classes, 1)
            )
        
        # 初始化权重（除了预训练的backbone部分）
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """初始化解码器部分的权重"""
        for m in [self.dec_block1, self.dec_block2, self.dec_block3, self.dec_block4, self.seg_head]:
            if isinstance(m, nn.Sequential):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)
            elif isinstance(m, nn.Module):
                for module in m.modules():
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0)
                    elif isinstance(module, nn.BatchNorm2d):
                        nn.init.constant_(module.weight, 1)
                        nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 编码器前向传播
        x_1, x_2, x_3, x_4 = self.backbone(x)
        
        # 解码器前向传播
        x = self.dec_block4(x_4)
        
        # 第一个跳跃连接
        x = x + x_3
        x = self.dec_block3(x)
        
        # 用于深度监督的第一个辅助输出
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 第二个跳跃连接
        x = x + x_2
        x = self.dec_block2(x)
        
        # 用于深度监督的第二个辅助输出
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 第三个跳跃连接
        x = x + x_1
        x = self.dec_block1(x)
        
        # 分割头
        x = self.seg_head(x)
        
        # 确保输出尺寸与输入相同
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 LinkNet 特有的参数：

    backbone_type: 骨干网络类型，默认为 'resnet18'
    act_type: 激活函数类型，默认为 'relu'
    pretrained: 是否使用预训练的骨干网络，默认为 True

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到第三个解码器块的输出（对应于中层特征）
    第二个辅助头连接到第二个解码器块的输出（对应于较浅层特征）

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，对非预训练部分进行初始化，提高模型收敛性。

结构保留：保留了原始 LinkNet 的核心架构和特性，包括：

    编码器-解码器架构
    跳跃连接
    解码器块的高效设计（通道降维-上采样-通道恢复）

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, DeConvBNAct, Activation 和 ResNet。
'''