"""
Paper:      ShelfNet for Fast Semantic Segmentation
Url:        https://arxiv.org/abs/1811.11254
Create by:  Simon
Date:       2025/06/05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, ConvBNAct, DeConvBNAct, Activation
from backbone import ResNet

__all__ = ['ShelfNet']

class SBlock(nn.Module):
    def __init__(self, channels, act_type):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.conv2 = ConvBNAct(channels, channels, 3, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x_l, x_v=0.):
        x = x_l + x_v
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        x += residual

        return self.act(x)


class EncoderBlock(nn.Module):
    def __init__(self, channels, act_type):
        super().__init__()
        self.block_A = SBlock(channels[0], act_type)
        self.down_A = ConvBNAct(channels[0], channels[1], 3, 2, act_type=act_type)

        self.block_B = SBlock(channels[1], act_type)
        self.down_B = ConvBNAct(channels[1], channels[2], 3, 2, act_type=act_type)

        self.block_C = SBlock(channels[2], act_type)
        self.down_C = ConvBNAct(channels[2], channels[3], 3, 2, act_type=act_type)

    def forward(self, x_a, x_b, x_c):
        x_a = self.block_A(x_a)
        x = self.down_A(x_a)

        x_b = self.block_B(x_b, x)
        x = self.down_B(x_b)

        x_c = self.block_C(x_c, x)
        x_d = self.down_C(x_c)

        return x_a, x_b, x_c, x_d


class DecoderBlock(nn.Module):
    def __init__(self, channels, act_type):
        super().__init__()
        self.block_D = SBlock(channels[3], act_type)
        self.up_D = DeConvBNAct(channels[3], channels[2], act_type=act_type)

        self.block_C = SBlock(channels[2], act_type)
        self.up_C = DeConvBNAct(channels[2], channels[1], act_type=act_type)

        self.block_B = SBlock(channels[1], act_type)
        self.up_B = DeConvBNAct(channels[1], channels[0], act_type=act_type)

        self.block_A = SBlock(channels[0], act_type)

    def forward(self, x_a, x_b, x_c, x_d, return_hid_feats=False):
        x_d = self.block_D(x_d)
        x = self.up_D(x_d)

        x_c = self.block_C(x_c, x)
        x = self.up_C(x_c)

        x_b = self.block_B(x_b, x)
        x = self.up_B(x_b)

        x_a = self.block_A(x_a, x)

        if return_hid_feats:
            return x_a, x_b, x_c
        else:
            return x_a


class ShelfNet(nn.Module):
    """
    ShelfNet for Fast Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 backbone_type='resnet34', hid_channels=[32,64,128,256], act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 骨干网络
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        else:
            raise NotImplementedError("目前只支持ResNet系列骨干网络")

        # 特征转换层
        self.conv_A = ConvBNAct(channels[0], hid_channels[0], 1, act_type=act_type)
        self.conv_B = ConvBNAct(channels[1], hid_channels[1], 1, act_type=act_type)
        self.conv_C = ConvBNAct(channels[2], hid_channels[2], 1, act_type=act_type)
        self.conv_D = ConvBNAct(channels[3], hid_channels[3], 1, act_type=act_type)

        # ShelfNet的主要组件
        self.decoder2 = DecoderBlock(hid_channels, act_type)
        self.encoder3 = EncoderBlock(hid_channels, act_type)
        self.decoder4 = DecoderBlock(hid_channels, act_type)

        # 分类头
        self.classifier = conv1x1(hid_channels[0], num_classes)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从decoder2输出
            self.aux_head1 = conv1x1(hid_channels[0], num_classes)
            
            # 辅助分支2 - 从encoder3后的decoder4中间层输出
            self.aux_head2 = conv1x1(hid_channels[1], num_classes)
        
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
        
        # 骨干网络特征提取
        x_a, x_b, x_c, x_d = self.backbone(x)

        # 特征转换
        x_a = self.conv_A(x_a)
        x_b = self.conv_B(x_b)
        x_c = self.conv_C(x_c)
        x_d = self.conv_D(x_d)

        # 第一个解码器
        x_a_dec2, x_b_dec2, x_c_dec2 = self.decoder2(x_a, x_b, x_c, x_d, return_hid_feats=True)
        
        # 第一个辅助输出
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x_a_dec2)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 编码器
        x_a_enc3, x_b_enc3, x_c_enc3, x_d_enc3 = self.encoder3(x_a_dec2, x_b_dec2, x_c_dec2)
        
        # 第二个解码器的中间过程
        if self.deep_supervision and self.training:
            # 获取decoder4的中间特征
            x_d_dec4 = self.decoder4.block_D(x_d_enc3)
            x_c_mid = self.decoder4.block_C(x_c_enc3, self.decoder4.up_D(x_d_dec4))
            x_b_mid = self.decoder4.block_B(x_b_enc3, self.decoder4.up_C(x_c_mid))
            
            # 第二个辅助输出
            aux2 = self.aux_head2(x_b_mid)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
            
            # 完成最后的解码过程
            x_final = self.decoder4.block_A(x_a_enc3, self.decoder4.up_B(x_b_mid))
        else:
            # 正常的第二个解码器
            x_final = self.decoder4(x_a_enc3, x_b_enc3, x_c_enc3, x_d_enc3)
        
        # 最终分类
        x = self.classifier(x_final)
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ShelfNet 特有的参数：

    backbone_type: 骨干网络类型，默认为 'resnet18'
    hid_channels: 各级特征的通道数，默认为 [32,64,128,256]
    act_type: 激活函数类型，默认为 'relu'

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到 decoder2 的输出
    第二个辅助头连接到 decoder4 的中间层输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 ShelfNet 的核心架构和特性，包括：

    骨干网络特征提取
    S-Block 设计
    多尺度特征融合
    编码器-解码器交替结构（"货架"结构）

依赖导入：从您现有的模块中导入所需组件，如 conv1x1, ConvBNAct, DeConvBNAct, Activation 和 ResNet。
'''