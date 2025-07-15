import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ConvBNAct
from backbone import ResNet

__all__ = ['UNetPlusPlus']

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        
        # 第一个卷积块
        self.conv1 = ConvBNAct(in_channels, out_channels // 2, 3, act_type=act_type)
        
        # 第二个卷积块
        self.conv2 = ConvBNAct(out_channels // 2, out_channels // 2, 3, act_type=act_type)
        
        # 第三个卷积块
        self.conv3 = ConvBNAct(out_channels // 2, out_channels, 3, act_type=act_type)
 
    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class UNetPlusPlus(nn.Module):
    """
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False,
                 backbone_type='resnet34', act_type='relu', **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.deep_supervision = deep_supervision
        
        # 骨干网络
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            if backbone_type in ['resnet18', 'resnet34']:
                filters = [64, 64, 128, 256, 512]  # 第一个是stem的输出通道数
            else:
                filters = [64, 256, 512, 1024, 2048]  # 第一个是stem的输出通道数
        else:
            raise NotImplementedError("目前仅支持ResNet系列骨干网络")
        
        # 深度监督权重参数
        if deep_supervision:
            self.mix = nn.Parameter(torch.FloatTensor(5))
            self.mix.data.fill_(1)
        
        # 解码器
        # 第一层解码器
        self.decoder0_1 = DecoderBlock(filters[0] + filters[1], filters[0], act_type)
        
        # 第二层解码器
        self.decoder1_1 = DecoderBlock(filters[1] + filters[2], filters[1], act_type)
        self.decoder0_2 = DecoderBlock(filters[0] * 2 + filters[1], filters[0], act_type)
        
        # 第三层解码器
        self.decoder2_1 = DecoderBlock(filters[2] + filters[3], filters[2], act_type)
        self.decoder1_2 = DecoderBlock(filters[1] * 2 + filters[2], filters[1], act_type)
        self.decoder0_3 = DecoderBlock(filters[0] * 3 + filters[1], filters[1], act_type)
        
        # 第四层解码器
        self.decoder3_1 = DecoderBlock(filters[3] + filters[4], filters[3], act_type)
        self.decoder2_2 = DecoderBlock(filters[2] * 2 + filters[3], filters[2], act_type)
        self.decoder1_3 = DecoderBlock(filters[1] * 3 + filters[2], filters[2], act_type)
        self.decoder0_4 = DecoderBlock(filters[0] * 3 + filters[1] + filters[2], filters[2], act_type)
        
        # 分割头
        self.logit1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.logit2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.logit3 = nn.Conv2d(filters[1], num_classes, kernel_size=1)
        self.logit4 = nn.Conv2d(filters[2], num_classes, kernel_size=1)
        
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
    
    def _upsize(self, x, scale_factor=2):
        """上采样函数"""
        return F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # 保存输入尺寸用于最终调整
        size = x.size()[2:]
        
        # 骨干网络特征提取
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.backbone.extract_features(x)
        
        # 解码过程
        # 第一层解码
        x0_1 = self.decoder0_1([x0_0, self._upsize(x1_0)])
        
        # 第二层解码
        x1_1 = self.decoder1_1([x1_0, self._upsize(x2_0)])
        x0_2 = self.decoder0_2([x0_0, x0_1, self._upsize(x1_1)])
        
        # 第三层解码
        x2_1 = self.decoder2_1([x2_0, self._upsize(x3_0)])
        x1_2 = self.decoder1_2([x1_0, x1_1, self._upsize(x2_1)])
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2, self._upsize(x1_2)])
        
        # 第四层解码
        x3_1 = self.decoder3_1([x3_0, self._upsize(x4_0)])
        x2_2 = self.decoder2_2([x2_0, x2_1, self._upsize(x3_1)])
        x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, self._upsize(x2_2)])
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2, x0_3, self._upsize(x1_3)])
        
        # 分割头
        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)
        
        # 调整到原始大小
        logit1 = F.interpolate(logit1, size=size, mode='bilinear', align_corners=True)
        logit2 = F.interpolate(logit2, size=size, mode='bilinear', align_corners=True)
        logit3 = F.interpolate(logit3, size=size, mode='bilinear', align_corners=True)
        logit4 = F.interpolate(logit4, size=size, mode='bilinear', align_corners=True)
        
        # 根据深度监督模式返回结果
        if self.deep_supervision and self.training:
            # 使用混合权重
            logit = self.mix[1] * logit1 + self.mix[2] * logit2 + self.mix[3] * logit3 + self.mix[4] * logit4
            return [logit1, logit2, logit3, logit4, logit]
        else:
            # 仅返回最终输出
            return logit4


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 UNet++ 特有的参数：

    backbone_type: 骨干网络类型，可选 'resnet18', 'resnet34', 'resnet50', 'resnet101'，默认为 'resnet34'
    act_type: 激活函数类型，默认为 'relu'

骨干网络：使用您框架中已有的 ResNet 类作为骨干网络，并根据不同的 ResNet 变体调整通道数。

深度监督：保留了原始 UNet++ 的深度监督机制，当 deep_supervision=True 时，模型会输出多个分割结果并使用可学习的权重参数进行加权组合。

解码器模块：重新实现了 DecoderBlock 模块，使用您框架中的 ConvBNAct 模块来简化代码。

输出格式：

    当 deep_supervision=True 且处于训练模式时，模型返回一个包含四个中间输出和一个加权组合输出的列表
    当 deep_supervision=False 或处于评估模式时，模型仅返回最终输出

输出大小调整：确保所有输出都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 UNet++ 的核心架构和特性，包括：

    嵌套的U型结构
    密集的跳跃连接
    深度监督机制

'''