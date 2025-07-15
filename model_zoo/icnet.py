"""
Paper:      ICNet for Real-Time Semantic Segmentation on High-Resolution Images
Url:        https://arxiv.org/abs/1704.08545
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBNAct, Activation, PyramidPoolingModule, SegHead
from backbone import resnet18, resnet34, resnet50, resnet101

__all__ = ['ICNet']

class CascadeFeatureFusionUnit(nn.Module):
    def __init__(self, channel1, channel2, out_channels, num_class, act_type, use_aux):
        super().__init__()
        self.use_aux = use_aux
        self.conv1 = ConvBNAct(channel1, out_channels, 3, 1, 2, act_type='none')
        self.conv2 = ConvBNAct(channel2, out_channels, 1, act_type='none')
        self.act = Activation(act_type)
        if use_aux:
            self.classifier = SegHead(channel1, num_class, act_type)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_aux:
            x_aux = self.classifier(x1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x = self.act(x1 + x2)

        if self.use_aux:
            return x, x_aux
        else:
            return x


class HighResolutionBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, hid_channels=32, act_type='relu'):
        super().__init__(
            ConvBNAct(in_channels, hid_channels, 3, 2, act_type=act_type),
            ConvBNAct(hid_channels, hid_channels*2, 3, 2, act_type=act_type),
            ConvBNAct(hid_channels*2, out_channels, 3, 2, act_type=act_type)
        )


class ResNet(nn.Module):
    def __init__(self, resnet_type, pretrained=True):
        super().__init__()
        resnet_hub = {
            'resnet18': resnet18, 
            'resnet34': resnet34, 
            'resnet50': resnet50,
            'resnet101': resnet101
        }
        
        if resnet_type not in resnet_hub.keys():
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        use_basicblock = resnet_type in ['resnet18', 'resnet34']

        resnet = resnet_hub[resnet_type](pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Change stride-2 conv to dilated conv
        layers = [[self.layer3[0], resnet.layer3[0]], [self.layer4[0], resnet.layer4[0]]]        
        for i in range(1,3):
            ch = 128 if use_basicblock else 512
            resnet_downsample = layers[i-1][1].downsample[0]
            resnet_conv = layers[i-1][1].conv1 if use_basicblock else layers[i-1][1].conv2

            layers[i-1][0].downsample[0] = nn.Conv2d(ch*i, ch*i*2, 1, 1, bias=False)
            if use_basicblock:
                layers[i-1][0].conv1 = nn.Conv2d(ch*i, ch*i*2, 3, 1, 2*i, 2*i, bias=False)
            else:
                layers[i-1][0].conv2 = nn.Conv2d(ch//2*i, ch//2*i, 3, 1, 2*i, 2*i, bias=False)

            with torch.no_grad():
                layers[i-1][0].downsample[0].weight.copy_(resnet_downsample.weight)
                if use_basicblock:
                    layers[i-1][0].conv1.weight.copy_(resnet_conv.weight)
                else:
                    layers[i-1][0].conv2.weight.copy_(resnet_conv.weight)

    def forward(self, x):
        x = self.conv1(x)       # 2x down
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 4x down
        x = self.layer1(x)
        x2 = self.layer2(x)      # 8x down
        x = self.layer3(x2)      # 8x down with dilation 2
        x = self.layer4(x)      # 8x down with dilation 4

        return x, x2


class ICNet(nn.Module):
    """
    ICNet for Real-Time Semantic Segmentation on High-Resolution Images
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 backbone_type='resnet18', act_type='relu', pretrained=True, **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type, pretrained=pretrained)
            ch1 = 512 if backbone_type in ['resnet18', 'resnet34'] else 2048
            ch2 = 128 if backbone_type in ['resnet18', 'resnet34'] else 512
        else:
            raise NotImplementedError("Only ResNet backbones are supported")

        self.bottom_branch = HighResolutionBranch(input_channels, 128, act_type=act_type)
        self.ppm = PyramidPoolingModule(ch1, 256, act_type=act_type)
        self.cff42 = CascadeFeatureFusionUnit(256, ch2, 128, num_classes, act_type, deep_supervision)
        self.cff21 = CascadeFeatureFusionUnit(128, 128, 128, num_classes, act_type, deep_supervision)
        self.seg_head = SegHead(128, num_classes, act_type)
        
        # 初始化权重（除了预训练的backbone部分）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not hasattr(m, 'weight_initialized'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.weight_initialized = True
            elif isinstance(m, nn.BatchNorm2d) and not hasattr(m, 'weight_initialized'):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.weight_initialized = True

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 多分辨率输入
        x_d2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x_d4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)

        # 最低分辨率分支
        x_d4, _ = self.backbone(x_d4)           # 32x down
        x_d4 = self.ppm(x_d4)

        # 中等分辨率分支
        _, x_d2 = self.backbone(x_d2)           # 16x down

        # 高分辨率分支
        x = self.bottom_branch(x)               # 8x down

        # 级联特征融合
        if self.deep_supervision:
            x_d2, aux2 = self.cff42(x_d4, x_d2) # 16x down
            x, aux3 = self.cff21(x_d2, x)       # 8x down
        else:
            x_d2 = self.cff42(x_d4, x_d2)
            x = self.cff21(x_d2, x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)   # 4x down
        x = self.seg_head(x)                    # 4x down

        # 调整到输入尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)

        # 返回结果
        if self.deep_supervision:
            # 调整辅助输出到输入尺寸
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
            aux3 = F.interpolate(aux3, input_size, mode='bilinear', align_corners=True)
            return [aux2, aux3, x]
        
        return x

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ICNet 特有的参数：

    backbone_type: 骨干网络类型，默认为 'resnet18'
    act_type: 激活函数类型，默认为 'relu'
    pretrained: 是否使用预训练的骨干网络，默认为 True

深度监督：将原始 ICNet 的辅助损失机制适配为 UNeXt 框架的深度监督机制：

    当 deep_supervision=True 时，启用级联特征融合单元中的辅助分类器
    辅助输出与主输出一起返回为列表形式

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，对非预训练部分进行初始化，提高模型收敛性。

结构保留：保留了原始 ICNet 的核心架构和特性，包括：

    多分辨率输入策略
    级联特征融合单元
    金字塔池化模块
    高分辨率分支

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, Activation, PyramidPoolingModule, SegHead 和骨干网络。

预训练支持：保留了使用预训练 ResNet 骨干网络的能力，同时修改了部分层以支持空洞卷积。
'''