"""
Paper:      ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
Url:        https://arxiv.org/abs/1803.06815
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, conv3x3, ConvBNAct, DeConvBNAct, Activation

__all__ = ['ESPNet']

class ESPModule(nn.Module):
    def __init__(self, in_channels, out_channels, K=5, ks=3, stride=1, act_type='prelu',):
        super().__init__()
        self.K = K
        self.stride = stride
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        channel_kn = out_channels // K
        channel_k1 = out_channels - (K -1) * channel_kn
        self.perfect_divisor = channel_k1 == channel_kn

        if self.perfect_divisor:
            self.conv_kn = conv1x1(in_channels, channel_kn, stride)
        else:
            self.conv_kn = conv1x1(in_channels, channel_kn, stride)
            self.conv_k1 = conv1x1(in_channels, channel_k1, stride)

        self.layers = nn.ModuleList()
        for k in range(1, K+1):
            dt = 2**(k-1)       # dilation
            channel = channel_k1 if k == 1 else channel_kn
            self.layers.append(ConvBNAct(channel, channel, ks, 1, dt, act_type=act_type))

    def forward(self, x):
        if self.use_skip:
            residual = x

        transform_feats = []
        if self.perfect_divisor:
            x = self.conv_kn(x)     # Reduce
            for i in range(self.K):
                transform_feats.append(self.layers[i](x))   # Split --> Transform

            for j in range(1, self.K):
                transform_feats[j] += transform_feats[j-1]      # Merge: Sum
        else:
            x1 = self.conv_k1(x)    # Reduce
            xn = self.conv_kn(x)    # Reduce
            transform_feats.append(self.layers[0](x1))      # Split --> Transform
            for i in range(1, self.K):
                transform_feats.append(self.layers[i](xn))   # Split --> Transform

            for j in range(2, self.K):
                transform_feats[j] += transform_feats[j-1]      # Merge: Sum

        x = torch.cat(transform_feats, dim=1)               # Merge: Concat

        if self.use_skip:
            x += residual

        return x


class L2Block(nn.Module):
    def __init__(self, in_channels, hid_channels, arch_type, alpha, use_skip, 
                    reinforce, act_type='prelu'):
        super().__init__()
        self.arch_type = arch_type
        self.alpha = alpha
        self.use_skip = use_skip
        self.reinforce = reinforce

        if reinforce:
            in_channels += 3

        self.conv1 = ESPModule(in_channels, hid_channels, stride=2, act_type=act_type)

        layers = []
        for _ in range(alpha):
            layers.append(ESPModule(hid_channels, hid_channels, act_type=act_type))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, x_input=None):
        x = self.conv1(x)
        if self.use_skip:
            skip = x

        x = self.layers(x)

        if self.use_skip:
            x = torch.cat([x, skip], dim=1)

        if self.reinforce:
            size = x.size()[2:]
            x_quarter = F.interpolate(x_input, size, mode='bilinear', align_corners=True)
            x = torch.cat([x, x_quarter], dim=1)

        return x


class L3Block(nn.Module):
    def __init__(self, in_channels, out_channels, arch_type, alpha, use_skip, 
                    reinforce, use_decoder, act_type='prelu'):
        super().__init__()
        self.arch_type = arch_type
        self.alpha = alpha
        self.use_skip = use_skip

        if reinforce:
            in_channels += 3

        self.conv1 = ESPModule(in_channels, 128, stride=2, act_type=act_type)

        layers = []
        for _ in range(alpha):
            layers.append(ESPModule(128, 128, act_type=act_type))
        self.layers = nn.Sequential(*layers)

        if use_decoder:
            self.conv_last = ConvBNAct(256, out_channels, 1, act_type=act_type)
        elif use_skip:
            self.conv_last = conv1x1(256, out_channels)
        else:
            self.conv_last = conv1x1(128, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_skip:
            skip = x

        x = self.layers(x)

        if self.use_skip:
            x = torch.cat([x, skip], dim=1)

        x = self.conv_last(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_class, l1_channel, l2_channel, act_type='prelu'):
        super().__init__()
        self.upconv_l3 = DeConvBNAct(num_class, num_class, act_type=act_type)
        self.conv_cat_l2 = ConvBNAct(l2_channel, num_class, 1)
        self.conv_l2 = ESPModule(2*num_class, num_class)
        self.upconv_l2 = DeConvBNAct(num_class, num_class, act_type=act_type)
        self.conv_cat_l1 = ConvBNAct(l1_channel, num_class, 1)
        self.conv_l1 = ESPModule(2*num_class, num_class)
        self.upconv_l1 = DeConvBNAct(num_class, num_class)

    def forward(self, x, x_l1, x_l2):
        x = self.upconv_l3(x)
        x_l2 = self.conv_cat_l2(x_l2)
        x = torch.cat([x, x_l2], dim=1)
        x = self.conv_l2(x)

        x = self.upconv_l2(x)
        x_l1 = self.conv_cat_l1(x_l1)
        x = torch.cat([x, x_l1], dim=1)
        x = self.conv_l1(x)

        x = self.upconv_l1(x)

        return x


class ESPNet(nn.Module):
    """
    ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 arch_type='espnet', K=5, alpha2=2, alpha3=8, 
                 block_channel=[16, 64, 128], act_type='prelu', **kwargs):
        super().__init__()
        arch_hub = ['espnet', 'espnet-a', 'espnet-b', 'espnet-c']
        if arch_type not in arch_hub:
            raise ValueError(f'Unsupport architecture type: {arch_type}.\n')
        
        self.arch_type = arch_type
        self.deep_supervision = deep_supervision

        use_skip = arch_type in ['espnet', 'espnet-b', 'espnet-c']
        reinforce = arch_type in ['espnet', 'espnet-c']
        use_decoder = arch_type in ['espnet']

        if arch_type == 'espnet-a':
            block_channel[2] = block_channel[1]

        self.use_skip = use_skip
        self.reinforce = reinforce
        self.use_decoder = use_decoder

        self.l1_block = ConvBNAct(input_channels, block_channel[0], 3, 2, act_type=act_type)
        self.l2_block = L2Block(block_channel[0], block_channel[1], arch_type, alpha2, use_skip, reinforce, act_type)
        self.l3_block = L3Block(block_channel[2], num_classes, arch_type, alpha3, use_skip, reinforce, use_decoder, act_type)

        if use_decoder:
            self.decoder = Decoder(num_classes, 19, 131, act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(block_channel[1] * 2 + 3 if reinforce else block_channel[1] * 2, 32, 3, 1, act_type=act_type),
                nn.Conv2d(32, num_classes, 1)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(256 if use_skip else 128, 64, 3, 1, act_type=act_type),
                nn.Conv2d(64, num_classes, 1)
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
        x_input = x
        
        # 第一阶段
        x = self.l1_block(x)
        
        if self.reinforce:
            size = x.size()[2:]
            x_half = F.interpolate(x_input, size, mode='bilinear', align_corners=True)
            x = torch.cat([x, x_half], dim=1)
            if self.use_decoder:
                x_l1 = x
        
        # 第二阶段
        if self.reinforce:
            x = self.l2_block(x, x_input)
            if self.use_decoder:
                x_l2 = x
        else:
            x = self.l2_block(x)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 第三阶段
        x = self.l3_block(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器阶段
        if self.use_decoder:
            x = self.decoder(x, x_l1, x_l2)
        else:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ESPNet 特有的参数：

    arch_type: 架构类型（'espnet', 'espnet-a', 'espnet-b', 'espnet-c'）
    K: ESP模块中的分支数
    alpha2: L2块中ESP模块的数量
    alpha3: L3块中ESP模块的数量
    block_channel: 各阶段的通道数
    act_type: 激活函数类型

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到L2块的输出
    第二个辅助头连接到L3块的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 ESPNet 的核心架构和特性，包括：

    ESP模块（高效空间金字塔卷积）
    不同的架构变体（espnet, espnet-a, espnet-b, espnet-c）
    强化连接和跳跃连接
    
依赖导入：从您现有的模块中导入所需组件，如 conv1x1, conv3x3, ConvBNAct, DeConvBNAct 和 Activation。
'''