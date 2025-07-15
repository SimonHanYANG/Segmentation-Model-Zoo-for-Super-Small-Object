"""
Paper:      ESPNetv2: A Light-weight, Power Efficient, and General Purpose 
            Convolutional Neural Network
Url:        https://arxiv.org/abs/1811.11431
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, DSConvBNAct, PWConvBNAct, ConvBNAct, PyramidPoolingModule, SegHead

__all__ = ['ESPNetv2']

def build_blocks(block, channels, num_block, act_type='relu'):
    layers = []
    for _ in range(num_block):
        layers.append(block(channels, act_type=act_type))
    return nn.Sequential(*layers)


class EESPModule(nn.Module):
    def __init__(self, channels, K=4, ks=3, stride=1, act_type='prelu'):
        super().__init__()
        assert channels % K == 0, 'Input channels should be integer multiples of K.\n'

        self.K = K
        channel_k = channels // K
        self.use_skip = stride == 1

        self.conv_init = nn.Conv2d(channels, channel_k, 1, groups=K, bias=False)
        self.layers = nn.ModuleList()
        for k in range(1, K+1):
            dt = 2**(k-1)       # dilation
            self.layers.append(DSConvBNAct(channel_k, channel_k, ks, stride, dt, act_type=act_type))
        self.conv_last = nn.Conv2d(channels, channels, 1, groups=K, bias=False)

        if not self.use_skip:
            self.pool = nn.AvgPool2d(3, 2, 1)
            self.conv_stride = nn.Sequential(
                                            ConvBNAct(3, 3, 3),
                                            conv1x1(3, channels*2)
                                        )

    def forward(self, x, img=None):
        if not self.use_skip and img is None:
            raise ValueError('Strided EESP unit needs downsampled input image.\n')

        residual = x
        transform_feats = []

        x = self.conv_init(x)     # Reduce
        for i in range(self.K):
            transform_feats.append(self.layers[i](x))   # Split --> Transform

        for j in range(1, self.K):
            transform_feats[j] += transform_feats[j-1]      # Merge: Sum

        x = torch.cat(transform_feats, dim=1)               # Merge: Concat
        x = self.conv_last(x)

        if self.use_skip:
            x += residual
        else:
            residual = self.pool(residual)
            x = torch.cat([x, residual], dim=1)
            img = self.conv_stride(img)
            x += img

        return x


class ESPNetv2(nn.Module):
    """
    ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 K=4, alpha3=3, alpha4=7, act_type='prelu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.l1_block = ConvBNAct(input_channels, 32, 3, 2, act_type=act_type)
        self.l2_block = EESPModule(32, K=K, stride=2, act_type=act_type)
        self.l3_block1 = EESPModule(64, K=K, stride=2, act_type=act_type)
        self.l3_block2 = build_blocks(lambda channels, act_type: EESPModule(channels, K=K, act_type=act_type), 
                                      128, alpha3, act_type=act_type)
        self.l4_block1 = EESPModule(128, K=K, stride=2, act_type=act_type)
        self.l4_block2 = build_blocks(lambda channels, act_type: EESPModule(channels, K=K, act_type=act_type), 
                                      256, alpha4, act_type=act_type)

        self.convl4_l3 = ConvBNAct(256, 128, 1)
        self.ppm = PyramidPoolingModule(256, 256, act_type=act_type, bias=True)
        self.decoder = SegHead(256, num_classes, act_type=act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(128, 64, 3, 1, act_type=act_type),
                nn.Conv2d(64, num_classes, 1)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(256, 128, 3, 1, act_type=act_type),
                nn.Conv2d(128, num_classes, 1)
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
        
        # 下采样输入图像用于EESP模块
        x_d4 = self.pool(self.pool(x))
        x_d8 = self.pool(x_d4)
        x_d16 = self.pool(x_d8)

        # 编码器阶段1
        x = self.l1_block(x)
        
        # 编码器阶段2
        x = self.l2_block(x, x_d4)
        
        # 编码器阶段3
        x = self.l3_block1(x, x_d8)
        x3 = self.l3_block2(x)
        size_l3 = x3.size()[2:]
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x3)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        # 编码器阶段4
        x = self.l4_block1(x3, x_d16)
        x = self.l4_block2(x)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        # 解码器阶段
        x = F.interpolate(x, size_l3, mode='bilinear', align_corners=True)
        x = self.convl4_l3(x)
        x = torch.cat([x, x3], dim=1)
        
        x = self.ppm(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x


'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 ESPNetv2 特有的参数：

    K: EESP模块中的分支数
    alpha3: L3块中EESP模块的数量
    alpha4: L4块中EESP模块的数量
    act_type: 激活函数类型

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到L3块的输出
    第二个辅助头连接到L4块的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 ESPNetv2 的核心架构和特性，包括：

    EESP模块（扩展的高效空间金字塔卷积）
    金字塔池化模块
    分层编码器-解码器结构
    图像下采样与特征融合策略
    
依赖导入：从您现有的模块中导入所需组件，如 conv1x1, DSConvBNAct, PWConvBNAct, ConvBNAct, PyramidPoolingModule 和 SegHead。

'''