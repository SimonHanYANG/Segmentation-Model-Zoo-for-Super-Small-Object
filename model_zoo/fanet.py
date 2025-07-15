"""
Paper:      Real-time Semantic Segmentation with Fast Attention
Url:        https://arxiv.org/abs/2007.03815
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from modules import ConvBNAct, DeConvBNAct, SegHead, Activation
from backbone import ResNet

__all__ = ['FANet']

class FastAttention(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.conv_q = ConvBNAct(in_channels, out_channels, 3, act_type='none')
        self.conv_k = ConvBNAct(in_channels, out_channels, 3, act_type='none')
        self.conv_v = ConvBNAct(in_channels, out_channels, 3, act_type='none')
        self.conv_fuse = ConvBNAct(out_channels, out_channels, 3, act_type=act_type)

    def forward(self, x):
        x_q = self.conv_q(x)
        x_k = self.conv_k(x)
        x_v = self.conv_v(x)
        residual = x_v

        B, C, H, W = x_q.size()
        n = H * W

        x_q = x_q.view(B, C, n)
        x_k = x_k.view(B, C, n)
        x_v = x_v.view(B, C, n)

        x_q = F.normalize(x_q, p=2, dim=1)
        x_k = F.normalize(x_k, p=2, dim=1).permute(0,2,1)

        y = (x_q @ (x_k @ x_v)) / n
        y = y.view(B, C, H, W)
        y = self.conv_fuse(y)
        y += residual

        return y


class FuseUp(nn.Module):
    def __init__(self, in_channels, out_channels, has_up=True, act_type='relu'):
        super().__init__()
        self.has_up = has_up
        if has_up:
            self.up = DeConvBNAct(in_channels, in_channels, act_type=act_type, inplace=True)

        self.conv = ConvBNAct(in_channels, out_channels, 3, act_type=act_type, inplace=True)

    def forward(self, x_fa, x_up=None):
        if self.has_up:
            if x_up is None:
                raise RuntimeError('Missing input from Up layer.\n')
            else:
                x_up = self.up(x_up)
            x_fa += x_up

        x_fa = self.conv(x_fa)

        return x_fa


class FANet(nn.Module):
    """
    FANet: Real-time Semantic Segmentation with Fast Attention
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 att_channel=32, backbone_type='resnet18', cat_feat=True,
                 act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        if backbone_type in ['resnet18', 'resnet34']:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512]
            self.num_stage = len(channels)

            # Reduce spatial dimension for Res-1
            downsample = ConvBNAct(channels[0], channels[0], 1, 2, act_type='none')
            self.backbone.layer1[0] = BasicBlock(channels[0], channels[0], 2, downsample)
        else:
            raise NotImplementedError("Only resnet18 and resnet34 backbones are supported")
            
        self.cat_feat = cat_feat

        # Fast Attention modules
        self.fast_attention = nn.ModuleList([
            FastAttention(channels[i], att_channel, act_type) 
            for i in range(self.num_stage)
        ])

        # FuseUp modules
        layers = [
            FuseUp(att_channel, att_channel, act_type=act_type) 
            for _ in range(self.num_stage-1)
        ]
        layers.append(FuseUp(att_channel, att_channel, has_up=False, act_type=act_type))
        self.fuse_up = nn.ModuleList(layers)

        # Segmentation head
        last_channel = 4*att_channel if cat_feat else att_channel
        self.seg_head = SegHead(last_channel, num_classes, act_type)
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(att_channel, att_channel//2, 3, 1, act_type=act_type),
                nn.Conv2d(att_channel//2, num_classes, 1)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(att_channel, att_channel//2, 3, 1, act_type=act_type),
                nn.Conv2d(att_channel//2, num_classes, 1)
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
        
        # 提取骨干网络特征
        x1, x2, x3, x4 = self.backbone(x)
        
        # 应用快速注意力和特征融合
        x4_att = self.fast_attention[3](x4)
        x4_fuse = self.fuse_up[3](x4_att)
        
        x3_att = self.fast_attention[2](x3)
        x3_fuse = self.fuse_up[2](x3_att, x4_fuse)
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x3_fuse)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        x2_att = self.fast_attention[1](x2)
        x2_fuse = self.fuse_up[1](x2_att, x3_fuse)
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(x2_fuse)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        x1_att = self.fast_attention[0](x1)
        x1_fuse = self.fuse_up[0](x1_att, x2_fuse)
        
        # 特征融合和分割头
        if self.cat_feat:
            size1 = x1_fuse.size()[2:]
            x4_resize = F.interpolate(x4_fuse, size1, mode='bilinear', align_corners=True)
            x3_resize = F.interpolate(x3_fuse, size1, mode='bilinear', align_corners=True)
            x2_resize = F.interpolate(x2_fuse, size1, mode='bilinear', align_corners=True)
            
            x = torch.cat([x1_fuse, x2_resize, x3_resize, x4_resize], dim=1)
            x = self.seg_head(x)
        else:
            x = self.seg_head(x1_fuse)
        
        # 调整到输入尺寸
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x
    
'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 FANet 特有的参数：

    att_channel: 注意力模块的通道数
    backbone_type: 骨干网络类型（'resnet18' 或 'resnet34'）
    cat_feat: 是否在最终分割头前拼接所有特征
    act_type: 激活函数类型

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到第二阶段特征融合的输出
    第二个辅助头连接到第三阶段特征融合的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 FANet 的核心架构和特性，包括：

    快速注意力机制
    特征融合和上采样模块
    多尺度特征聚合

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct, DeConvBNAct, SegHead, Activation 和 ResNet。
'''