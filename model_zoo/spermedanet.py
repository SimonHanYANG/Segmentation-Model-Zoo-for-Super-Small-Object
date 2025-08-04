"""
SpermEDANet: 精子亚细胞结构精确分割的增强型密集非对称卷积网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, conv3x3, ConvBNAct, Activation

__all__ = ['SpermEDANet']

# 这里自定义一个类似ConvBNAct的函数，确保参数正确传递
class CustomConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, act_type='relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Activation(act_type)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class EDAModule(nn.Module):
    def __init__(self, in_channels, k, dilation=1, act_type='relu'):
        super().__init__()
        self.conv = nn.Sequential(
                        CustomConvBNAct(in_channels, k, 1, act_type=act_type),
                        nn.Conv2d(k, k, (3, 1), padding=(1, 0), bias=False),
                        CustomConvBNAct(k, k, (1, 3), padding=(0, 1), act_type=act_type),
                        nn.Conv2d(k, k, (3, 1), dilation=dilation, 
                                    padding=(dilation, 0), bias=False),
                        CustomConvBNAct(k, k, (1, 3), padding=(0, dilation), dilation=dilation, act_type=act_type)
                    )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = torch.cat([x, residual], dim=1)
        return x

class EDABlock(nn.Module):
    def __init__(self, in_channels, k, num_block, dilations, act_type):
        super().__init__()
        assert len(dilations) == num_block, 'number of dilation rate should be equal to number of block'

        layers = []
        for i in range(num_block):
            dt = dilations[i]
            layers.append(EDAModule(in_channels, k, dt, act_type))
            in_channels += k
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class MSFM(nn.Module):
    """多尺度融合感知模块 - 增强对超小目标的感知"""
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.branch1 = CustomConvBNAct(in_channels, out_channels // 4, 3, padding=1, act_type=act_type)
        self.branch2 = nn.Sequential(
            CustomConvBNAct(in_channels, out_channels // 4, 3, padding=2, dilation=2, act_type=act_type),
        )
        self.branch3 = nn.Sequential(
            CustomConvBNAct(in_channels, out_channels // 4, 3, padding=4, dilation=4, act_type=act_type),
        )
        self.branch4 = nn.Sequential(
            CustomConvBNAct(in_channels, out_channels // 4, 1, act_type=act_type),
        )
        self.fusion = CustomConvBNAct(out_channels, out_channels, 1, act_type=act_type)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)

class BEAM(nn.Module):
    """边界增强注意力模块 - 增强对精子亚细胞结构边界的辨识"""
    def __init__(self, in_channels, reduction=16, act_type='relu'):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 边界检测分支
        self.edge_branch = nn.Sequential(
            CustomConvBNAct(in_channels, in_channels, 3, padding=1, act_type=act_type),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力分支
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 边界增强卷积
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            Activation(act_type)
        )
        
    def forward(self, x):
        # 边界注意力
        edge_map = self.edge_branch(x)
        
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = avg_out + max_out
        
        # 结合边界和通道注意力
        att = channel_att * edge_map
        
        # 边界增强
        enhanced = self.edge_enhance(x * att)
        
        return x + enhanced

class LSAM(nn.Module):
    """轻量级结构自适应模块 - 适应不同形态的精子结构"""
    def __init__(self, in_channels, act_type='relu'):
        super().__init__()
        self.squeeze = CustomConvBNAct(in_channels, in_channels // 8, 1, act_type=act_type)
        self.process = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 8, 3, padding=1, groups=in_channels // 8),
            nn.BatchNorm2d(in_channels // 8),
            Activation(act_type),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        squeezed = self.squeeze(x)
        processed = self.process(squeezed)
        attention = self.expand(processed)
        return x * attention

class MPFE(nn.Module):
    """多路径特征增强模块 - 增强对精子头部和中段的特征提取"""
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        # 精子头部分支 (细粒度特征)
        self.head_branch = nn.Sequential(
            CustomConvBNAct(in_channels, out_channels // 2, 1, act_type=act_type),
            CustomConvBNAct(out_channels // 2, out_channels // 2, 3, padding=1, act_type=act_type),
        )
        
        # 精子中段分支 (线性结构特征)
        self.midpiece_branch = nn.Sequential(
            CustomConvBNAct(in_channels, out_channels // 2, 1, act_type=act_type),
            nn.Conv2d(out_channels // 2, out_channels // 2, (5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channels // 2),
            Activation(act_type),
            nn.Conv2d(out_channels // 2, out_channels // 2, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(out_channels // 2),
            Activation(act_type),
        )
        
        # 特征融合
        self.fusion = CustomConvBNAct(out_channels, out_channels, 1, act_type=act_type)
        
    def forward(self, x):
        head_feat = self.head_branch(x)
        midpiece_feat = self.midpiece_branch(x)
        merged = torch.cat([head_feat, midpiece_feat], dim=1)
        return self.fusion(merged)

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels - in_channels, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_act = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                            )

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return self.bn_act(x)

class SpermEDANet(nn.Module):
    """
    SpermEDANet: 精子亚细胞结构精确分割的增强型密集非对称卷积网络
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 k=40, num_b1=5, num_b2=8, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 初始特征提取
        self.init_conv = CustomConvBNAct(input_channels, 16, 3, padding=1, act_type=act_type)
        
        # 第一阶段：下采样 + 多路径特征增强
        self.stage1 = nn.Sequential(
            DownsamplingBlock(16, 32, act_type),
            MPFE(32, 32, act_type)
        )
        
        # 第二阶段：下采样 + EDA块 + 边界增强
        self.stage2_d = DownsamplingBlock(32, 64, act_type)
        self.stage2 = EDABlock(64, k, num_b1, [1,1,1,2,2], act_type)
        self.stage2_beam = BEAM(64 + k * num_b1, reduction=8, act_type=act_type)
        
        # 计算第二阶段输出通道数
        stage2_out_channels = 64 + k * num_b1
        
        # 第三阶段：下采样 + 多尺度融合 + EDA块 + 结构自适应
        self.stage3_d = nn.Sequential(
            CustomConvBNAct(stage2_out_channels, 128, 3, stride=2, padding=1, act_type=act_type),
            MSFM(128, 128, act_type)
        )
        self.stage3 = EDABlock(128, k, num_b2, [2,2,4,4,8,8,16,16], act_type)
        self.stage3_lsam = LSAM(128 + k * num_b2, act_type=act_type)
        
        # 计算第三阶段输出通道数
        stage3_out_channels = 128 + k * num_b2
        
        # 主分割头
        self.project = nn.Sequential(
            MSFM(stage3_out_channels, 128, act_type),
            conv1x1(128, num_classes)
        )
        
        # 深度监督分支
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                MSFM(stage2_out_channels, 64, act_type),
                conv1x1(64, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                MSFM(128, 64, act_type),
                conv1x1(64, num_classes)
            )
        
        # 初始化权重
        self._init_weights()
        
        # 保存中间特征用于损失计算
        self.features = {}

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_features(self):
        """返回保存的中间特征，用于计算损失"""
        return self.features

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        input_size = x.size()[2:]
        
        # 初始特征提取
        x = self.init_conv(x)
        self.features['init_feat'] = x
        
        # 第一阶段
        x = self.stage1(x)
        self.features['stage1'] = x
        
        # 第二阶段
        x = self.stage2_d(x)
        x = self.stage2(x)
        x = self.stage2_beam(x)  # 应用边界增强
        stage2_features = x
        self.features['stage2'] = stage2_features
        
        # 辅助分支1（如果使用深度监督）
        if self.deep_supervision:
            aux1 = self.aux_head1(stage2_features)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        # 第三阶段第一部分
        x = self.stage3_d(stage2_features)
        self.features['stage3_pre'] = x
        
        # 辅助分支2（如果使用深度监督）
        if self.deep_supervision:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        # 第三阶段第二部分
        x = self.stage3(x)
        x = self.stage3_lsam(x)  # 应用结构自适应模块
        self.features['stage3'] = x
        
        # 主分割头
        x = self.project(x)
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x