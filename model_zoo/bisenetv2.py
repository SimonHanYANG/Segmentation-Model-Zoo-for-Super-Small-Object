"""
Paper:      BiSeNet V2: Bilateral Network with Guided Aggregation for 
            Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/2004.02147
Create by:  Simon
Date:       2025/06/04
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv3x3, conv1x1, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation, SegHead

__all__ = ['BiSeNetv2']

class DetailBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__(
            ConvBNAct(in_channels, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 64, 3, 2, act_type=act_type),
            ConvBNAct(64, 64, 3, 1, act_type=act_type),
            ConvBNAct(64, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, 128, 3, 2, act_type=act_type),
            ConvBNAct(128, 128, 3, 1, act_type=act_type),
            ConvBNAct(128, out_channels, 3, 1, act_type=act_type)
        )


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.conv_init = ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type)
        self.left_branch = nn.Sequential(
                            ConvBNAct(out_channels, out_channels//2, 1, act_type=act_type),
                            ConvBNAct(out_channels//2, out_channels, 3, 2, act_type=act_type)
                    )
        self.right_branch = nn.MaxPool2d(3, 2, 1)
        self.conv_last = ConvBNAct(out_channels*2, out_channels, 3, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv_init(x)
        x_left = self.left_branch(x)
        x_right = self.right_branch(x)
        x = torch.cat([x_left, x_right], dim=1)
        x = self.conv_last(x)

        return x


class GatherExpansionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type='relu', expand_ratio=6,):
        super().__init__()
        self.stride = stride
        hid_channels = int(round(in_channels * expand_ratio))

        layers = [ConvBNAct(in_channels, in_channels, 3, act_type=act_type)]

        if stride == 2:
            layers.extend([
                            DWConvBNAct(in_channels, hid_channels, 3, 2, act_type='none'),
                            DWConvBNAct(hid_channels, hid_channels, 3, 1, act_type='none')
                        ])
            self.right_branch = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, 2, act_type='none'),
                                    PWConvBNAct(in_channels, out_channels, act_type='none')
                            )            
        else:
            layers.append(DWConvBNAct(in_channels, hid_channels, 3, 1, act_type='none'))

        layers.append(PWConvBNAct(hid_channels, out_channels, act_type='none'))
        self.left_branch = nn.Sequential(*layers)
        self.act = Activation(act_type)

    def forward(self, x):
        res = self.left_branch(x)

        if self.stride == 2:
            res = self.right_branch(x) + res
        else:
            res = x + res

        return self.act(res)


class ContextEmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.pool = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.BatchNorm2d(in_channels)
                    )
        self.conv_mid = ConvBNAct(in_channels, in_channels, 1, act_type=act_type)
        self.conv_last = conv3x3(in_channels, out_channels)

    def forward(self, x):
        res = self.pool(x)
        res = self.conv_mid(res)
        x = res + x
        x = self.conv_last(x)

        return x


class BilateralGuidedAggregationLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.detail_high = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, act_type=act_type),
                                    conv1x1(in_channels, in_channels)
                        )
        self.detail_low = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type),
                                    nn.AvgPool2d(3, 2, 1)
                        )
        self.semantic_high = nn.Sequential(
                                    ConvBNAct(in_channels, in_channels, 3, act_type=act_type),
                                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                    nn.Sigmoid()
                            )
        self.semantic_low = nn.Sequential(
                                    DWConvBNAct(in_channels, in_channels, 3, act_type=act_type),
                                    conv1x1(in_channels, in_channels),
                                    nn.Sigmoid()
                            )
        self.conv_last = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)

    def forward(self, x_d, x_s):
        x_d_high = self.detail_high(x_d)
        x_d_low = self.detail_low(x_d)

        x_s_high = self.semantic_high(x_s)
        x_s_low = self.semantic_low(x_s)
        x_high = x_d_high * x_s_high
        x_low = x_d_low * x_s_low

        size = x_high.size()[2:]
        x_low = F.interpolate(x_low, size, mode='bilinear', align_corners=True)
        res = x_high + x_low
        res = self.conv_last(res)

        return res


class SemanticBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, act_type='relu', use_aux=False):
        super().__init__()
        self.use_aux = use_aux
        self.stage1to2 = StemBlock(in_channels, 16, act_type)
        self.stage3 = nn.Sequential(
                            GatherExpansionLayer(16, 32, 2, act_type),
                            GatherExpansionLayer(32, 32, 1, act_type),
                        )
        self.stage4 = nn.Sequential(
                            GatherExpansionLayer(32, 64, 2, act_type),
                            GatherExpansionLayer(64, 64, 1, act_type),
                        )
        self.stage5_1to4 = nn.Sequential(
                                GatherExpansionLayer(64, 128, 2, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                                GatherExpansionLayer(128, 128, 1, act_type),
                            )
        self.stage5_5 = ContextEmbeddingBlock(128, out_channels, act_type)

        if self.use_aux:
            self.seg_head2 = SegHead(16, num_class, act_type)
            self.seg_head3 = SegHead(32, num_class, act_type)
            self.seg_head4 = SegHead(64, num_class, act_type)
            self.seg_head5 = SegHead(128, num_class, act_type)

    def forward(self, x):
        x = self.stage1to2(x)
        if self.use_aux:
            aux2 = self.seg_head2(x)

        x = self.stage3(x)
        if self.use_aux:
            aux3 = self.seg_head3(x)

        x = self.stage4(x)
        if self.use_aux:
            aux4 = self.seg_head4(x)

        x = self.stage5_1to4(x)
        if self.use_aux:
            aux5 = self.seg_head5(x)

        x = self.stage5_5(x)

        if self.use_aux:
            return x, aux2, aux3, aux4, aux5
        else:
            return x


class BiSeNetv2(nn.Module):
    """
    BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, act_type='relu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 主要分支
        self.detail_branch = DetailBranch(input_channels, 128, act_type)
        self.semantic_branch = SemanticBranch(input_channels, 128, num_classes, act_type, deep_supervision)
        self.bga_layer = BilateralGuidedAggregationLayer(128, 128, act_type)
        self.seg_head = SegHead(128, num_classes, act_type)
        
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
        
        # 详细分支
        x_d = self.detail_branch(x)
        
        # 语义分支
        if self.deep_supervision:
            x_s, aux2, aux3, aux4, aux5 = self.semantic_branch(x)
        else:
            x_s = self.semantic_branch(x)
        
        # 双边引导聚合
        x = self.bga_layer(x_d, x_s)
        
        # 分割头
        x = self.seg_head(x)
        
        # 上采样到原始尺寸
        x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 深度监督
        if self.deep_supervision:
            # 上采样辅助输出到原始尺寸
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
            aux3 = F.interpolate(aux3, input_size, mode='bilinear', align_corners=True)
            aux4 = F.interpolate(aux4, input_size, mode='bilinear', align_corners=True)
            aux5 = F.interpolate(aux5, input_size, mode='bilinear', align_corners=True)
            return [aux2, aux3, aux4, aux5, x]
        
        return x
    

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 BiSeNetv2 特有的参数：act_type。

深度监督：将原始实现中的 use_aux 参数替换为 UNeXt 框架中的 deep_supervision 参数，保持功能一致。

输出格式：当 deep_supervision=True 时，模型返回一个包含所有辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

'''