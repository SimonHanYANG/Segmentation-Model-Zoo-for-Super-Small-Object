"""
Paper:      Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis
Url:        https://ieeexplore.ieee.org/abstract/document/8793923
Create by:  Simon
Date:       2025/06/04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import conv1x1, DSConvBNAct, ConvBNAct, DeConvBNAct, Activation

__all__ = ['MiniNet']

class ConvModule(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(channels, channels, (1,3), padding=(0, dilation), 
                                    dilation=dilation, groups=channels, bias=False),
                        Activation(act_type),
                        nn.Conv2d(channels, channels, (3,1), padding=(dilation, 0), 
                                    dilation=dilation, groups=channels, bias=False),
                        Activation(act_type),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels, channels, (3,1), padding=(dilation, 0), 
                                    dilation=dilation, groups=channels, bias=False),
                        Activation(act_type),
                        nn.Conv2d(channels, channels, (1,3), padding=(0, dilation), 
                                    dilation=dilation, groups=channels, bias=False),
                    )
        self.dropout = nn.Dropout(p=0.25)
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x1 = self.conv1(x)
        x = self.conv2(x1)

        x += x1
        x = self.dropout(x)
        x += residual

        return self.act(x)


class MiniNet(nn.Module):
    """
    MiniNet: Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, act_type='selu', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # Downsample block
        self.down1 = DSConvBNAct(input_channels, 12, 3, 2, act_type=act_type)
        self.down2 = DSConvBNAct(12, 24, 3, 2, act_type=act_type)
        self.down3 = DSConvBNAct(24, 48, 3, 2, act_type=act_type)
        self.down4 = DSConvBNAct(48, 96, 3, 2, act_type=act_type)
        
        # Branch 1
        self.branch1 = nn.Sequential(
                            ConvModule(96, 1, act_type),
                            ConvModule(96, 2, act_type),
                            ConvModule(96, 4, act_type),
                            ConvModule(96, 8, act_type),
                        )
        
        # Branch 2
        self.branch2_down = DSConvBNAct(96, 192, 3, 2, act_type=act_type)
        self.branch2 = nn.Sequential(
                            ConvModule(192, 1, act_type),
                            DSConvBNAct(192, 386, 3, 2, act_type=act_type),
                            ConvModule(386, 1, act_type),
                            ConvModule(386, 1, act_type),
                            DeConvBNAct(386, 192, act_type=act_type),
                            ConvModule(192, 1, act_type),
                        )
        self.branch2_up = DeConvBNAct(192*2, 96, act_type=act_type)
        
        # Upsample Block
        self.up4 = nn.Sequential(
                        DeConvBNAct(96*3, 96, act_type=act_type),
                        ConvModule(96, 1, act_type),
                        conv1x1(96, 48)
                    )
        self.up3 = DeConvBNAct(48*2, 24, act_type=act_type)
        self.up2 = DeConvBNAct(24*2, 12, act_type=act_type)
        self.up1 = DeConvBNAct(12*2, num_classes, act_type=act_type)
        
        # 深度监督
        if deep_supervision:
            # 辅助分支1 - 从上采样阶段3
            self.aux_head1 = nn.Sequential(
                ConvBNAct(48, 24, 3, act_type=act_type),
                DeConvBNAct(24, 24, act_type=act_type),
                DeConvBNAct(24, 12, act_type=act_type),
                DeConvBNAct(12, num_classes, act_type=act_type)
            )
            
            # 辅助分支2 - 从上采样阶段2
            self.aux_head2 = nn.Sequential(
                ConvBNAct(24, 12, 3, act_type=act_type),
                DeConvBNAct(12, 12, act_type=act_type),
                DeConvBNAct(12, num_classes, act_type=act_type)
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
        
        # 下采样阶段
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)

        # 分支1
        x_b1 = self.branch1(x_d4)

        # 分支2
        x_d5 = self.branch2_down(x_d4)
        x_b2 = self.branch2(x_d5)
        x_b2 = torch.cat([x_b2, x_d5], dim=1)
        x_b2 = self.branch2_up(x_b2)

        # 特征融合
        x = torch.cat([x_b1, x_b2, x_d4], dim=1)
        
        # 上采样阶段
        x = self.up4(x)
        
        # 第一个辅助输出
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, input_size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x, x_d3], dim=1)
        x = self.up3(x)
        
        # 第二个辅助输出
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, input_size, mode='bilinear', align_corners=True)
        
        x = torch.cat([x, x_d2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x_d1], dim=1)
        x = self.up1(x)
        
        # 确保输出尺寸与输入相同
        if x.size()[2:] != input_size:
            x = F.interpolate(x, input_size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, x]
        
        return x
    

'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 MiniNet 特有的参数：

    act_type: 激活函数类型，默认为 'selu'（原始论文中使用的激活函数）

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加两个辅助分割头：

    第一个辅助头连接到上采样阶段3（up4输出）
    第二个辅助头连接到上采样阶段2（up3输出）

输出格式：当 deep_supervision=True 时，模型返回一个包含两个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 MiniNet 的核心架构和特性，包括：

    双分支结构
    特殊的卷积模块（分解的卷积操作）
    高效的编码器-解码器设计
    多尺度特征融合

依赖导入：从您现有的模块中导入所需组件，如 conv1x1, DSConvBNAct, ConvBNAct, DeConvBNAct, 和 Activation。
'''