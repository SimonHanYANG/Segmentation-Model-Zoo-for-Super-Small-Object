import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import ConvBNAct

__all__ = ['UNet']

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.double_conv = nn.Sequential(
            ConvBNAct(in_channels, out_channels, 3, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, act_type=act_type)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, act_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, act_type='relu', bilinear=True):
        super().__init__()

        # 如果使用双线性插值上采样，则不需要转置卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, act_type)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, act_type)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 输入可能不是2的整数倍，需要进行裁剪
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # 连接特征图
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet: Convolutional Networks for Biomedical Image Segmentation
    
    Adapted to match the UNeXt training framework interface
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 filters=(64, 128, 256, 512, 1024), bilinear=True, act_type='relu', dropout=0.5, **kwargs):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        self.filters = filters
        
        # 编码器部分
        self.inc = DoubleConv(input_channels, filters[0], act_type)
        self.down1 = Down(filters[0], filters[1], act_type)
        self.down2 = Down(filters[1], filters[2], act_type)
        self.down3 = Down(filters[2], filters[3], act_type)
        self.down4 = Down(filters[3], filters[4] // 2 if bilinear else filters[4], act_type)
        
        # 添加dropout
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout2d(dropout)
        
        # 解码器部分
        self.up1 = Up(filters[4], filters[3] // 2 if bilinear else filters[3], act_type, bilinear)
        self.up2 = Up(filters[3], filters[2] // 2 if bilinear else filters[2], act_type, bilinear)
        self.up3 = Up(filters[2], filters[1] // 2 if bilinear else filters[1], act_type, bilinear)
        self.up4 = Up(filters[1], filters[0], act_type, bilinear)
        
        # 输出层
        self.outc = OutConv(filters[0], num_classes)
        
        # 深度监督
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(filters[3], filters[3]//2, 3, act_type=act_type),
                OutConv(filters[3]//2, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(filters[2], filters[2]//2, 3, act_type=act_type),
                OutConv(filters[2]//2, num_classes)
            )
            self.aux_head3 = nn.Sequential(
                ConvBNAct(filters[1], filters[1]//2, 3, act_type=act_type),
                OutConv(filters[1]//2, num_classes)
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

    def forward(self, x):
        # 保存输入尺寸用于最终调整
        size = x.size()[2:]
        
        # 编码器路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.dropout1(x4)
        x5 = self.down4(x4)
        x5 = self.dropout2(x5)
        
        # 解码器路径
        x = self.up1(x5, x4)
        
        # 第一个辅助输出（如果启用深度监督）
        if self.deep_supervision and self.training:
            aux1 = self.aux_head1(x)
            aux1 = F.interpolate(aux1, size, mode='bilinear', align_corners=True)
        
        x = self.up2(x, x3)
        
        # 第二个辅助输出（如果启用深度监督）
        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(x)
            aux2 = F.interpolate(aux2, size, mode='bilinear', align_corners=True)
        
        x = self.up3(x, x2)
        
        # 第三个辅助输出（如果启用深度监督）
        if self.deep_supervision and self.training:
            aux3 = self.aux_head3(x)
            aux3 = F.interpolate(aux3, size, mode='bilinear', align_corners=True)
        
        x = self.up4(x, x1)
        x = self.outc(x)
        
        # 调整到原始大小
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        # 返回结果
        if self.deep_supervision and self.training:
            return [aux1, aux2, aux3, x]
        
        return x
    
'''
接口兼容性：模型接受与 UNeXt 相同的主要参数：num_classes, input_channels, deep_supervision，并添加了 UNet 特有的参数：

    filters: 各层的通道数，默认为 (64, 128, 256, 512, 1024)
    bilinear: 是否使用双线性插值进行上采样，默认为 True
    act_type: 激活函数类型，默认为 'relu'
    dropout: Dropout 比率，默认为 0.5

深度监督：添加了深度监督支持，当 deep_supervision=True 时，在网络的中间阶段添加三个辅助分割头：

    第一个辅助头连接到 up1 的输出
    第二个辅助头连接到 up2 的输出
    第三个辅助头连接到 up3 的输出

输出格式：当 deep_supervision=True 时，模型返回一个包含三个辅助输出和主输出的列表，格式与 UNeXt 框架一致。

输出大小调整：确保所有输出（主输出和辅助输出）都调整到与输入图像相同的尺寸。

权重初始化：添加了适当的权重初始化方法，提高模型收敛性。

结构保留：保留了原始 UNet 的核心架构和特性，包括：

    编码器-解码器结构
    跳跃连接
    双卷积块
    Dropout 层
    
模块化设计：将 UNet 的各个组件（如 DoubleConv, Down, Up, OutConv）设计为独立的模块，提高代码的可读性和可维护性。

依赖导入：从您现有的模块中导入所需组件，如 ConvBNAct。
'''