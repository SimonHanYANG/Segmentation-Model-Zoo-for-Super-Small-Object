"""
Paper:      U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection
Url:        https://arxiv.org/abs/2005.09007
Create by:  Simon
Date:       2025/07/25
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['U2Net', 'U2NetP']

class REBNCONV(nn.Module):
    """优化版本的残差边缘块，带批归一化和卷积"""
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


def _upsample_like(src, target):
    """将tensor 'src'上采样到与tensor 'target'相同的空间大小"""
    return F.interpolate(src, size=target.shape[2:], mode='bilinear', align_corners=True)


class RSU7(nn.Module):
    """RSU-7块: 7层残差U形块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU6(nn.Module):
    """RSU-6块: 6层残差U形块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU5(nn.Module):
    """RSU-5块: 5层残差U形块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4(nn.Module):
    """RSU-4块: 4层残差U形块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class RSU4F(nn.Module):
    """RSU-4F块: 带扩张卷积的4层残差U形块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch)

    def forward(self, x):
        hxin = self.rebnconvin(x)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class U2Net(nn.Module):
    """
    U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection
    
    针对高分辨率图像和内存使用进行优化的实现
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super(U2Net, self).__init__()
        self.deep_supervision = deep_supervision
        out_ch = num_classes
        
        # 优化网络参数以减少内存占用
        # 对于高分辨率图像(1920x1440)，我们需要降低特征通道数
        
        # 编码器 - 降低中间通道数以减少内存使用
        self.stage1 = RSU7(input_channels, 16, 32)  # 原始是32,64
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(32, 16, 64)  # 原始是64,32,128
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 32, 128)  # 原始是128,64,256
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(128, 64, 256)  # 原始是256,128,512
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(256, 128, 256)  # 原始是512,256,512
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(256, 128, 256)  # 原始是512,256,512

        # 解码器
        self.stage5d = RSU4F(512, 128, 256)  # 原始是1024,256,512
        self.stage4d = RSU4(512, 64, 128)    # 原始是1024,128,256
        self.stage3d = RSU5(256, 32, 64)     # 原始是512,64,128
        self.stage2d = RSU6(128, 16, 32)     # 原始是256,32,64
        self.stage1d = RSU7(64, 8, 32)       # 原始是128,16,64

        # 侧输出 (深度监督)
        self.side1 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(256, out_ch, 3, padding=1)

        # 融合输出
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
        
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
        # 记录原始输入尺寸用于上采样
        input_size = x.size()[2:]
        
        # 阶段 1
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        # 阶段 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # 阶段 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # 阶段 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # 阶段 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # 阶段 6 (底部)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # 解码器 - 阶段 5d
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        # 解码器 - 阶段 4d
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 解码器 - 阶段 3d
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 解码器 - 阶段 2d
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 解码器 - 阶段 1d
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # 侧输出
        d1 = self.side1(hx1d)
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)

        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)

        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)

        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=input_size, mode='bilinear', align_corners=True)

        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=input_size, mode='bilinear', align_corners=True)

        # 融合侧输出
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        # 根据deep_supervision标志返回结果
        if self.deep_supervision:
            # 使用sigmoid确保输出在0-1范围内
            return [torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), 
                    torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)]
        else:
            return torch.sigmoid(d0)


class U2NetP(nn.Module):
    """
    U^2-Net-P (U^2-Net的小型版本)
    
    针对高分辨率图像和内存使用进行优化的实现
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super(U2NetP, self).__init__()
        self.deep_supervision = deep_supervision
        out_ch = num_classes
        
        # 编码器
        self.stage1 = RSU7(input_channels, 8, 16)  # 原始是16,64
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(16, 8, 32)  # 原始是64,16,64
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(32, 8, 32)  # 原始是64,16,64
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(32, 8, 32)  # 原始是64,16,64
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(32, 8, 32)  # 原始是64,16,64
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(32, 8, 32)  # 原始是64,16,64

        # 解码器
        self.stage5d = RSU4F(64, 8, 32)  # 原始是128,16,64
        self.stage4d = RSU4(64, 8, 32)   # 原始是128,16,64
        self.stage3d = RSU5(64, 8, 32)   # 原始是128,16,64
        self.stage2d = RSU6(64, 8, 16)   # 原始是128,16,64
        self.stage1d = RSU7(32, 8, 16)   # 原始是128,16,64

        # 侧输出 (深度监督)
        self.side1 = nn.Conv2d(16, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(16, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(32, out_ch, 3, padding=1)

        # 融合输出
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
        
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
        # 记录原始输入尺寸用于上采样
        input_size = x.size()[2:]
        
        # 阶段 1
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)

        # 阶段 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # 阶段 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # 阶段 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # 阶段 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # 阶段 6 (底部)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # 解码器 - 阶段 5d
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        # 解码器 - 阶段 4d
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        # 解码器 - 阶段 3d
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        # 解码器 - 阶段 2d
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        # 解码器 - 阶段 1d
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # 侧输出
        d1 = self.side1(hx1d)
        d1 = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=True)

        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=True)

        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=input_size, mode='bilinear', align_corners=True)

        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=input_size, mode='bilinear', align_corners=True)

        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=input_size, mode='bilinear', align_corners=True)

        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=input_size, mode='bilinear', align_corners=True)

        # 融合侧输出
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        # 根据deep_supervision标志返回结果
        if self.deep_supervision:
            # 使用sigmoid确保输出在0-1范围内
            return [torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), 
                    torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)]
        else:
            return torch.sigmoid(d0)