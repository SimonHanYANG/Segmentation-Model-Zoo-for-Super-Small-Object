"""
FEDASNet: Fidelity-Enhanced Direction-Aware Segmentation Network
修复版本 - 解决通道不匹配和API更新问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['FEDASNet']


class ConvBNAct(nn.Module):
    """卷积+BN+激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=None, dilation=1, groups=1, act_type='relu'):
        super().__init__()
        if padding is None:
            if isinstance(kernel_size, (list, tuple)):
                padding = tuple(k//2 for k in kernel_size)
            else:
                padding = kernel_size // 2 if dilation == 1 else dilation
                
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.act = nn.PReLU()
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Activation(nn.Module):
    """激活函数封装"""
    def __init__(self, act_type='relu'):
        super().__init__()
        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            self.act = nn.PReLU()
        elif act_type == 'leaky':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        return self.act(x)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1卷积"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class InitialBlock(nn.Module):
    """初始块"""
    def __init__(self, in_channels, out_channels, act_type='prelu', kernel_size=3):
        super().__init__()
        assert out_channels > in_channels
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, kernel_size, 2, act_type=act_type)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return x


class DirectionAwareConv(nn.Module):
    """方向感知卷积模块 - 优化版本"""
    def __init__(self, in_channels, out_channels, num_directions=4):
        super().__init__()
        self.num_directions = num_directions
        self.direction_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // num_directions, 
                     kernel_size=(5, 1), padding=(2, 0), bias=False)
            for _ in range(num_directions)
        ])
        self.fusion = conv1x1(out_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        outputs = []
        
        angles = [0, 45, 90, 135][:self.num_directions]
        
        for i, (conv, angle) in enumerate(zip(self.direction_convs, angles)):
            if angle == 0:
                feat = conv(x)
            elif angle == 90:
                x_rot = x.transpose(-1, -2)
                feat = conv(x_rot).transpose(-1, -2)
            else:
                feat = conv(x)
            
            outputs.append(feat)
        
        x = torch.cat(outputs, dim=1)
        x = self.fusion(x)
        x = self.bn(x)
        return x


class LocalContrastEnhancement(nn.Module):
    """局部对比度增强模块 - 优化版本"""
    def __init__(self, channels):
        super().__init__()
        mid_channels = max(channels // 8, 16)
        self.param_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, mid_channels),
            nn.ReLU(inplace=True),
            conv1x1(mid_channels, 2)
        )
        
    def forward(self, x):
        params = self.param_predictor(x)
        
        alpha = torch.sigmoid(params[:, 0:1, :, :]) * 2
        beta = torch.tanh(params[:, 1:2, :, :])
        
        enhanced = alpha * (x - beta) + beta
        return enhanced


class Bottleneck(nn.Module):
    """基础瓶颈块"""
    def __init__(self, in_channels, out_channels, conv_type, act_type='prelu', 
                 upsample_type='regular', dilation=1, drop_p=0.1, shrink_ratio=0.25):
        super().__init__()
        self.conv_type = conv_type
        hid_channels = int(in_channels * shrink_ratio)
        
        if conv_type == 'regular':
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1, act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, act_type=act_type),
            )
        elif conv_type == 'downsampling':
            self.left_pool = nn.MaxPool2d(2, 2, return_indices=True)
            self.left_conv = ConvBNAct(in_channels, out_channels, 1, act_type='')
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 3, 2, act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, act_type=act_type),
            )
        elif conv_type == 'upsampling':
            self.left_conv = ConvBNAct(in_channels, out_channels, 1, act_type='')
            self.left_pool = nn.MaxUnpool2d(2, 2)
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1, act_type=act_type),
                Upsample(hid_channels, hid_channels, scale_factor=2,  
                        kernel_size=3, upsample_type=upsample_type),
            )
        elif conv_type == 'dilate':
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1, act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, dilation=dilation, act_type=act_type),
            )
        elif conv_type == 'asymmetric':
            self.right_init_conv = nn.Sequential(
                ConvBNAct(in_channels, hid_channels, 1, act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, (5,1), act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, (1,5), act_type=act_type),
            )
        
        self.right_last_conv = nn.Sequential(
            conv1x1(hid_channels, out_channels),
            nn.Dropout(drop_p)
        )
        self.act = Activation(act_type)

    def forward(self, x, indices=None):
        x_right = self.right_last_conv(self.right_init_conv(x))
        
        if self.conv_type == 'downsampling':
            x_left, indices = self.left_pool(x)
            x_left = self.left_conv(x_left)
            x = self.act(x_left + x_right)
            return x, indices
        elif self.conv_type == 'upsampling':
            if indices is None:
                raise ValueError('Upsampling-type conv needs pooling indices.')
            x_left = self.left_conv(x)
            x_left = self.left_pool(x_left, indices)
            x = self.act(x_left + x_right)
        else:
            x = self.act(x + x_right)
            
        return x


class FidelityBottleneck(nn.Module):
    """保真度增强瓶颈块"""
    def __init__(self, in_channels, out_channels, conv_type, act_type='prelu',
                 use_dac=False, use_lce=False, **kwargs):
        super().__init__()
        self.conv_type = conv_type
        self.base_bottleneck = Bottleneck(in_channels, out_channels, conv_type, 
                                         act_type, **kwargs)
        
        self.use_dac = use_dac
        self.use_lce = use_lce
        
        if use_dac:
            self.dac = DirectionAwareConv(out_channels, out_channels)
        
        if use_lce:
            self.lce = LocalContrastEnhancement(out_channels)
            
    def forward(self, x, indices=None):
        if self.conv_type == 'downsampling':
            x, indices = self.base_bottleneck(x, indices)
        else:
            x = self.base_bottleneck(x, indices)
        
        if self.use_dac:
            x = x + self.dac(x)
        
        if self.use_lce:
            x = self.lce(x)
        
        if self.conv_type == 'downsampling':
            return x, indices
        return x


class Upsample(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, 
                 upsample_type=None, act_type='relu'):
        super().__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            padding = (kernel_size - 1) // 2
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, 
                                             stride=scale_factor, padding=padding, 
                                             output_padding=1, bias=False)
        else:
            self.up_conv = nn.Sequential(
                ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            )

    def forward(self, x):
        return self.up_conv(x)


class EfficientSelfAttention(nn.Module):
    """高效的自注意力模块 - 使用下采样和局部注意力"""
    def __init__(self, in_channels, reduction=8, spatial_reduction=4):
        super().__init__()
        self.spatial_reduction = spatial_reduction
        mid_channels = max(in_channels // reduction, 16)
        
        self.downsample = nn.AvgPool2d(spatial_reduction) if spatial_reduction > 1 else nn.Identity()
        
        self.query_conv = conv1x1(in_channels, mid_channels)
        self.key_conv = conv1x1(in_channels, mid_channels)
        self.value_conv = conv1x1(in_channels, in_channels)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        x_down = self.downsample(x)
        _, _, H_d, W_d = x_down.size()
        
        proj_query = self.query_conv(x_down).view(B, -1, H_d * W_d).permute(0, 2, 1)
        proj_key = self.key_conv(x_down).view(B, -1, H_d * W_d)
        proj_value = self.value_conv(x_down).view(B, -1, H_d * W_d)
        
        # 使用float32计算注意力以保证数值稳定性
        with torch.amp.autocast(device_type='cuda', enabled=False):
            proj_query = proj_query.float()
            proj_key = proj_key.float()
            
            scale = 1.0 / math.sqrt(proj_key.size(1))
            attention = torch.bmm(proj_query, proj_key) * scale
            attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H_d, W_d)
        
        if self.spatial_reduction > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        
        out = self.gamma * out + x
        return out


class RegionAwareModule(nn.Module):
    """区域感知模块 - 优化版本"""
    def __init__(self, feat_channels):
        super().__init__()
        c1, c2, c3 = feat_channels
        
        # 使用统一的中间通道数
        self.mid_channels = 32
        
        self.proj1 = ConvBNAct(c1, self.mid_channels, 1, act_type='relu')
        self.proj2 = ConvBNAct(c2, self.mid_channels, 1, act_type='relu')
        self.proj3 = ConvBNAct(c3, self.mid_channels, 1, act_type='relu')
        
        self.attention = EfficientSelfAttention(self.mid_channels, reduction=4, spatial_reduction=4)
        
        self.region_head = nn.Sequential(
            ConvBNAct(self.mid_channels, 16, 3, 1, act_type='relu'),
            conv1x1(16, 1),
            nn.Sigmoid()
        )
        
        # 添加通道适配器，用于匹配解码器的通道数
        self.channel_adapters = nn.ModuleDict({
            '64': conv1x1(self.mid_channels, 64),
            '16': conv1x1(self.mid_channels, 16)
        })
        
    def forward(self, feat1, feat2, feat3):
        target_size = feat2.shape[2:]
        
        f1 = F.interpolate(self.proj1(feat1), size=target_size, 
                          mode='bilinear', align_corners=True)
        f2 = self.proj2(feat2)
        f3 = self.proj3(feat3)
        
        fused = f1 + f2 + f3
        attn_feat = self.attention(fused)
        
        region_map = self.region_head(attn_feat)
        
        return region_map, attn_feat
    
    def get_adapted_features(self, attn_feat, target_channels):
        """获取适配到目标通道数的特征"""
        if str(target_channels) in self.channel_adapters:
            return self.channel_adapters[str(target_channels)](attn_feat)
        else:
            # 如果没有预定义的适配器，动态创建一个
            adapter = conv1x1(self.mid_channels, target_channels).to(attn_feat.device)
            return adapter(attn_feat)


class BlurrinessEstimator(nn.Module):
    """模糊度估计器 - 优化版本"""
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(in_channels // 4, 16)
        self.estimator = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, 3, 1, act_type='relu'),
            nn.AdaptiveAvgPool2d(1),
            conv1x1(mid_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        blur_map = self.estimator(x)
        return blur_map.expand(-1, -1, x.size(2), x.size(3))


class BoundaryBottleneck(nn.Module):
    """边界增强瓶颈块"""
    def __init__(self, in_channels, out_channels, conv_type, act_type='prelu',
                 use_boundary=False, **kwargs):
        super().__init__()
        self.conv_type = conv_type
        self.base_bottleneck = Bottleneck(in_channels, out_channels, conv_type,
                                         act_type, **kwargs)
        
        self.use_boundary = use_boundary
        self.out_channels = out_channels
        
        if use_boundary:
            self.blur_estimator = BlurrinessEstimator(out_channels)
            self.clear_branch = ConvBNAct(out_channels, out_channels, 3, 1, act_type=act_type)
            self.blur_branch = nn.Sequential(
                ConvBNAct(out_channels, out_channels, 3, 1, act_type=act_type),
                ConvBNAct(out_channels, out_channels, 3, 1, act_type=act_type)
            )
            self.fusion = ConvBNAct(out_channels * 2, out_channels, 1, act_type=act_type)
            
    def forward(self, x, indices=None, attn_feat=None, attn_adapter=None):
        if self.conv_type == 'upsampling' and indices is not None:
            x = self.base_bottleneck(x, indices)
        else:
            x = self.base_bottleneck(x)
            
        if self.use_boundary and attn_feat is not None and attn_adapter is not None:
            # 使用适配器调整注意力特征的通道数
            attn_feat_adapted = attn_adapter(attn_feat, self.out_channels)
            
            if x.shape[2:] != attn_feat_adapted.shape[2:]:
                attn_feat_adapted = F.interpolate(attn_feat_adapted, size=x.shape[2:],
                                                mode='bilinear', align_corners=True)
            
            x = x + attn_feat_adapted * 0.1
            
            blur_map = self.blur_estimator(x)
            clear_feat = self.clear_branch(x) * (1 - blur_map)
            blur_feat = self.blur_branch(x) * blur_map
            x = self.fusion(torch.cat([clear_feat, blur_feat], dim=1))
            
        return x


class FEDASNet(nn.Module):
    """FEDAS-Net主网络 - 修复版本"""
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False,
                 act_type='prelu', upsample_type='deconvolution', **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        self._features = {}
        
        # 初始块
        self.initial = InitialBlock(input_channels, 16, act_type)
        
        # 编码器
        # Stage 1
        self.encoder1_pool = FidelityBottleneck(16, 64, 'downsampling', act_type,
                                               use_dac=True, use_lce=True, drop_p=0.01)
        self.encoder1_blocks = nn.Sequential(
            FidelityBottleneck(64, 64, 'regular', act_type, use_lce=True, drop_p=0.01),
            FidelityBottleneck(64, 64, 'regular', act_type, drop_p=0.01),
            FidelityBottleneck(64, 64, 'regular', act_type, drop_p=0.01),
            FidelityBottleneck(64, 64, 'regular', act_type, use_dac=True, drop_p=0.01),
        )
        
        # Stage 2
        self.encoder2_pool = FidelityBottleneck(64, 128, 'downsampling', act_type)
        self.encoder2_blocks = nn.Sequential(
            FidelityBottleneck(128, 128, 'regular', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, dilation=2),
            FidelityBottleneck(128, 128, 'asymmetric', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, dilation=4),
            FidelityBottleneck(128, 128, 'regular', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, dilation=8),
            FidelityBottleneck(128, 128, 'asymmetric', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, use_lce=True, dilation=16),
        )
        
        # Stage 3
        self.encoder3_blocks = nn.Sequential(
            FidelityBottleneck(128, 128, 'regular', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, dilation=2),
            FidelityBottleneck(128, 128, 'asymmetric', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, dilation=4),
            FidelityBottleneck(128, 128, 'regular', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, dilation=8),
            FidelityBottleneck(128, 128, 'asymmetric', act_type),
            FidelityBottleneck(128, 128, 'dilate', act_type, use_lce=True, dilation=16),
        )
        
        # 区域感知模块
        self.region_aware = RegionAwareModule([64, 128, 128])
        
        # 解码器
        self.decoder1 = BoundaryBottleneck(128, 64, 'upsampling', act_type,
                                          upsample_type=upsample_type, use_boundary=True)
        self.decoder1_extra = BoundaryBottleneck(64, 64, 'regular', act_type, use_boundary=True)
        
        self.decoder2 = BoundaryBottleneck(64, 16, 'upsampling', act_type,
                                          upsample_type=upsample_type, use_boundary=True)
        
        # 最终输出
        self.final_up = Upsample(16, 16, scale_factor=2, act_type=act_type)
        self.final_conv = nn.Conv2d(16 + 1, num_classes, 1)
        
        # 深度监督
        if deep_supervision:
            self.aux_head1 = nn.Sequential(
                ConvBNAct(64, 32, 3, 1, act_type=act_type),
                conv1x1(32, num_classes)
            )
            self.aux_head2 = nn.Sequential(
                ConvBNAct(128, 64, 3, 1, act_type=act_type),
                conv1x1(64, num_classes)
            )
        
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
    
    def get_features(self):
        return self._features.copy()
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        self._features = {}
        
        # 初始处理
        x = self.initial(x)
        
        # 编码器
        x1, indices1 = self.encoder1_pool(x)
        x1 = self.encoder1_blocks(x1)
        self._features['encoder1'] = x1
        
        if self.deep_supervision:
            aux1 = self.aux_head1(x1)
            aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=True)
        
        x2, indices2 = self.encoder2_pool(x1)
        x2 = self.encoder2_blocks(x2)
        self._features['encoder2'] = x2
        
        if self.deep_supervision:
            aux2 = self.aux_head2(x2)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=True)
        
        x3 = self.encoder3_blocks(x2)
        self._features['encoder3'] = x3
        
        self._features['encoder_features'] = [x1, x2, x3]
        
        # 区域感知
        region_map, attn_feat = self.region_aware(x1, x2, x3)
        self._features['region_map'] = region_map
        self._features['attention_features'] = attn_feat
        
        # 解码器 - 使用适配的注意力特征
        x = self.decoder1(x3, indices2, attn_feat, self.region_aware.get_adapted_features)
        x = self.decoder1_extra(x, None, attn_feat, self.region_aware.get_adapted_features)
        x = self.decoder2(x, indices1, attn_feat, self.region_aware.get_adapted_features)
        
        # 最终输出
        x = self.final_up(x)
        region_map_resized = F.interpolate(region_map, size=x.shape[2:],
                                          mode='bilinear', align_corners=True)
        x = torch.cat([x, region_map_resized], dim=1)
        x = self.final_conv(x)
        
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        if self.deep_supervision:
            return [aux1, aux2, x]
        
        return x