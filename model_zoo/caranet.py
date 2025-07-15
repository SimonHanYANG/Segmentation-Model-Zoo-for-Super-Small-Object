import torch
import torch.nn as nn
import torch.nn.functional as F
from .pretrain.Res2Net_v1b import res2net50_v1b_26w_4s, res2net101_v1b_26w_4s
from .lib.conv_layer import Conv, BNPReLU
from .lib.axial_atten import AA_kernel
from .lib.context_module import CFPModule
from .lib.partial_decoder import aggregation

__all__ = ['CaraNet']

class CaraNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()
        
        # ---- ResNet Backbone ----
        self.resnet = res2net101_v1b_26w_4s(pretrained=True)
        
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Receptive Field Block
        self.rfb2_1 = Conv(512, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb3_1 = Conv(1024, 32, 3, 1, padding=1, bn_acti=True)
        self.rfb4_1 = Conv(2048, 32, 3, 1, padding=1, bn_acti=True)

        # Partial Decoder
        self.agg1 = aggregation(32)
        
        self.CFP_1 = CFPModule(32, d=8)
        self.CFP_2 = CFPModule(32, d=8)
        self.CFP_3 = CFPModule(32, d=8)

        # AA kernels with fixed output channels
        self.aa_kernel_1 = nn.Sequential(
            self.CFP_1,
            Conv(32, 32, 1, 1, padding=0, bn_acti=True)
        )
        self.aa_kernel_2 = nn.Sequential(
            self.CFP_2,
            Conv(32, 32, 1, 1, padding=0, bn_acti=True)
        )
        self.aa_kernel_3 = nn.Sequential(
            self.CFP_3,
            Conv(32, 32, 1, 1, padding=0, bn_acti=True)
        )

        # Reverse attention blocks
        self.ra1_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra2_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        self.ra3_conv1 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(32, 32, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(32, 1, 3, 1, padding=1, bn_acti=True)
        
        # 修改最后的输出层以适应多类别分割
        if num_classes > 1:
            self.ra1_conv3 = Conv(32, num_classes, 3, 1, padding=1, bn_acti=True)
            self.ra2_conv3 = Conv(32, num_classes, 3, 1, padding=1, bn_acti=True)
            self.ra3_conv3 = Conv(32, num_classes, 3, 1, padding=1, bn_acti=True)
        
        self.deep_supervision = deep_supervision

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        
        # ----------- low-level features -------------
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        
        x2_rfb = self.rfb2_1(x2) # 512 - 32
        x3_rfb = self.rfb3_1(x3) # 1024 - 32
        x4_rfb = self.rfb4_1(x4) # 2048 - 32
        
        decoder_1 = self.agg1(x4_rfb, x3_rfb, x2_rfb) # 1,44,44
        lateral_map_1 = F.interpolate(decoder_1, scale_factor=8, mode='bilinear', align_corners=False)
        
        # ------------------- atten-one -----------------------
        decoder_2 = F.interpolate(decoder_1, scale_factor=0.25, mode='bilinear', align_corners=False)
        cfp_out_1 = x4_rfb  # 将直接使用x4_rfb
        decoder_2_ra = -1*(torch.sigmoid(decoder_2)) + 1
        
        aa_atten_3 = self.aa_kernel_3(cfp_out_1)
        aa_atten_3_o = decoder_2_ra.expand(-1, 32, -1, -1).mul(aa_atten_3)
        
        ra_3 = self.ra3_conv1(aa_atten_3_o) # 32 - 32
        ra_3 = self.ra3_conv2(ra_3) # 32 - 32
        ra_3 = self.ra3_conv3(ra_3) # 32 - 1
        
        x_3 = ra_3 + decoder_2
        lateral_map_2 = F.interpolate(x_3, scale_factor=32, mode='bilinear', align_corners=False)
        
        # ------------------- atten-two -----------------------      
        decoder_3 = F.interpolate(x_3, scale_factor=2, mode='bilinear', align_corners=False)
        cfp_out_2 = x3_rfb  # 将直接使用x3_rfb
        decoder_3_ra = -1*(torch.sigmoid(decoder_3)) + 1
        
        aa_atten_2 = self.aa_kernel_2(cfp_out_2)
        aa_atten_2_o = decoder_3_ra.expand(-1, 32, -1, -1).mul(aa_atten_2)
        
        ra_2 = self.ra2_conv1(aa_atten_2_o) # 32 - 32
        ra_2 = self.ra2_conv2(ra_2) # 32 - 32
        ra_2 = self.ra2_conv3(ra_2) # 32 - 1
        
        x_2 = ra_2 + decoder_3
        lateral_map_3 = F.interpolate(x_2, scale_factor=16, mode='bilinear', align_corners=False)        
        
        # ------------------- atten-three -----------------------
        decoder_4 = F.interpolate(x_2, scale_factor=2, mode='bilinear', align_corners=False)
        cfp_out_3 = x2_rfb  # 将直接使用x2_rfb
        decoder_4_ra = -1*(torch.sigmoid(decoder_4)) + 1
        
        aa_atten_1 = self.aa_kernel_1(cfp_out_3)
        aa_atten_1_o = decoder_4_ra.expand(-1, 32, -1, -1).mul(aa_atten_1)
        
        ra_1 = self.ra1_conv1(aa_atten_1_o) # 32 - 32
        ra_1 = self.ra1_conv2(ra_1) # 32 - 32
        ra_1 = self.ra1_conv3(ra_1) # 32 - 1
        
        x_1 = ra_1 + decoder_4
        lateral_map_5 = F.interpolate(x_1, scale_factor=8, mode='bilinear', align_corners=False)
        
        # 根据是否需要深度监督返回不同结果
        if self.deep_supervision:
            return [lateral_map_5, lateral_map_3, lateral_map_2, lateral_map_1]
        else:
            return lateral_map_5