'''
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-06-10 17:02:20
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-06-10 19:16:13
FilePath: /UNeXt-pytorch/model_zoo/lib/self_attention.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:15:44 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layer import Conv
import math

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out