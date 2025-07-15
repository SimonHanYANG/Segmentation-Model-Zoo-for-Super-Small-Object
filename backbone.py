'''
Author: SimonHanYANG SimonCK666@163.com
Date: 2025-06-04 10:43:37
LastEditors: SimonHanYANG SimonCK666@163.com
LastEditTime: 2025-06-05 11:05:56
FilePath: /UNeXt-pytorch/model_zoo/backbone.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch.nn as nn


class ResNet(nn.Module):
    # Load ResNet pretrained on ImageNet from torchvision, see
    # https://pytorch.org/vision/stable/models/resnet.html
    def __init__(self, resnet_type, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101, 'resnet152':resnet152}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        resnet = resnet_hub[resnet_type](pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)       # 2x down
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 4x down
        x1 = self.layer1(x)
        x2 = self.layer2(x1)      # 8x down
        x3 = self.layer3(x2)      # 16x down
        x4 = self.layer4(x3)      # 32x down

        return x1, x2, x3, x4


class Mobilenetv2(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v2

        mobilenet = mobilenet_v2(pretrained=pretrained)

        self.layer1 = mobilenet.features[:4]
        self.layer2 = mobilenet.features[4:7]
        self.layer3 = mobilenet.features[7:14]
        self.layer4 = mobilenet.features[14:18]

    def forward(self, x):
        x1 = self.layer1(x)     # 4x down
        x2 = self.layer2(x1)    # 8x down
        x3 = self.layer3(x2)    # 16x down
        x4 = self.layer4(x3)    # 32x down

        return x1, x2, x3, x4