"""
Create by Chengqi.Lv
2020/3/23
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
import torch
import torch.nn as nn

BN_MOMENTUM = 0.1

def fill_up_weights(up):
    for m in up.modules():
        if isinstance(m, nn.ConvTranspose2d):
            w = m.weight.data
            f = math.ceil(w.size(2) / 2)
            c = (2 * f - 1 - f % 2) / (2. * f)
            for i in range(w.size(2)):
                for j in range(w.size(3)):
                    w[0, 0, i, j] = \
                        (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
            for c in range(1, w.size(0)):
                w[c, 0, :, :] = w[0, 0, :, :]
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Vggx2SmallDeconvFPN(nn.Module):
    def __init__(self,
                 input_channel=(64,),
                 activation='ReLU',
                 deconv_channels = [48, 40, 32],
                 inputFPN_channels = [32, 16, 12]
                 ):
        super(Vggx2SmallDeconvFPN, self).__init__()
        assert isinstance(input_channel, tuple)
        assert len(input_channel) == 1, 'deconv upsample module does not support feature fusion, please use FPN'
        self.inplanes = input_channel[0]
        self.deconv_channels = deconv_channels
        self.inputFPN_channels = inputFPN_channels
        
        self.deconv_layer1=nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=128,
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv_layer2=nn.Sequential(

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=128,
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.deconv_layer3=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=128,
                bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )

        self.deconv_layer4=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=64,
                bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

        )

        self.layer = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


    def forward(self, input):
        out = self.deconv_layer1(input[4]) + input[3]
        out = self.deconv_layer2(out) + input[2]
        out = self.deconv_layer3(out) + input[1]
        out = self.deconv_layer4(out) + input[0]
        out = self.layer(out)
        return out
