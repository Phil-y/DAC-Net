import torch
import torch.fft
import torch.nn.functional as F
import numpy as np
from torch import nn
from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from typing import Type, Callable, Tuple, Optional, Set, List, Union


import math
import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding









def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()


class DABlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DABlock, self).__init__()
        inter_channels = in_channels // 16

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.pa = PAM_Unit(inter_channels)
        self.ca = CAM_Unit(inter_channels)

        self.conv5b = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv5d = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

    def forward(self, x):

        feat1 = self.conv5a(x)
        pa_feat = self.pa(feat1)
        pa_conv = self.conv5b(pa_feat)

        feat2 = self.conv5c(x)
        ca_feat = self.ca(feat2)
        ca_conv = self.conv5d(ca_feat)

        feat_sum = pa_conv + ca_conv

        paca_output = self.conv6(feat_sum)
        return paca_output






    
class DAC_Net(nn.Module):
    def __init__(self, in_c=3, n_classes=1, dim=[32,64,128,256,512],split='fc',bridge=True):
        super().__init__()

        self.bridge = bridge

        self.e1 = nn.Sequential(
            nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),
        )
        print()
        self.e2 = nn.Sequential(
            nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),

        )
        self.e3 = nn.Sequential(
            DWSEBlock(dim[1], dim[2]),

        )
        self.e4 = nn.Sequential(
            nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),


        )
        self.e5 = nn.Sequential(
            nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),


        )
        if bridge:
            self.cssc = CSScale_Conection(dim, split)


        self.d5 = nn.Sequential(
            nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),
        )
        self.d4 = nn.Sequential(
           nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),
        )
        self.d3 = nn.Sequential(
            DWSEBlock(dim[2], dim[1]),
        )
        self.d2 = nn.Sequential(
           nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),
        )

        self.d1 = nn.Sequential(
            nn.Conv2d(dim[0], n_classes,kernel_size=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # ------encoder------#
        out = F.gelu(F.max_pool2d(self.ebn1(self.e1(x)), 2, 2))
        t1 = out
        # print("t1 shape:{}".format(t1.shape))
        out = F.gelu(F.max_pool2d(self.ebn2(self.e2(out)), 2, 2))
        t2 = out
        # print("t2 shape:{}".format(t2.shape))
        out = F.gelu(F.max_pool2d(self.ebn3(self.e3(out)), 2, 2))
        t3 = out
        # print("t3 shape:{}".format(t3.shape))
        out = F.gelu(F.max_pool2d(self.ebn4(self.e4(out)), 2, 2))
        t4 = out
        # print("t4 shape:{}".format(t4.shape))
        out = F.gelu(self.e5(out))


        if self.bridge: t1, t2, t3, t4 =self.cssc(t1, t2, t3, t4)
        # ------decoder------#

        out5 = F.gelu(self.dbn1(self.d5(out)))
        # print("out5 shape:{}".format(out5.shape))
        out5 = torch.add(out5, t4)
        # print("out5 shape:{}".format(out5.shape))
        out4 = F.gelu(F.interpolate(self.dbn2(self.d4(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))
        # print("out4 shape:{}".format(out4.shape))
        out4 = torch.add(out4, t3)
        # print("out4 shape:{}".format(out4.shape))
        out3 = F.gelu(F.interpolate(self.dbn3(self.d3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))

        out3 = torch.add(out3, t2)
        out2 = F.gelu(F.interpolate(self.dbn4(self.d2(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))


        out2 = torch.add(out2, t1)
        out1 = F.interpolate(self.d1(out2), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)

        return torch.sigmoid(out1)



