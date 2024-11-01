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







class AtrousDualAttention(nn.Module):
    def __init__(self, in_c, out_c, k_size=3, dilated_ratio=[7,5,2,1]):
        super().__init__()
        self.atrous0 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[0] - 1)) // 2,
                              dilation=dilated_ratio[0], groups=in_c // 4)
        self.atrous1 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[1] - 1)) // 2,
                              dilation=dilated_ratio[1], groups=in_c // 4)
        self.atrous2 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[2] - 1)) // 2,
                              dilation=dilated_ratio[2], groups=in_c // 4)
        self.atrous3 = nn.Conv2d(in_c // 4, in_c // 4, kernel_size=k_size, stride=1,
                              padding=(k_size + (k_size - 1) * (dilated_ratio[3] - 1)) // 2,
                              dilation=dilated_ratio[3], groups=in_c // 4)
        self.norm = nn.GroupNorm(4, in_c)
        self.conv = nn.Conv2d(in_c, in_c, 1)

        self.da = DABlock(in_c, out_c)


    def forward(self, z):
        z = torch.chunk(z, 4, dim=1)
        z0 = self.atrous0(z[0])
        z1 = self.atrous1(z[1])
        z2 = self.atrous2(z[2])
        z3 = self.atrous3(z[3])
        z = F.gelu(self.conv(self.norm(torch.cat((z0, z1, z2, z3), dim=1))))
        z = self.da(z)
        return z







class PAM_Unit(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Unit, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Unit(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Unit, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


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







def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor





class DWSEBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
            debug=False
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(DWSEBlock, self).__init__()

        self.debug = debug
        # Save parameter
        self.drop_path_rate: float = drop_path
        # Ignore inplace parameter if GELU is used
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1)),
            DepthwiseSeparableConv(in_chs=in_channels, out_chs=out_channels, stride=1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(in_chs=out_channels, rd_ratio=0.25),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1))
        )
        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1)) \
                            if (in_channels != out_channels) else nn.Identity()


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        output = self.main_path(input)
        # print(output.shape)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
            # print(output.shape)
        output = output + self.skip_path(input)
        # print(output.shape)
        return output





class Channel_Scale_Conection(nn.Module):
    def __init__(self, c_list, split='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split = split
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0]) if split == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = nn.Linear(c_list_sum, c_list[1]) if split == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = nn.Linear(c_list_sum, c_list[2]) if split == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = nn.Linear(c_list_sum, c_list[3]) if split == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, h1, h2, h3, h4):
        H = torch.cat((self.avgpool(h1),
                         self.avgpool(h2),
                         self.avgpool(h3),
                         self.avgpool(h4),),
                         dim=1)
        # print(att.shape)
        att = self.get_all_att(H.squeeze(-1).transpose(-1, -2))
        if self.split != 'fc':
            att = att.transpose(-1, -2)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))

        if self.split == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(h1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(h2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(h3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(h4)

        else:
            att1 = att1.unsqueeze(-1).expand_as(h1)
            att2 = att2.unsqueeze(-1).expand_as(h2)
            att3 = att3.unsqueeze(-1).expand_as(h3)
            att4 = att4.unsqueeze(-1).expand_as(h4)

        return att1, att2, att3, att4


class Spatial_Scale_Conection(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4):
        t_list = [t1, t2, t3, t4]
        ssb_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            ssb = torch.cat([avg_out, max_out], dim=1)
            ssb = self.shared_conv2d(ssb)
            ssb_list.append(ssb)
        return ssb_list[0], ssb_list[1], ssb_list[2], ssb_list[3]


class CSScale_Conection(nn.Module):
    def __init__(self, c_list, split='fc'):
        super().__init__()

        self.csc = Channel_Scale_Conection(c_list, split=split)
        self.ssc = Spatial_Scale_Conection()

    def forward(self, t1, t2, t3, t4):
        r1, r2, r3, r4= t1, t2, t3, t4

        ssc1, ssc2, ssc3, ssc4 = self.ssc(t1, t2, t3, t4)
        t1, t2, t3, t4 = ssc1 * t1, ssc2 * t2, ssc3 * t3, ssc4 * t4

        r1_, r2_, r3_, r4_ = t1, t2, t3, t4
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4
        #
        csc1, csc2, csc3, csc4 = self.csc(t1, t2, t3, t4)
        t1, t2, t3, t4 = csc1 * t1, csc2 * t2, csc3 * t3, csc4 * t4

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_





    
class DAC_Net(nn.Module):
    def __init__(self, in_c=3, n_classes=1, dim=[32,64,128,256,512],split='fc',bridge=True):
        super().__init__()

        self.bridge = bridge

        self.e1 = nn.Sequential(
            nn.Conv2d(in_c, dim[0],3, stride=1, padding=1),
        )
        print()
        self.e2 = nn.Sequential(
            DWSEBlock(dim[0], dim[1]),

        )
        self.e3 = nn.Sequential(
            DWSEBlock(dim[1], dim[2]),

        )
        self.e4 = nn.Sequential(
            AtrousDualAttention(dim[2], dim[3]),


        )
        self.e5 = nn.Sequential(
            AtrousDualAttention(dim[3], dim[4]),


        )
        if bridge:
            self.cssc = CSScale_Conection(dim, split)


        self.d5 = nn.Sequential(
            AtrousDualAttention(dim[4], dim[3]),
        )
        self.d4 = nn.Sequential(
            AtrousDualAttention(dim[3], dim[2]),
        )
        self.d3 = nn.Sequential(
            DWSEBlock(dim[2], dim[1]),
        )
        self.d2 = nn.Sequential(
            DWSEBlock(dim[1], dim[0]),
        )

        self.ebn1 = nn.GroupNorm(4, dim[0])
        self.ebn2 = nn.GroupNorm(4, dim[1])
        self.ebn3 = nn.GroupNorm(4, dim[2])
        self.ebn4 = nn.GroupNorm(4, dim[3])

        self.dbn1 = nn.GroupNorm(4, dim[3])
        self.dbn2 = nn.GroupNorm(4, dim[2])
        self.dbn3 = nn.GroupNorm(4, dim[1])
        self.dbn4 = nn.GroupNorm(4, dim[0])

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



# from thop import profile
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# nets = DAC_Net().cuda()
# input = torch.randn(1, 3, 224, 224).to(device)
# flops, params = profile(nets, inputs=(input, ))
# print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops单位G，para单位M


