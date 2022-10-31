"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

__all__ = ['SplAtConv2d']

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    基数cardinality =groups= 1 groups对应nn.conv2d的一个参数，即特征层内的cardinal组数
    基数radix = 2  用于SplAtConv2d block中的特征通道数的放大倍数，即cardinal组内split组数
    reduction_factor =4 缩放系数用于fc2和fc3之间减少参数量
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        # padding=1 => (1, 1)
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        # reduction_factor主要用于减少三组卷积的通道数，进而减少网络的参数量
        # inter_channels 对应fc1层的输出通道数 (64*2//4, 32)=>32
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        # 注意这里使用了深度可分离卷积 groups !=1，实现对不同radix组的特征层进行分离的卷积操作

        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                           groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)

        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        # [1,64,h,w] = [1,128,h,w]
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        # rchannel通道数量
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            # [1, 128, h, w] = [[1,64,h,w], [1,64,h,w]]
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
             # [[1,64,h,w], [1,64,h,w]] => [1,64,h,w]
            gap = sum(splited)
        else:
            gap = x
        # [1,64,h,w] => [1, 64, 1, 1]
        gap = F.adaptive_avg_pool2d(gap, 1)
        # [1, 64, 1, 1] => [1, 32, 1, 1]
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        # [1, 32, 1, 1] => [1, 128, 1, 1]
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        # attens [[1,64,1,1], [1,64,1,1]]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            # [1,64,1,1]*[1,64,h,w] => [1,64,h,w]
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        # contiguous()这个函数，把tensor变成在内存中连续分布的形式
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            # [1, 128, 1, 1] => [1, 2, 1, 64]
            # 分组进行softmax操作
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            # 对radix维度进行softmax操作
            x = F.softmax(x, dim=1)
            # [1, 2, 1, 64] => [1, 128]
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x