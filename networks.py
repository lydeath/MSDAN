import math
# PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F

import split_attention as split

from model import architecture

import ASPP


class MSDAN(nn.Module):
    def __init__(self, recurrent_iter=3, use_GPU=True):
        super(MSDAN, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        self.pyramid = ASPP.ASPP()

        self.SplAtConv2d = split.SplAtConv2d(in_channels=32, channels=32, kernel_size=3, stride=1, padding=1,
                                             dilation=1, groups=1, bias=True, radix=2)

        self.architecture = architecture.IMDN(in_nc=32, upscale=1)

    def forward(self, input):  # input(16, 3, 100, 100)
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input  # (16, 3, 100, 100)
        h = Variable(torch.zeros(batch_size, 32, row, col))  # (16, 32, 100, 100)
        c = Variable(torch.zeros(batch_size, 32, row, col))  # (16, 32, 100, 100)

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = input
            x = torch.cat((input, x), 1)  # (16, 6, 100, 100)

            x = self.pyramid(x)

            # x = self.conv0(x)  # (16, 32, 100, 100)

            x = torch.cat((x, h), 1)  # (16, 64, 100, 100)

            i = self.conv_i(x)  # (16, 32, 100, 100)
            f = self.conv_f(x)  # (16, 32, 100, 100)
            g = self.conv_g(x)  # (16, 32, 100, 100)
            o = self.conv_o(x)  # (16, 32, 100, 100)
            c = f * c + i * g  # (16, 32, 100, 100)
            x = o * torch.tanh(c)  # (16, 32, 100, 100)
            # print(x.shape)

            x = self.SplAtConv2d(x)

            x = self.architecture(x)

            x_list.append(x)

        return x, x_list


if __name__ == '__main__':
    ts = torch.Tensor(16, 3, 64, 64)
    vr = Variable(ts)
    net = MSDAN()


