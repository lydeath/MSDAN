from inspect import Parameter

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Softmax, Conv2d


class ASPP(nn.Module):  # deeplab

    def __init__(self, dim, in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim),
                                       nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2),
                                   nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4),
                                   nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6),
                                   nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU())

    def forward(self, x):
        print('输入')
        print(x.shape)
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        x = self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))
        print('输出')
        print(x.shape)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


if __name__ == '__main__':
    ts = torch.Tensor(16, 6, 100, 100)
    vr = Variable(ts)
    net = ASPP(dim=6, in_dim=32)
    net = net(vr)
