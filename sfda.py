import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from tensorboard import summary
from torch import Tensor
from typing import Any, List, Tuple
from collections import OrderedDict


class _DenseLayer(nn.Module):
    """DenseBlock中的内部结构 DenseLayer: BN + ReLU + Conv(1x1) + BN + ReLU + Conv(3x3)"""

    def __init__(self,
                 num_input_features: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        """
        :param num_input_features: 输入channel
        :param growth_rate: 论文中的 k = 32
        :param bn_size: 1x1卷积的filternum = bn_size * k  通常bn_size=4
        :param drop_rate: dropout 失活率
        :param memory_efficient: Memory-efficient版的densenet  默认是不使用的
        """
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=num_input_features,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:  # inputs: [16,64,56,56]（输入） [16,32,56,56] [16,32,56,56]

        # 以DenseBlock=1为例  每一次都会叠加上一个DenseLayer的输出（channel=32） 64+32*5=224
        # [16,64,56,56] [16,96,56,56] [16,128,56,56] [16,160,56,56] [16,192,56,56] [16,224,56,56]
        # 注意：这个concat和之后的DenseBlock中的concat非常重要，理解这两句就能理解DenseNet中密集连接的精髓
        concat_features = torch.cat(inputs, 1)

        # 以DenseBlock=1为例 bottleneck_output: 一直都是生成[16,128,56,56]
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        """判断是否需要更新梯度（training）"""
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        """
        torch.utils.checkpoint: 用计算换内存（节省内存）。 详情可看： https://arxiv.org/abs/1707.06990
        torch.utils.checkpoint并不保存中间激活值，而是在反向传播时重新计算它们。 它可以应用于模型的任何部分。
        具体而言，在前向传递中,function将以torch.no_grad()的方式运行,即不存储中间激活值 相反,前向传递将保存输入元组和function参数。
        在反向传播时，检索保存的输入和function参数，然后再次对函数进行正向计算，现在跟踪中间激活值，然后使用这些激活值计算梯度。
        """

        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):  # 确保inputs的格式满足要求
            prev_features = [inputs]
        else:
            prev_features = inputs

        # 判断是否使用memory_efficient的densenet  and  是否需要更新梯度（training）
        # torch.utils.checkpoint不适用于torch.autograd.grad（）,而仅适用于torch.autograd.backward（）
        if self.memory_efficient and self.any_requires_grad(prev_features):
            # torch.jit 模式下不合适用memory_efficient
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            # 调用efficient densenet  思路：用计算换显存
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            # 调用普通的densenet
            # 以DenseBlock=1为例 bottleneck_output: 一直都是生成[16,128,56,56]
            bottleneck_output = self.bn_function(prev_features)

        # 以DenseBlock=1为例 new_features: 一直都是生成[16,32,56,56]
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        """
        :param num_layers: 该DenseBlock中DenseLayer的个数
        :param num_input_features: 该DenseBlock的输入Channel，每经过一个DenseBlock都会进行叠加
                        叠加方式：num_features = num_features + num_layers * growth_rate
        :param bn_size: 1x1卷积的filternum = bn_size*k  通常bn_size=4
        :param growth_rate: 指的是论文中的k  小点比较好  论文中是32
        :param drop_rate: dropout rate after each dense layer
        :param memory_efficient: If True, uses checkpointing. Much more memory efficient
        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        # 以DenseBlock1 为例子：features=7个List  [16,64,56,56]（输入） [16,32,56,56]x6
        # 每一个[16,32,56,56]都是用当前DenseBlock的之前所有的信息concat之后得到的
        features = [init_features]
        for name, layer in self.items():  # 遍历该DenseBlock的所有DenseLayer
            new_features = layer(features)
            features.append(new_features)  # 将该层从输出append进new_features，传给下一个DenseBlock，在bn_function中append

        # 获取之前所有层（DenseLayer）的信息  并传给下一个Transition层
        # 以DenseBlock=1为例  传给Transition数据为 [16,64+32*6,56,56]
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self,
                 num_input_features: int,
                 num_output_features: int):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features,
                                          num_output_features,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC"""

    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False):
        """
        :param growth_rate: 指的是论文中的k  小点比较好  论文中是32
        :param block_config: 每一个DenseBlock中_DenseLayer的个数
        :param num_init_features: 整个网络第一个卷积（Conv0）的kernel_size = 64
        :param bn_size: 1x1卷积的filternum = bn_size*k  通常bn_size=4
        :param drop_rate: dropout rate after each dense layer 一般为0 不用的
        :param num_classes: 数据集类别数
        :param memory_efficient: If True, uses checkpointing. Much more memory efficient
        """
        super(DenseNet, self).__init__()

        # first conv0+bn0+relu0+pool0
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # num_features：DenseBlock的输入Channel，每经过一个DenseBlock都会进行叠加
        # 叠加方式：num_features = num_features + num_layers * growth_rate
        num_features = num_init_features
        # each dense block
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            # Transition个数 = DenseBlock个数-1  DenseBlock4后面是没有Transition的
            # 经过Transition channel直接 // 2
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # finnal batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(**kwargs: Any) -> DenseNet:
    # Top-1 error: 25.35%
    # 'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
    return DenseNet(growth_rate=32,  # k=32
                    block_config=(6, 12, 24, 16),  # 每一个 DenseBlock 中 _DenseLayer的个数
                    num_init_features=64,  # 第一个Dense Block的输入channel
                    **kwargs)


def densenet169(**kwargs: Any) -> DenseNet:
    # Top-1 error: 24.00%
    # 'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32, 32),
                    num_init_features=64,
                    **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.80%
    # 'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth'
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48, 32),
                    num_init_features=64,
                    **kwargs)


def densenet161(**kwargs: Any) -> DenseNet:
    # Top-1 error: 22.35%
    # 'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36, 24),
                    num_init_features=96,
                    **kwargs)


if __name__ == '__main__':
    """测试模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = densenet121()
    # summary(model, (3, 224, 224))
    model_pre_weight_path = "./weights/densenet121_pre.pth"
    model.load_state_dict(torch.load(model_pre_weight_path, map_location=device), strict=False)
    in_channel = model.classifier.in_features  # 得到fc层的输入channel
    model.classifier = nn.Linear(in_channel, 5)

    print(model)
