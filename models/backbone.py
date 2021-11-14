import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    '''不对称舍入使“val”可被“除数”整除。使用默认偏差时，将向上取整，除非该数字不大于较小的可除值的10%'''
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    '''缩放张量深度，如参考MobileNet代码中所示，更喜欢向上而不是向下涂抹'''
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        # 这里在默认参数下，depth计算结果仍为[32, 16, 24, 40, 80, 96, 192, 320]
        depths = _get_depths(alpha)
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)

        self.conv0 = nn.Sequential(
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9']
        self.conv2 = MNASNet.layers._modules['10']

        # depths[4] = 80, out1: 输入通道数80，输出通道数80，卷积核大小1
        self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)
        self.out_channels = [depths[4]]

        # 最终通道数 80
        final_chs = depths[4]
        # inner1: 输入通道数40，输出通道数80，卷积核大小1 
        # inner2: 输入通道数24，输出通道数80，卷积核大小1
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)

        # out2: 输入通道数80，输出通道数40，卷积核大小3，填充1
        # out3: 输入通道数80，输出通道数24，卷积核大小3，填充1
        self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
        self.out_channels.append(depths[3])
        self.out_channels.append(depths[2])
        # out_channels最终为 [80,40,24]

    def forward(self, x):
        # 计算到mnasnet1_0的 '10'层
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        # 保存计算到mnasnet1_0的 '10'层的中间结果
        intra_feat = conv2
        outputs = []
        # 中间结果经过out1: 输入通道数80，输出通道数80，卷积核大小1
        out = self.out1(intra_feat)
        outputs.append(out)

        # 进行上采样，输入是中间计算结果，缩放因子2，采用最近邻插值
        # 上采样后再加上conv1（mnasnet1_0第9层结果）进行卷积，输入通道数40，输出通道数80，卷积核大小1 的结果。
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        # 中间结果经过out2: 输入通道数80，输出通道数40，卷积核大小3，填充1
        out = self.out2(intra_feat)
        outputs.append(out)

        # 进行上采样，输入是上一个中间计算结果，缩放因子2，采用最近邻插值
        # 上采样后再加上conv0（mnasnet1_0第8层结果）进行卷积，输入通道数40，输出通道数80，卷积核大小1 的结果
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        # 中间结果经过out3: 输入通道数80，输出通道数24，卷积核大小3，填充1
        out = self.out3(intra_feat)
        outputs.append(out)

        # 取从后向前的元素, [ 1 2 3 4 5 ] -> [ 5 4 3 2 1 ]
        return outputs[::-1]
