import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
import numpy as np
from net_block import pac
from utils import *

import os
import glob

value_in_max = 255
value_out_max = 255

n_epoch = 100
lr = 1e-4
lr_milestone = [40, 80]
lr_gamma = 0.5

lres_npz_dir = '../dataset/lres_npz'
hres_npz_dir = '../dataset/hres_npz'
predict_npz_dir = '../dataset/test_npz'

train_first_percentage = 0.98
test_first_percentage = -0.02

scale = 4
patch_size = 128
patch_stride = 100
nframes = 3
channel = 'all'
batch_size = 12
num_workers = 16
num_workers_test = 6
aug = True
train_sample_reduce = 0.06


# batch_size = 2
# num_workers = 0
# num_workers_test = 0


class VConv2d(nn.modules.conv._ConvNd):
    """
    Versatile Filters
    Paper: https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', delta=0, g=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(VConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # super(VConv2d, self).__init__(
        #     in_channels, out_channels, kernel_size, stride, padding, dilation,
        #     False, _pair(0), groups, bias)
        self.s_num = int(np.ceil(self.kernel_size[0]/2))  # s in paper
        self.delta = delta  # c-\hat{c} in paper
        self.g = g  # g in paper
        self.weight = nn.Parameter(torch.Tensor(
            int(out_channels/self.s_num/(1+self.delta/self.g)), in_channels // groups, *kernel_size))
        self.reset_parameters()

    def forward(self, x):
        x_list = []
        s_num = self.s_num
        ch_ratio = (1+self.delta/self.g)
        ch_len = self.in_channels - self.delta
        for s in range(s_num):
            for start in range(0, self.delta+1, self.g):
                weight1 = self.weight[:, :ch_len,
                                      s:self.kernel_size[0]-s, s:self.kernel_size[0]-s]
                if self.padding[0]-s < 0:
                    h = x.size(2)
                    x1 = x[:, start:start+ch_len, s:h-s, s:h-s]
                    padding1 = _pair(0)
                else:
                    x1 = x[:, start:start+ch_len, :, :]
                    padding1 = _pair(self.padding[0]-s)
                x_list.append(F.conv2d(x1, weight1, self.bias[int(self.out_channels*(s*ch_ratio+start)/s_num/ch_ratio):int(self.out_channels*(s*ch_ratio+start+1)/s_num/ch_ratio)], self.stride,
                                       padding1, self.dilation, self.groups))
        x = torch.cat(x_list, 1)
        return x


class GRU(nn.Module):
    def __init__(self, inc):
        super().__init__()

        self.conv_r = nn.Sequential(
            VConv2d(inc*2, inc, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_z = nn.Sequential(
            VConv2d(inc*2, inc, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_n = nn.Sequential(
            VConv2d(inc*2, inc, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, h0):
        """out_ch = in_ch"""
        r = self.conv_r(torch.cat((x, h0), dim=1))
        z = self.conv_z(torch.cat((x, h0), dim=1))
        n = self.conv_n(torch.cat((x, r*h0), dim=1))
        h = (1-z)*n+z*h0
        return h

class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional dimensions
        - Output: (N, *), same shape as the input
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class VDM(nn.Module):
    # video denoise module via using 3D residual cnn
    def __init__(self, in_ch, feat_ch, out_ch):
        super().__init__()

        # extract feature
        self.conv1 = nn.Conv3d(
            in_ch, feat_ch, kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2))

        self.layer1 = nn.Sequential(
            nn.Conv3d(feat_ch, feat_ch*2, kernel_size=(3, 1, 1),
                      stride=1, padding=(1, 0, 0)),
            Mish(),
            nn.Conv3d(feat_ch*2, feat_ch*2, kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1)),
            Mish(),
            nn.Conv3d(feat_ch*2, feat_ch*2,
                      kernel_size=3, stride=1, padding=1),
        )

        self.layer2_1 = nn.Sequential(
            nn.Conv3d(feat_ch*2, feat_ch, kernel_size=3,
                      stride=1, padding=2, dilation=2),
            Mish(),
            nn.Conv3d(feat_ch, feat_ch, kernel_size=3,
                      stride=1, padding=2, dilation=2),
        )
        self.layer2_2 = nn.Sequential(
            nn.Conv3d(feat_ch*2, feat_ch, kernel_size=3,
                      stride=1, padding=4, dilation=4),
            Mish(),
            nn.Conv3d(feat_ch, feat_ch, kernel_size=3,
                      stride=1, padding=4, dilation=4)
        )
        self.layer2_3 = nn.Sequential(
            nn.Conv3d(feat_ch*2, feat_ch, kernel_size=3,
                      stride=1, padding=8, dilation=8),
            Mish(),
            nn.Conv3d(feat_ch, feat_ch, kernel_size=3,
                      stride=1, padding=8, dilation=8)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(feat_ch*3, feat_ch*2, kernel_size=(3, 1, 1),
                      stride=1, padding=(1, 0, 0)),
            Mish(),
            nn.Conv3d(feat_ch*2, feat_ch*2, kernel_size=(1, 3, 3),
                      stride=1, padding=(0, 1, 1)),
            Mish(),
            nn.Conv3d(feat_ch*2, feat_ch*2,
                      kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = nn.Conv3d(
            feat_ch*2, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.layer1(output)

        output1 = self.layer2_1(output)
        output2 = self.layer2_2(output)
        output3 = self.layer2_3(output)
        output = self.layer3(torch.cat((output1, output2, output3), dim=1))

        output = self.conv2(output)

        return x - output


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = VConv2d(in_channels=ch, out_channels=ch*2,
                             kernel_size=3, stride=1, padding=1)
        self.mish = Mish()
        self.conv2 = VConv2d(in_channels=ch*2, out_channels=ch,
                             kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """out_ch = in_ch"""
        return self.conv2(self.mish(self.conv1(x))) + x

class ReconBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up1 = VConv2d(in_channels=in_ch, out_channels=out_ch *
                           2*4, kernel_size=3, stride=1, padding=1)
        self.res1 = ResBlock(out_ch*2)
        self.res2 = ResBlock(out_ch*2)
        self.res3 = ResBlock(out_ch*2)

        self.up2 = VConv2d(
            in_channels=out_ch*2, out_channels=out_ch*4, kernel_size=3, stride=1, padding=1)
        self.res4 = ResBlock(out_ch)
        self.res5 = ResBlock(out_ch)
        self.res6 = ResBlock(out_ch)

        self.ps = nn.PixelShuffle(2)

    def forward(self, feat):
        # inputs: multi-scale features
        up1 = self.up1(feat)
        up1 = self.ps(up1)
        res = self.res1(up1)
        res = self.res2(res)
        res = self.res3(res) + up1

        up2 = self.up2(res)
        up2 = self.ps(up2)
        res = self.res4(up2)
        res = self.res5(res) 
        res = self.res6(res) + up2

        return res


class GlobalFeatureExtract(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.gfe = nn.Sequential(
            ResBlock(in_ch),
            ResBlock(in_ch),
            ResBlock(in_ch),

            VConv2d(in_channels=in_ch, out_channels=in_ch *
                    2, kernel_size=3, stride=2, padding=1),
            ResBlock(in_ch*2),
            ResBlock(in_ch*2),
            ResBlock(in_ch*2),

            VConv2d(in_channels=in_ch*2, out_channels=in_ch *
                    4, kernel_size=3, stride=2, padding=1),
            ResBlock(in_ch*4),
            ResBlock(in_ch*4),
            ResBlock(in_ch*4),
            nn.AdaptiveAvgPool2d(1),
            ResBlock(in_ch*4),
        )

    def forward(self, x):
        """out_ch = in_ch * 4"""
        return torch.tanh(self.gfe(x))


class LocalFeatureExtract(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.lfe_1 = nn.Sequential(
            ResBlock(in_ch),
            ResBlock(in_ch),
            ResBlock(in_ch)
        )

        self.gru_1 = GRU(in_ch)

        self.lfe_conv1 = VConv2d(
            in_channels=in_ch, out_channels=in_ch*2, kernel_size=3, stride=2, padding=1)
        self.lfe_2 = nn.Sequential(
            ResBlock(in_ch*2),
            ResBlock(in_ch*2),
            ResBlock(in_ch*2)
        )

        self.gru_2 = GRU(in_ch*2)

        self.lfe_conv2 = VConv2d(
            in_channels=in_ch*2, out_channels=in_ch*4, kernel_size=3, stride=2, padding=1)
        self.lfe_3 = nn.Sequential(
            ResBlock(in_ch*4),
            ResBlock(in_ch*4),
            ResBlock(in_ch*4)
        )

    def forward(self, x):
        """out_ch = in_ch * 4"""
        lfe_1 = self.lfe_1(x)
        x = self.gru_1(lfe_1, x)

        x = self.lfe_conv1(x)
        lfe_2 = self.lfe_2(x)
        x = self.gru_2(lfe_2, x)

        x = self.lfe_conv2(x)
        x = self.lfe_3(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 6*nframes
        output_channel = 8

        self.vdm = VDM(in_ch=6, feat_ch=32, out_ch=6)

        self.conv = nn.Conv2d(in_channels=input_channel, out_channels=32,
                              kernel_size=1, stride=1, padding=0)

        # two encoders
        self.l_feat = LocalFeatureExtract(in_ch=32)
        self.g_feat = GlobalFeatureExtract(in_ch=32)

        # fuse local and global feature
        self.fuse_l_g_feat = ResBlock(128)
        self.gru = GRU(128)

        # decoder
        self.decoder = ReconBlock(128, 32)
        self.decoder_rec = ResBlock(64)
        self.decoder_conv = VConv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # reconstruction
        self.upconv1 = VConv2d(
            in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1)
        self.upconv2 = VConv2d(
            in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1)

        self.mish = Mish()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(
            in_channels=64, out_channels=output_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.vdm(x)  # ch = 32
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

        pad_size0 = x.shape[-1] % 4
        pad_size1 = x.shape[-2] % 4
        pad_size = [(pad_size0+1)//2, pad_size0//2,
                    (pad_size1+1)//2, pad_size1//2]
        x = torch.nn.functional.pad(x, pad_size, mode='replicate')

        vc_x = self.conv(x)  # ch = 32

        l_feat = self.l_feat(vc_x)  # ch = 128
        g_feat = self.g_feat(vc_x)  # ch = 128

        fuse_lg = self.fuse_l_g_feat(g_feat*l_feat)  # ch = 128
        fuse_lg = self.gru(fuse_lg, l_feat)  # ch = 128
        fuse_lg = self.decoder(fuse_lg)  # ch = 32
        fuse_lg = torch.cat((fuse_lg, vc_x), dim=1)  # ch = 64
        fuse_lg = self.decoder_rec(fuse_lg)  # ch = 64
        fuse_lg = self.decoder_conv(fuse_lg)  # ch = 64

        out = self.mish(self.pixel_shuffle(
            self.upconv1(fuse_lg)))  # ch = 64
        out = self.mish(self.pixel_shuffle(
            self.upconv2(out)))  # ch = 64
        out = self.conv_last(out)  # ch = out_channel

        out_size = [0 + 4*(pad_size[2]), out.shape[-2] - 4*(pad_size[3]),
                    0 + 4*(pad_size[0]), out.shape[-1] - 4*(pad_size[1])]
        
        out = out[:, :, out_size[0]:out_size[1], out_size[2]:out_size[3]]
        return out


# Loss function begin <<<


class Loss(nn.Module):
    """Charbonnier Loss (L1) + CosineSimilarity"""

    def __init__(self, eps=1e-6, loss_lambda=2):
        super().__init__()
        self.eps = eps
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=eps)
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        cosine_term = (1 - self.similarity(x, y)).mean()

        return loss + self.loss_lambda * cosine_term

cbloss = Loss()
cbloss = cbloss.cuda()

def criterion(output, target):
    return cbloss(output, target)
# Loss Function end >>>
