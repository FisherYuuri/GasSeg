# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from ..utils import DAPPM, PAPPM, BasicBlock, Bottleneck
import numpy as np
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
class PagFM(BaseModule):
    """Pixel-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        channels (int): The number of channels.
        after_relu (bool): Whether to use ReLU before attention.
            Default: False.
        with_channel (bool): Whether to use channel attention.
            Default: False.
        upsample_mode (str): The mode of upsample. Default: 'bilinear'.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(typ='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 after_relu: bool = False,
                 with_channel: bool = False,
                 upsample_mode: str = 'bilinear',
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(typ='ReLU', inplace=True),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.after_relu = after_relu
        self.with_channel = with_channel
        self.upsample_mode = upsample_mode
        self.f_i = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(
            in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if with_channel:
            self.up = ConvModule(
                channels, in_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        if after_relu:
            self.relu = MODELS.build(act_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.

        Returns:
            Tensor: The feature map with pixel-attention-guided fusion.
        """
        if self.after_relu:
            x_p = self.relu(x_p)
            x_i = self.relu(x_i)

        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        f_p = self.f_p(x_p)

        if self.with_channel:
            sigma = torch.sigmoid(self.up(f_p * f_i))
        else:
            sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))

        x_i = F.interpolate(
            x_i,
            size=x_p.shape[2:],
            mode=self.upsample_mode,
            align_corners=False)

        out = sigma * x_i + (1 - sigma) * x_p
        return out


class Bag(BaseModule):
    """Boundary-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The kernel size of the convolution. Default: 3.
        padding (int): The padding of the convolution. Default: 1.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer.
            Default: dict(order=('norm', 'act', 'conv')).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 conv_cfg: OptConfigType = dict(order=('norm', 'act', 'conv')),
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        """Forward function.

        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.
            x_d (Tensor): The featrue map from D branch.

        Returns:
            Tensor: The feature map with boundary-attention-guided fusion.
        """
        sigma = torch.sigmoid(x_d)
        return self.conv(sigma * x_p + (1 - sigma) * x_i)


class LightBag(BaseModule):
    """Light Boundary-attention-guided fusion module.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer. Default: None.
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.f_p = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.f_i = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x_p: Tensor, x_i: Tensor, x_d: Tensor) -> Tensor:
        """Forward function.
        Args:
            x_p (Tensor): The featrue map from P branch.
            x_i (Tensor): The featrue map from I branch.
            x_d (Tensor): The featrue map from D branch.

        Returns:
            Tensor: The feature map with light boundary-attention-guided
                fusion.
        """
        sigma = torch.sigmoid(x_d)

        f_p = self.f_p((1 - sigma) * x_i + x_p)
        f_i = self.f_i(x_i + sigma * x_p)

        return f_p + f_i

class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

class LSKA(nn.Module):
    # Large-Separable-Kernel-Attention
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/tree/main
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=(2,0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1,1), padding=(0,(3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1,1), padding=((3-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=(4,0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1,1), padding=(0,9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1,1), padding=(9,0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1,1), padding=(0,15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1,1), padding=(15,0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1,1), padding=(0,18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1,1), padding=(18,0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1,1), padding=(0,24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1,1), padding=(24,0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn


class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]


class EIEStem(nn.Module):
    def __init__(self, inc, hidc, ouc) -> None:
        super().__init__()

        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.sobel_branch(x), self.pool_branch(x)], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

@MODELS.register_module()
class PIDNet(BaseModule):
    """PIDNet backbone.

    This backbone is the implementation of `PIDNet: A Real-time Semantic
    Segmentation Network Inspired from PID Controller
    <https://arxiv.org/abs/2206.02066>`_.
    Modified from https://github.com/XuJiacong/PIDNet.

    Licensed under the MIT License.

    Args:
        in_channels (int): The number of input channels. Default: 3.
        channels (int): The number of channels in the stem layer. Default: 64.
        ppm_channels (int): The number of channels in the PPM layer.
            Default: 96.
        num_stem_blocks (int): The number of blocks in the stem layer.
            Default: 2.
        num_branch_blocks (int): The number of blocks in the branch layer.
            Default: 3.
        align_corners (bool): The align_corners argument of F.interpolate.
            Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU', inplace=True).
        init_cfg (dict): Config dict for initialization. Default: None.
    """

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer
        # self.stem = EIEStem(in_channels, channels, channels)
        # self.addstem = self._make_layer(
        #         BasicBlock, channels, channels * 2, num_stem_blocks, stride=2)
        self.stem = self._make_stem_layer(in_channels, channels,
                                          num_stem_blocks)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2**(i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2))

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

        # D Branch
        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1)
            ])
            channel_expand = 1
            # spp_module = SPPF_LSKA
            spp_module = PAPPM
            dfm_module = LightBag
            act_cfg_dfm = None
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2,
                                        channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2)
            ])
            channel_expand = 2
            spp_module = DAPPM
            dfm_module = Bag
            act_cfg_dfm = act_cfg

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.spp = spp_module(
            channels * 16, ppm_channels, channels * 4, num_scales=5)
        # self.spp = spp_module(
        #     channels * 16, channels * 4)
        self.dfm = dfm_module(
            channels * 4, channels * 4, norm_cfg=norm_cfg, act_cfg=act_cfg_dfm)

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1))

    def _make_stem_layer(self, in_channels: int, channels: int,
                         num_blocks: int) -> nn.Sequential:
        """Make stem layer.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.

        Returns:
            nn.Sequential: The stem layer.
        """

        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.append(
            self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _make_layer(self,
                    block: BasicBlock,
                    in_channels: int,
                    channels: int,
                    num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        """Make layer for PIDNet backbone.
        Args:
            block (BasicBlock): Basic block.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Sequential: The Branch Layer.
        """
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))
        return nn.Sequential(*layers)

    def _make_single_layer(self,
                           block: Union[BasicBlock, Bottleneck],
                           in_channels: int,
                           channels: int,
                           stride: int = 1) -> nn.Module:
        """Make single layer for PIDNet backbone.
        Args:
            block (BasicBlock or Bottleneck): Basic block or Bottleneck.
            in_channels (int): Number of input channels.
            channels (int): Number of output channels.
            stride (int): Stride of the first block. Default: 1.

        Returns:
            nn.Module
        """

        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None)
        return block(
            in_channels, channels, stride, downsample, act_cfg_out=None)

    def init_weights(self):
        """Initialize the weights in backbone.

        Since the D branch is not initialized by the pre-trained model, we
        initialize it with the same method as the ResNet.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_cfg is not None:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], map_location='cpu')
            self.load_state_dict(ckpt, strict=False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor or tuple[Tensor]: If self.training is True, return
                tuple[Tensor], else return Tensor.
        """
        w_out = x.shape[-1] // 8
        h_out = x.shape[-2] // 8

        # stage 0-2
        x = self.stem(x)
        # stage 3
        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.d_branch_layers[0](x)

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_p = x_p.clone()

        # stage 4
        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.d_branch_layers[1](self.relu(x_d))

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.training:
            temp_d = x_d.clone()

        # stage 5
        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.d_branch_layers[2](self.relu(x_d))

        x_i = self.spp(x_i)
        x_i = F.interpolate(
            x_i,
            size=[h_out, w_out],
            mode='bilinear',
            align_corners=self.align_corners)
        out = self.dfm(x_p, x_i, x_d)
        return (temp_p, out, temp_d) if self.training else out
