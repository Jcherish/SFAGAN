from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
#from scipy import ndimage
from swinmodel import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self,  img_size=256, num_classes=1, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head


        self.swin_unet = SwinTransformerSys(img_size=256,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1,
                                embed_dim=96,
                                depths=[2,2,2,2],
                                num_heads=[3,6,12,24],
                                window_size=8,
                                mlp_ratio=4,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits
# from thop import profile
# x = torch.rand(4,3,256,256)
# model=SwinUnet()
# flops, params = profile(model, (x,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))