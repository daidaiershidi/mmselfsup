# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ALGORITHMS, build_backbone, build_head
from .base import BaseModel


def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


class HOGLayerC(nn.Module):

    def __init__(self, nbins=9, pool=8, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    def _reshape(self, hog_feat):
        hog_feat = hog_feat.flatten(1, 2)
        unfold_size = hog_feat.shape[-1] // 14
        hog_feat = (
            hog_feat.permute(0, 2, 3,
                             1).unfold(1, unfold_size, unfold_size).unfold(
                                 2, unfold_size,
                                 unfold_size).flatten(1, 2).flatten(2))
        return hog_feat

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3)
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3)
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros((b, c, self.nbins, h, w),
                          dtype=torch.float,
                          device=x.device)
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(
                    h, self.gaussian_window)
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern

        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)

        return self._reshape(out)


@ALGORITHMS.register_module()
class MaskFeat(BaseModel):
    """MaskFeat. Implementation of `Masked Autoencoders Are Scalable Vision
    Learners.
     <https://arxiv.org/abs/2111.06377>`_.
    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self, backbone=None, head=None, hog_para=None, init_cfg=None):
        super().__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert head is not None
        self.head = build_head(head)
        self.hog_layer = HOGLayerC(**hog_para)

    def init_weights(self):
        super().init_weights()

    def extract_feat(self, img, mask):
        """Function to extract features from backbone.
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
        Returns:
            tuple[Tensor]: backbone outputs.
        """
        return self.backbone(img, mask)

    def forward_train(self, input, **kwargs):
        """Forward computation during training.
        Args:
            input (Tensor, Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # start_t = time.time()
        img = input[0]
        mask = input[1]

        # # # replace fake data
        # print(img.shape, mask.shape)
        # print(
        #     '\n\nPath : /mnt/lustre/liukaiyuan.vendor/mmselfsup/mmselfsup/models/algorithms/maskfeat.py\n\n'
        # )
        # # device = img.device
        # img = torch.load(
        #     '/mnt/lustre/liukaiyuan.vendor/duiqi/pipeline/train/mm/img.wt',
        #     map_location='cpu')
        # mask = torch.load(
        #     '/mnt/lustre/liukaiyuan.vendor/duiqi/pipeline/train/mm/mask.wt',
        #     map_location='cpu')

        # # # update count
        # with open(
        #         '/mnt/lustre/liukaiyuan.vendor/duiqi/pipeline/train/mm/iter/cnt.txt',
        #         'r') as f:
        #     f = f.readline()
        # cnt = int(f)
        # if cnt == 6:  # 保存前6个iter
        #     assert 1 == 0
        # else:
        #     with open(
        #             '/mnt/lustre/liukaiyuan.vendor/duiqi/pipeline/train/mm/iter/cnt.txt',
        #             'w') as f:
        #         f = f.write(str(cnt + 1))

        # main
        hog = self.hog_layer(img)
        # print('algorithms hog:', time.time() - start_t)
        latent = self.backbone(img, mask)
        # print('algorithms backbone:', time.time() - start_t)
        losses = self.head(latent, hog, mask)
        # print('algorithms head:', time.time() - start_t)

        # # # save loss
        # # cnt = 0
        # print('\nsave {} losses, {}'.format(cnt, losses))
        # # print(self.backbone.mask_token.mean(), '\n')
        # torch.save(
        #     losses,
        #     '/mnt/lustre/liukaiyuan.vendor/duiqi/pipeline/train/mm/iter/losses_{}.wt'
        #     .format(cnt))

        return losses
