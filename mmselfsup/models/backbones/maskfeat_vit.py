# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import VisionTransformer
from mmcv.cnn.utils.weight_init import trunc_normal_
from torch import nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class MaskFeatViT(VisionTransformer):
    """Vision Transformer for MaskFeat pre-training.
    A PyTorch implement of: `Masked Autoencoders Are Scalable Vision Learners
     <https://arxiv.org/abs/2111.06377>`_.
    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        mask_ratio (bool): The ratio of total number of patches to be masked.
            Defaults to 0.40.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 out_indices=-1,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 mask_ratio=0.40,
                 init_cfg=None):
        super().__init__(
            arch=arch,
            img_size=img_size,
            patch_size=patch_size,
            out_indices=out_indices,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_cfg=norm_cfg,
            final_norm=final_norm,
            output_cls_token=output_cls_token,
            interpolate_mode=interpolate_mode,
            patch_cfg=patch_cfg,
            layer_cfgs=layer_cfgs,
            init_cfg=init_cfg)

        self.mask_ratio = mask_ratio
        self.mask_token = nn.parameter.Parameter(
            torch.zeros(1, 1, self.embed_dims))
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]

    def init_weights(self):
        super().init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):

            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.mask_token, std=.02)
            trunc_normal_(self.pos_embed, std=0.02)

            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        B = x.shape[0]
        x, _ = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        B, L, _ = x.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        mask = mask.flatten(1).unsqueeze(-1)
        x = x * (1 - mask.int()) + mask_tokens * mask

        # append cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return x[:, 1:]
