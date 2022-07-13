# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from einops import rearrange
from mmcls.models import LabelSmoothLoss
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule
from skimage.feature import hog
from torch import nn

from ..builder import HEADS


@HEADS.register_module()
class MaskFeatPretrainHead(BaseModule):
    """Pre-training head for MaskFeat.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
            Defaults to 768.
        hog_dim (int): The dim of the hog feature. Defaults to 108.
    """

    def __init__(self, embed_dim=768, hog_dim=108):
        super(MaskFeatPretrainHead, self).__init__()
        self.head = nn.Linear(embed_dim, hog_dim)

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def extract_hog4gray(self,
                         image,
                         orientations=9,
                         pixels_per_cell=(8, 8),
                         cells_per_block=(1, 1),
                         block_norm='L2'):
        """Get the hog features for ecah gray images."""
        hog_feature = hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            feature_vector=False)
        return hog_feature

    def extract_hog4RGB(self,
                        image,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1),
                        block_norm='L2'):
        """Get the hog features for ecah RGB images."""
        hog_features_r = self.extract_hog4gray(image[:, :, 0], orientations,
                                               pixels_per_cell,
                                               cells_per_block, block_norm)
        hog_features_g = self.extract_hog4gray(image[:, :, 1], orientations,
                                               pixels_per_cell,
                                               cells_per_block, block_norm)
        hog_features_b = self.extract_hog4gray(image[:, :, 2], orientations,
                                               pixels_per_cell,
                                               cells_per_block, block_norm)
        hog_features = np.concatenate(
            [hog_features_r, hog_features_g, hog_features_b], axis=-1)
        hog_features = rearrange(
            hog_features,
            '(ph dh) (pw dw) ch cw c -> ph pw (dh dw ch cw c)',
            ph=14,
            pw=14)
        return hog_features

    def extract_hog_for_batch(self, img):
        """Get the hog features for batch images."""
        img = img.permute(0, 2, 3, 1)
        hog_feature = np.stack(list(map(self.extract_hog4RGB, img)), axis=0)
        return torch.Tensor(np.transpose(hog_feature, (0, 3, 1, 2)))

    def loss(self, pred, target, mask):
        """Compute the loss."""
        N, C, H, W = target.shape
        losses = dict()

        if pred.is_cuda:
            target = target.cuda(pred.device)

        target = target.permute(0, 2, 3, 1).reshape(N, H * W, C)
        loss = (pred - target)**2
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum()
        losses['loss'] = loss
        return losses

    def forward(self, img, latent, mask):
        hog_features = self.extract_hog_for_batch(img)
        latent = self.head(latent)
        losses = self.loss(latent, hog_features, mask)

        return losses


@HEADS.register_module()
class MaskFeatFinetuneHead(BaseModule):
    """Fine-tuning head for MaskFeat.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000, label_smooth_val=0.1):
        super(MaskFeatFinetuneHead, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.criterion = LabelSmoothLoss(label_smooth_val, num_classes)

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        """"Get the logits."""
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses
