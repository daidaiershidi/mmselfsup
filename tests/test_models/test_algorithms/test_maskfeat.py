# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.algorithms import MaskFeat

backbone=dict(
    type='MaskFeatViT',
    arch='b',
    patch_size=16,
    drop_path_rate=0,
)
head=dict(type='MaskFeatPretrainHead', hog_dim=108)
hog_para=dict(nbins=9, pool=8, gaussian_window=16)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae():
    with pytest.raises(AssertionError):
        alg = MaskFeat(backbone=backbone, head=None, hog_para=hog_para)
    with pytest.raises(AssertionError):
        alg = MaskFeat(backbone=None, head=head, hog_para=hog_para)
    alg = MaskFeat(backbone=backbone, head=head, hog_para=hog_para)

    fake_input = torch.randn((2, 3, 224, 224))
    fake_mask = torch.randn((2, 14, 14)).bool()
    fake_loss = alg.forward_train((fake_input, fake_mask))
    fake_feature = alg.extract_feat(fake_input, fake_mask)
    
    torch.save(fake_input, '/mnt/lustre/liukaiyuan.vendor/duiqi/pre-train/fake_input')
    torch.save(fake_mask, '/mnt/lustre/liukaiyuan.vendor/duiqi/pre-train/fake_mask')
    torch.save(fake_loss, '/mnt/lustre/liukaiyuan.vendor/duiqi/pre-train/fake_loss')
    torch.save(fake_feature, '/mnt/lustre/liukaiyuan.vendor/duiqi/pre-train/fake_feature')
    torch.save(alg.state_dict(), '/mnt/lustre/liukaiyuan.vendor/duiqi/pre-train/alg')
    
    assert isinstance(fake_loss['loss'].item(), float)
    assert list(fake_feature.shape) == [2, 197, 768]
