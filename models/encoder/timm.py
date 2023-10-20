import logging
import os

import timm
import torch
import torch.nn as nn
from einops import rearrange

from models.model import register

logger = logging.getLogger(__name__)


class _Wrapper(nn.Module):

    def __init__(self, model, tag):
        super().__init__()
        self.model = model
        self.tag = tag

    def forward(self, x):
        feat = self.model.forward_features(x)
        if 'swin' in self.tag:
            feat = rearrange(feat, 'b h w c -> b c h w')
        if 'vit_base_32_timm_laion2b' in self.tag or 'vit_base_32_timm_openai' in self.tag:
            # TODO: [CLS] is prepended to the patches.
            feat = rearrange(feat[:, 1:], 'b (h w) c -> b c h w', h=7)
        return feat

# 1024x7x7
@register('convnext_base_in1k')
def convnext_base_in1k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'convnext_base',
        pretrained=pretrained
    ), 'convnext_base_in1k')

# 1024x7x7
@register('convnext_base_in22k')
def convnext_base_in22k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'convnext_base_in22k',
        pretrained=pretrained
    ), 'convnext_base_in22k')

# 1024x7x7
@register('convnext_base_laion400m')
def convnext_base_in22k(pretrained=False, **kwargs):
    model = timm.create_model(
        'convnext_base',
        pretrained=False
    )
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(
            os.path.dirname(__file__),
            '../../assets/weights',
            'convnext_base_224_laion400m.pth'), map_location='cpu'), strict=False)
        logger.info('Pretrained LAION-400m convnext-base loaded.')
    return _Wrapper(model, 'convnext_base_laion400m')

# 1024x7x7
@register('convnext_base_laion2b')
def convnext_base_laion2b(pretrained=False, **kwargs):
    m = timm.create_model(
        'convnext_base.clip_laion2b',
        pretrained=pretrained
    )
    if kwargs.get('reset_clip_s2b2'):
        logger.info('Resetting the last conv layer of convnext-base to random init.')
        s = m.state_dict()
        for i in s.keys():
            if 'stages.3.blocks.2' in i and ('weight' in i or 'bias' in i):
                s[i].normal_()
        m.load_state_dict(s, strict=True)

    return _Wrapper(m, 'convnext_base_laion2b')

# 7x7x1024
@register('swin_base_timm_in1k')
def swin_base_in1k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'swin_base_patch4_window7_224',
        pretrained=pretrained
    ), 'swin_base_timm_in1k')

# 7x7x1024
@register('swin_base_timm_in22k')
def swin_base_in22k(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'swin_base_patch4_window7_224_in22k',
        pretrained=pretrained
    ), 'swin_base_timm_in22k')

# 50x768
@register('vit_base_32_timm_laion2b')
def vit_b_32_laion2b(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'vit_base_patch32_clip_224.laion2b',
        pretrained=pretrained
    ), 'vit_base_32_timm_laion2b')

# 50x768
@register('vit_base_32_timm_openai')
def vit_b_32_openai(pretrained=False, **kwargs):
    return _Wrapper(timm.create_model(
        'vit_base_patch32_clip_224.openai',
        pretrained=pretrained
    ), 'vit_base_32_timm_openai')

# # 3072x11x15
# @register('nfnet_f0')
# def nfnet_f0(pretrained=False, **kwargs):
#     return timm.create_model(
#         'dm_nfnet_f0',
#         pretrained=pretrained
#     )

# # 3072x11x15
# @register('nfnet_f1')
# def nfnet_f1(pretrained=False, **kwargs):
#     return timm.create_model(
#         'dm_nfnet_f1',
#         pretrained=pretrained
#     )
