import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from models.model import register

logger = logging.getLogger(__name__)


@register('meta_baseline')
class MetaBaseline(nn.Module):

    def __init__(self, method='cos', temp=10., temp_learnable=True):
        super().__init__()
        self.method = method

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, x_shot_feat, x_query_feat, batch_size, n_way, n_shot, n_query):
        shot_shape = [batch_size, n_way, n_shot]
        query_shape = [batch_size, n_query]

        x_shot = x_shot_feat.view(*shot_shape, -1)
        x_query = x_query_feat.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)  # [ep_per_batch, way, feature_len]
            x_query = F.normalize(x_query, dim=-1)  # [ep_per_batch, query, feature_len]
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)  # [ep_per_batch, query, way]
        return logits
