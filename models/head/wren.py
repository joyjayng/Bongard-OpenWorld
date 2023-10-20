import logging

import torch
import torch.nn as nn

from models.model import register

logger = logging.getLogger(__name__)


@register('wren')
class WReN(nn.Module):

    def __init__(self, n_shot, input_feat_dim, method='original'):
        super().__init__()
        self.n_shot = n_shot
        self.feat_len = input_feat_dim
        self.method = method

        self.mlp_g = nn.Sequential(nn.Linear(self.feat_len * 2, self.feat_len),
                                   nn.ReLU(),
                                   nn.Linear(self.feat_len, self.feat_len),
                                   nn.ReLU())

        if self.method == 'original':
            self.mlp_f = nn.Sequential(nn.Linear(self.feat_len, self.feat_len),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Linear(self.feat_len, 1))
        elif self.method == 'modified':
            self.mlp_f = nn.Sequential(nn.Linear(self.feat_len * 2, self.feat_len),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Linear(self.feat_len, 2))
        else:
            raise  Exception('method should be in [original, modified]')

        logging.info('wren, {}'.format(method))

    def forward(self, x_shot_feat, x_query_feat, batch_size, n_way, n_shot, n_query):
        shot_shape = [batch_size, n_way, n_shot]
        query_shape = [batch_size, n_query]

        x_shot_feat = x_shot_feat.view(*shot_shape, -1)  # [ep_per_batch, n_way, n_shot, feat_len]
        x_shot_pos, x_shot_neg = x_shot_feat[:, 0, :, :], x_shot_feat[:, 1, :, :]  # [ep_per_batch, n_way, n_shot, feat_len]
        x_query_feat = x_query_feat.view(*query_shape, -1)  # [ep_per_batch, n_way * n_query, feat_len]

        num_query_samples = x_query_feat.size(1)  # n_query
        logits = []
        for i in range(num_query_samples):
            x_query_single = x_query_feat[:, i:i + 1]  # [ep_per_batch, 1, feat_len]

            # positive
            emb_pos = torch.cat([x_shot_pos, x_query_single], dim=1)  # [ep_per_batch, n_shot + 1, feat_len]
            emb_pairs_pos = self.group_embeddings_batch(emb_pos)  # [ep_per_batch, num_pairs, 2 * feat_len]

            # negative
            emb_neg = torch.cat([x_shot_neg, x_query_single], dim=1)  # [ep_per_batch, n_shot + 1, feat_len]
            emb_pairs_neg = self.group_embeddings_batch(emb_neg)  # [ep_per_batch, num_pairs, 2 * feat_len]

            # relation
            emb_pairs = torch.cat([emb_pairs_pos, emb_pairs_neg], dim=1)  # [ep_per_batch, 2 * num_pairs, 2 * feat_len]
            emb_rels = self.mlp_g(emb_pairs.view(-1, self.feat_len * 2))  # [ep_per_batch * 2 * num_pairs, feat_len]
            emb_rels = emb_rels.view(batch_size * n_way, -1, self.feat_len)  # [ep_per_batch * 2, num_pairs, feat_len]
            if self.method == 'original':
                logit = self.mlp_f(torch.sum(emb_rels, dim=1))  # [ep_per_batch * 2, 1]
            elif self.method == 'modified':
                emb_sums = torch.sum(emb_rels, dim=1).view(batch_size, -1)  # [ep_per_batch, 2 * feat_len]
                logit = self.mlp_f(emb_sums)  # [ep_per_batch, 2]
            else:
                logit = None

            logit = logit.view(-1, n_way)  # [ep_per_batch, n_way]
            logits.append(logit.unsqueeze(1))

        logits = torch.cat(logits, dim=1)  # [ep_per_batch, n_query, n_way]

        return logits

    def group_embeddings_batch(self, embeddings):
        num_emb = self.n_shot + 1
        embeddings = embeddings.view(-1, num_emb, self.feat_len)

        emb_pairs = torch.cat(
            [embeddings.unsqueeze(1).expand(-1, num_emb, -1, -1),
             embeddings.unsqueeze(2).expand(-1, -1, num_emb, -1)],
            dim=-1).view(-1, num_emb ** 2, 2 * self.feat_len)

        use_indices = torch.tensor([i * num_emb + j for i in range(num_emb)
                                    for j in range(num_emb) if i != j])
        emb_pairs = emb_pairs[:, use_indices]  # [bs, num_pairs, 2 * feat_len]

        return emb_pairs
