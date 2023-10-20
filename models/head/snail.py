import logging
import math

import numpy as np
import torch
import torch.nn as nn
from einops import pack
from torch.autograd import Variable

from models.model import register
from utils import make_nk_label

logger = logging.getLogger(__name__)


@register('snail')
class SnailFewShot(nn.Module):
    def __init__(self, n_way, n_shot, input_feat_dim, dynamic_k=True):
        super().__init__()

        self.N = n_way
        self.K = n_shot
        num_channels = input_feat_dim + self.N
        self.dynamic_k = dynamic_k

        num_filters = int(math.ceil(math.log(self.N * self.K + 1, 2)))
        self.attention1 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.tc1 = TCBlock(num_channels, self.N * self.K + 1, 256)
        num_channels += num_filters * 256
        self.attention2 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.tc2 = TCBlock(num_channels, self.N * self.K + 1, 256)
        num_channels += num_filters * 256
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        num_channels += 256
        self.fc = nn.Linear(num_channels, self.N)

    def forward(self, x_shot_feat, x_query_feat, batch_size, n_way, n_shot, n_query):
        assert n_query == 1, "SNAIL only support 1 query sample"

        x_tot = pack([x_shot_feat, x_query_feat], 'b * d')[0] # [bs, n_way * n_shot + 1, n_feat]

        labels_support = make_nk_label(n_way, n_shot, batch_size)  # [bs * n_way * n_shot]
        labels_support = labels_support.to(x_tot.device).unsqueeze(-1) # [bs * n_way * n_shot, 1]
        labels_support_onehot = torch.FloatTensor(labels_support.size(0), 2).to(x_tot.device)

        labels_support_onehot.zero_()
        labels_support_onehot.scatter_(1, labels_support, 1)  # [bs * n_way * n_shot, n_way]
        labels_support_onehot = labels_support_onehot.view(batch_size, -1, n_way)
        labels_query_zero = torch.Tensor(np.zeros((batch_size, 1, n_way))).to(x_tot.device)
        labels = torch.cat([labels_support_onehot, labels_query_zero], dim=1)  # [bs, n_way * n_shot + 1, n_way]

        x = torch.cat((x_tot, labels), dim=-1)  # [bs, n_way * n_shot + 1, n_feat + n_way]
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)  # [bs, n_way * n_shot + 1, n_way]
        return x[:, -1, :]  # [bs, n_way]


class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation] # TODO: make this correct for different strides/padding


class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = torch.tanh(xf) * torch.sigmoid(xg) # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)


class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        self.dense_blocks = nn.ModuleList([DenseBlock(in_channels + i * filters, 2 ** (i+1), filters)
                                           for i in range(int(math.ceil(math.log(seq_length, 2))))])

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        for block in self.dense_blocks:
            input = block(input)
        return torch.transpose(input, 1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.BoolTensor(mask).to(input.device)

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = torch.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2) # shape: (N, T, in_channels + value_size)
