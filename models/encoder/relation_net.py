import logging

import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes

from models.model import register

logger = logging.getLogger(__name__)


@register('rn_encoder')
class RelationalNetworkEncoder(nn.Module):
    def __init__(self, input_feat_dim):
        super(RelationalNetworkEncoder, self).__init__()

        self.proj = nn.Conv2d(input_feat_dim, input_feat_dim // 2, kernel_size=1)

        # relational encoding
        self.g_mlp = nn.Sequential(
            nn.Linear(input_feat_dim, input_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(input_feat_dim // 2, input_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(input_feat_dim // 2, input_feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out_dim = input_feat_dim // 2

    def forward(self, im, boxes, normalized_boxes):
        del boxes
        del normalized_boxes
        # BxCxHxW
        x = self.proj(im)

        # relational encoding
        b, c, h, w = x.shape
        hw = h * w
        # bxhwxc
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (b * 1 * hw * c)
        x_i = x_i.repeat(1, hw, 1, 1)  # (b * hw * hw  * c)
        x_j = torch.unsqueeze(x_flat, 2)  # (b * hw * 1 * c)
        x_j = x_j.repeat(1, 1, hw, 1)  # (b * hw * hw  * c)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (b * hw * hw  * 2c)

        # reshape for passing through network
        x_full = x_full.view(b * hw * hw, -1)  # (b*hw*hw)*2c

        x_g = self.g_mlp(x_full)

        # reshape again and sum
        x_g = x_g.view(b, hw * hw, -1)

        x_g = x_g.sum(1)

        return x_g


@register('rn_bbox_encoder')
class RelationalBBoxNetworkEncoder(nn.Module):
    def __init__(self, input_feat_dim):
        super(RelationalBBoxNetworkEncoder, self).__init__()

        self.proj = nn.Conv2d(input_feat_dim, input_feat_dim // 2, kernel_size=1)

        # ROI Pooler
        self.roi_pooler = ROIPooler(
           output_size=7,
           scales=(1/32,), # TODO: this works for resnet50, swin*, nfnet
           sampling_ratio=0,
           pooler_type='ROIAlignV2',
        )
        self.roi_processor = nn.Sequential(
            nn.Conv2d(input_feat_dim // 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*7*7, 1024),
            nn.ReLU()
        )
        self.roi_processor_ln = nn.LayerNorm(1024)
        rn_in_planes = 1024 * 2

        # bbox coord encoding
        self.roi_processor_box = nn.Linear(4, 256)
        self.roi_processor_box_ln = nn.LayerNorm(256)
        rn_in_planes = (1024 + 256) * 2

        # relational encoding
        self.g_mlp = nn.Sequential(
            nn.Linear(rn_in_planes, rn_in_planes // 2),
            nn.ReLU(),
            nn.Linear(rn_in_planes // 2, rn_in_planes // 2),
            nn.ReLU(),
            nn.Linear(rn_in_planes // 2, rn_in_planes // 2),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out_dim = rn_in_planes // 2

    def process_single_image_rois(self, roi_feats):
        # relational encoding
        M, C = roi_feats.shape
        b = 1
        # 1xMxC
        x_flat = roi_feats.unsqueeze(0)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (b * 1 * M * c)
        x_i = x_i.repeat(1, M, 1, 1)  # (b * M * M  * c)
        x_j = torch.unsqueeze(x_flat, 2)  # (b * M * 1 * c)
        x_j = x_j.repeat(1, 1, M, 1)  # (b * M * M  * c)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3) # (b * M * M  * 2c)

        # reshape for passing through network
        x_full = x_full.view(b * M * M, -1)  # (b*M*M)*2c

        x_g = self.g_mlp(x_full)

        # reshape again and sum
        x_g = x_g.view(b, M * M, -1)

        x_g = x_g.sum(1)
        return x_g

    def forward(self, im_feat, boxes, normalized_boxes):
        # BxCxHxW
        x = im_feat

        if len(x.shape) != 4:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.proj(x)

        # RoI pooling/align
        all_boxes = []
        for boxes_i in boxes:
            # remove empty box
            boxes_i = Boxes(torch.stack(list(filter(lambda x: not torch.all(x == 0), boxes_i)), dim=0))
            all_boxes.append(boxes_i)
        num_boxes = [boxes_i.tensor.shape[0] for boxes_i in all_boxes]

        roi_feats = self.roi_pooler([x], all_boxes)
        roi_feats = self.roi_processor(roi_feats)
        roi_feats = self.roi_processor_ln(roi_feats)
        # Add bbox pos features
        all_norm_boxes = []
        for norm_boxes_i in normalized_boxes:
            all_norm_boxes.append(torch.stack(list(filter(lambda x: not torch.all(x == 0), norm_boxes_i)), dim=0))
        norm_bbox_tensor = torch.cat([box for box in all_norm_boxes]).to(roi_feats.device)
        norm_bbox_tensor = norm_bbox_tensor * 2 - 1
        roi_box_feats = self.roi_processor_box_ln(self.roi_processor_box(norm_bbox_tensor))
        roi_feats = torch.cat([roi_feats, roi_box_feats], dim=-1)

        feats_list = []
        start_idx = 0
        for num_boxes_i in num_boxes:
            end_idx = start_idx + num_boxes_i
            feats_i = self.process_single_image_rois(roi_feats[start_idx:end_idx])
            feats_list.append(feats_i)
            start_idx = end_idx
        assert end_idx == roi_feats.shape[0], '{} vs {}'.format(end_idx, roi_feats.shape[0])
        feats = torch.cat(feats_list, dim=0)
        # BxC
        return feats