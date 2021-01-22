# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp

from functools import partial
from six.moves import map, zip

# solo / mmd 
from mmdet.models.builder import build_loss

def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class AttentionFocalLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, input_size, anchor_mark, loss_cate, seg_num_grids, scale_ranges):
        super(AttentionFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.cate_out_channels = num_classes - 1
        self.input_size = input_size
        self.anchor_mark = anchor_mark
        self.loss_cate = build_loss(loss_cate)
        self.seg_num_grids = seg_num_grids
        self.scale_ranges = scale_ranges
        self.sigma = 0.2

    def forward(self, cate_preds, targets, best_truth_idx_t):
        gt_bbox_list  = [targets[i][:, :-1] * self.input_size for i in range(len(targets))]
        gt_label_list = [targets[i][:, -1] for i in range(len(targets))]
        cate_label_list = multi_apply(
            self.solo_target_single,
            gt_bbox_list,
            gt_label_list,
            best_truth_idx_t)
        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*cate_label_list)
        ]
        flatten_cate_labels = torch.cat(cate_labels)
        num_pos = (flatten_cate_labels > 0).sum()

        cate_preds = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
            for cate_pred in cate_preds
        ]
        flatten_cate_preds = torch.cat(cate_preds)

        loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_pos + 1)
        return loss_cate

    def solo_target_single(self, gt_bboxes_raw, gt_labels_raw, best_truth_idx_raw):

        device = gt_labels_raw[0].device

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        cate_label_list = []
        level_num = len(self.anchor_mark)
        # for idx, (lower_bound, upper_bound), num_grid \
        #         in zip(range(level_num), self.scale_ranges, self.seg_num_grids):
        for idx, num_grid \
                in zip(range(level_num), self.seg_num_grids):

            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            # hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()

            if idx == 0:
                begin = 0
            else:
                begin = self.anchor_mark[idx - 1] + 1
            end = self.anchor_mark[idx]
            if end == self.anchor_mark[-1]:
                best_truth_idx = best_truth_idx_raw[begin: ]
            else:
                best_truth_idx = best_truth_idx_raw[begin: end + 1]
            positive_index = (best_truth_idx > -1).nonzero()
            best_truth_idx = best_truth_idx[positive_index]
            tmp = list(set(best_truth_idx.flatten().cpu().numpy().tolist()))
            hit_indices = torch.tensor(tmp, dtype=torch.int64, device=device)

            if len(hit_indices) == 0:
                cate_label_list.append(cate_label)
                continue
            gt_bboxes = gt_bboxes_raw[hit_indices]
            gt_labels = gt_labels_raw[hit_indices]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # mass center
            center_ws = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) / 2
            center_hs = (gt_bboxes[:, 3] + gt_bboxes[:, 1]) / 2

            for gt_label, half_h, half_w, center_h, center_w in zip(gt_labels, half_hs, half_ws, center_hs, center_ws):
                coord_w = int((center_w / self.input_size) // (1. / num_grid))
                coord_h = int((center_h / self.input_size) // (1. / num_grid))

                # left, top, right, down
                top_box = max(0, int(((center_h - half_h) / self.input_size) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / self.input_size) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / self.input_size) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / self.input_size) // (1. / num_grid)))

                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                cate_label[top:(down+1), left:(right+1)] = gt_label
            cate_label_list.append(cate_label)
        return cate_label_list
    
    # def forward(self, cate_preds, targets):
    #     gt_bbox_list  = [targets[i][:, :-1] * self.input_size for i in range(len(targets))]
    #     gt_label_list = [targets[i][:, -1] for i in range(len(targets))]
    #     cate_label_list = multi_apply(
    #         self.solo_target_single,
    #         gt_bbox_list,
    #         gt_label_list)
    #     # cate
    #     cate_labels = [
    #         torch.cat([cate_labels_level_img.flatten()
    #                    for cate_labels_level_img in cate_labels_level])
    #         for cate_labels_level in zip(*cate_label_list)
    #     ]
    #     flatten_cate_labels = torch.cat(cate_labels)
    #     num_pos = (flatten_cate_labels > 0).sum()

    #     cate_preds = [
    #         cate_pred.permute(0, 2, 3, 1).reshape(-1, self.cate_out_channels)
    #         for cate_pred in cate_preds
    #     ]
    #     flatten_cate_preds = torch.cat(cate_preds)

    #     loss_cate = self.loss_cate(flatten_cate_preds, flatten_cate_labels, avg_factor=num_pos + 1)
    #     return loss_cate

    # def solo_target_single(self, gt_bboxes_raw, gt_labels_raw):

    #     device = gt_labels_raw[0].device

    #     gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
    #             gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

    #     cate_label_list = []
    #     for (lower_bound, upper_bound), num_grid \
    #             in zip(self.scale_ranges, self.seg_num_grids):

    #         cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)

    #         hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
    #         if len(hit_indices) == 0:
    #             cate_label_list.append(cate_label)
    #             continue
    #         gt_bboxes = gt_bboxes_raw[hit_indices]
    #         gt_labels = gt_labels_raw[hit_indices]

    #         half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
    #         half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

    #         # mass center
    #         center_ws = (gt_bboxes[:, 2] + gt_bboxes[:, 0]) / 2
    #         center_hs = (gt_bboxes[:, 3] + gt_bboxes[:, 1]) / 2

    #         for gt_label, half_h, half_w, center_h, center_w in zip(gt_labels, half_hs, half_ws, center_hs, center_ws):
    #             coord_w = int((center_w / self.input_size) // (1. / num_grid))
    #             coord_h = int((center_h / self.input_size) // (1. / num_grid))

    #             # left, top, right, down
    #             top_box = max(0, int(((center_h - half_h) / self.input_size) // (1. / num_grid)))
    #             down_box = min(num_grid - 1, int(((center_h + half_h) / self.input_size) // (1. / num_grid)))
    #             left_box = max(0, int(((center_w - half_w) / self.input_size) // (1. / num_grid)))
    #             right_box = min(num_grid - 1, int(((center_w + half_w) / self.input_size) // (1. / num_grid)))

    #             top = max(top_box, coord_h-1)
    #             down = min(down_box, coord_h+1)
    #             left = max(coord_w-1, left_box)
    #             right = min(right_box, coord_w+1)

    #             cate_label[top:(down+1), left:(right+1)] = gt_label
    #         cate_label_list.append(cate_label)
    #     return cate_label_list