import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc_refinedet, coco_refinedet
import os

import numpy as np
from itertools import product as product
from functools import partial
from six.moves import map, zip
from math import sqrt as sqrt
import cv2
# mmd
from mmcv.ops import DeformConv2d
from mmcv.cnn import normal_init, kaiming_init, constant_init, xavier_init, bias_init_with_prob, ConvModule

class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, ARM, ODM, TCB, num_classes, seg_num_grids=[36, 24, 16, 12], bn=True):
        super(RefineDet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes == 21][str(size)]
        self.size = size
        self.bn = bn
        if size != 512 and size != 320:
            self.conv3_3_layer = (16, 23)[self.bn]
        self.conv4_3_layer = (23, 33)[self.bn]
        self.conv5_3_layer = (30, 43)[self.bn]
        self.extra_1_layer = (4, 6)[self.bn]
        if size == 640 or size == 5126:
            self.extra_2_layer = (8, 12)[self.bn]

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        if size != 512 and size != 320:
            self.conv3_3_L2Norm = L2Norm(256, 10)
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)

        # self.arm_loc = nn.ModuleList(ARM[0])
        # self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
    

        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])
        self.step = len(self.cfg['feature_maps']) - 1

        # attention head
        self.lvl_num = len(self.cfg['feature_maps'])
        self.seg_num_grids = seg_num_grids
        self.in_channels = arm[str(size)]
        self.seg_feat_channels = 256
        self.stacked_convs = 1
        self._init_solo_layers()

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
    
    # modified from https://github.com/WXinlong/SOLO/blob/master/mmdet/models/anchor_heads/solo_head.py
    def _init_solo_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        head_num = len(self.in_channels)
        cate_convs = []
        grid_cate = []
        for i in range(head_num):
            first_conv =ConvModule(
                self.in_channels[i],
                self.seg_feat_channels,
                3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                bias=norm_cfg is None)
            mid_conv = ConvModule(
                self.seg_feat_channels,
                self.seg_feat_channels,
                3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                bias=norm_cfg is None)
            mid_layers = [mid_conv for _ in range(self.stacked_convs - 1)]
            last_conv = nn.Conv2d(self.seg_feat_channels, self.num_classes - 1, 3, padding=1)
            cate_convs += [first_conv] + mid_layers
            grid_cate += [last_conv]
        self.cate_convs = nn.ModuleList(cate_convs)
        self.grid_cate = nn.ModuleList(grid_cate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, img_id=None, img_gt=None):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        tcb_source = list()
        # arm_loc = list()
        # arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        if self.phase == 'test':
            feat_sizes = list()

        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(self.conv5_3_layer):
            x = self.vgg[k](x)
            if self.size != 512 and self.size != 320 and self.conv3_3_layer - 1 == k:
                s = self.conv3_3_L2Norm(x)
                sources.append(s)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])
            if self.conv4_3_layer - 1 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])
            elif self.conv5_3_layer - 1 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])

        # apply vgg up to fc7
        for k in range(self.conv5_3_layer, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        if self.phase == 'test':
            feat_sizes.append(x.shape[2:])

        # apply extra layers and cache source layer outputs
        for k in range(len(self.extras)):
            x = self.extras[k](x)
            if self.extra_1_layer - 1 == k:
                sources.append(x)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])
            if (self.size == 640 or self.size == 5126) and self.extra_2_layer - 1 == k:
                sources.append(x)
                if self.phase == 'test':
                    feat_sizes.append(x.shape[2:])

        # apply attention head
        attention_sources = list()
        attention_maps = list()
        for (x, lvl, cate) in zip(sources, range(self.lvl_num), self.grid_cate):
            for conv in self.cate_convs[lvl*self.stacked_convs: (lvl+1)*self.stacked_convs]:
                x = conv(x)
            cate_feat = cate(x)
            attention_maps.append(cate_feat)

            cate_pred = self.sigmoid(cate_feat)
            attention_sources.append(cate_pred)

        ## https://blog.csdn.net/weixin_41735859/article/details/106474768
        if self.phase == 'test' and img_id is not None:
            save_dir = './eval/attention_maps'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for index, level in enumerate(attention_sources):
                i = self.cfg['steps'][index]
                level = F.interpolate(level, size=(self.size, self.size), mode='bilinear') # bilinear, bicubic,nearest
                level = level.squeeze(0)
                level = level.cpu().numpy().copy()
                level = np.transpose(level, (1, 2, 0))
                # plt.imsave(os.path.join(save_dir, str(img_id) + '_' + str(i) + '.png'), level[:,:,0])#, cmap='gray')
                
                cam = np.maximum(level, 0)
                cam -= np.min(cam)
                cam = cam / cam.max()
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                cam_img = 0.5 * heatmap + 0.5 * img_gt
                save_path = os.path.join(save_dir, str(img_id) + '_' + str(i) + '.png')
                # cv2.imwrite(save_path, heatmap)
                cv2.imwrite(save_path, cam_img)

        # calculate TCB features
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(self.step-k)*3 + i](s)
            if k != 0:
                u = p
                u = self.tcb1[self.step-k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(self.step-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        tcb_source.reverse()

        # apply attention
        tcb_source_new = list()
        for attention, tcbx in zip(attention_sources, tcb_source):
            feature = tcbx * torch.exp(attention)
            tcb_source_new.append(feature)

        # apply alignconv to source layers
        for (x, l, c) in zip(tcb_source_new, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        if self.phase == "test":
            output = (
                odm_loc.view(odm_loc.size(0), -1, 4),           # odm loc preds
                self.softmax(odm_conf.view(odm_conf.size(0), -1,
                             self.num_classes)),                # odm conf preds
                feat_sizes
            )
        else:
            output = (
                attention_maps,
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
            )
        return output


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            vgg_weights = torch.load(pretrained)
            print('Loading base network...')
            self.vgg.load_state_dict(vgg_weights)
        elif pretrained is None:
            for m in self.vgg.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
        else:
            raise TypeError('pretrained must be a str or None')
        # initialize newly added layers' weights with xavier method
        self.extras.apply(init_method)
        self.odm_loc.apply(init_method)
        self.odm_conf.apply(init_method)
        self.tcb0.apply(init_method)
        self.tcb1.apply(init_method)
        self.tcb2.apply(init_method)
        # initialize attention head
        for m in self.cate_convs:
            normal_init(m.conv, std=0.01)
        bias_cate = bias_init_with_prob(0.01)
        for cate in self.grid_cate:
            normal_init(cate, std=0.01, bias=bias_cate)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def init_method(m):
    if isinstance(m, nn.Conv2d):
        xavier_init(m, distribution='uniform', bias=0)
    elif isinstance(m, nn.ConvTranspose2d):
        xavier_init(m, distribution='uniform', bias=0)
    elif isinstance(m, nn.BatchNorm2d):
        constant_init(m, 1)


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(conv6.out_channels),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(conv7.out_channels), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                conv2d = nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[k + 1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers

def arm_multibox(in_channels, anchor_nums):
    arm_loc_layers = []
    arm_conf_layers = []
    for in_channel, anchor_num in zip(in_channels, anchor_nums):
        arm_loc_layers += [nn.Conv2d(in_channel, anchor_num * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(in_channel, anchor_num * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)

def adm_multibox(level_channels, anchor_nums, num_classes):
    assert set(anchor_nums) == {3}
    adm_loc_layers1 = []
    adm_loc_layers2 = []
    adm_loc_layers3 = []
    adm_conf_layers1 = []
    adm_conf_layers2 = []
    adm_conf_layers3 = []
    for _ in level_channels:
            adm_loc_layers1 += [DeformConv2d(256, 4, kernel_size=3, padding=1)]
            adm_loc_layers2 += [DeformConv2d(256, 4, kernel_size=3, padding=1)]
            adm_loc_layers3 += [DeformConv2d(256, 4, kernel_size=3, padding=1)]
            adm_conf_layers1 += [DeformConv2d(256, num_classes, kernel_size=3, padding=1)]
            adm_conf_layers2 += [DeformConv2d(256, num_classes, kernel_size=3, padding=1)]
            adm_conf_layers3 += [DeformConv2d(256, num_classes, kernel_size=3, padding=1)]
    return (
        (adm_loc_layers1, adm_loc_layers2, adm_loc_layers3), 
        (adm_conf_layers1, adm_conf_layers2, adm_conf_layers3))

def odm_multibox(level_channels, anchor_nums, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    for i in range(len(level_channels)):
        odm_loc_layers += [nn.Conv2d(256, anchor_nums[i] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, anchor_nums[i] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)

base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '5125': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '5126': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '640': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],
    '5125': [256, 'S', 512],
    '5126': [256, 'S', 512, 128, 'S', 256],
    '640': [256, 'S', 512, 128, 'S', 256],
}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],  # number of boxes per feature map location
    '5125': [3, 3, 3, 3, 3],  # number of boxes per feature map location
    '5126': [3, 3, 3, 3, 3, 3],  # number of boxes per feature map location
    '640': [3, 3, 3, 3, 3, 3],  # number of boxes per feature map location
}

tcb = {
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],
    '5125': [256, 512, 512, 1024, 512],
    '5126': [256, 512, 512, 1024, 512, 256],
    '640': [256, 512, 512, 1024, 512, 256],
}

arm = {
    '512': [512, 512, 1024, 512],
    '5125': [256, 512, 512, 1024, 512],
    '5126': [256, 512, 512, 1024, 512, 256],
    '640': [256, 512, 512, 1024, 512, 256],
}

def build_refinedet(phase, size=320, num_classes=21, seg_num_grids=[36, 24, 16, 12], backbone_dict=dict(bn=True)):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    bn = backbone_dict['bn']
    base_ = vgg(base[str(size)], 3, bn)
    extras_ = add_extras(extras[str(size)], 1024, bn)
    ARM_ = arm_multibox(arm[str(size)], mbox[str(size)])
    # ADM_ = adm_multibox(arm[str(size)], mbox[str(size)], num_classes)
    ODM_ = odm_multibox(arm[str(size)], mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(phase, size, base_, extras_, ARM_, ODM_, TCB_, num_classes, seg_num_grids, bn)