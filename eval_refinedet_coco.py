"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import COCOroot, MEANS, COCODetection
import torch.utils.data as data

from layers import Detect_RefineDet
from utils.nms_wrapper import nms, soft_nms
from data import coco_refinedet
from layers import PriorBox
from plot_curve import plot_map, plot_loss

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import json


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth', type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='File path to save results')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
# parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool, help='Cleanup and remove results files following eval')
parser.add_argument('--input_size', default='512', choices=['320', '512'], type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--retest', default=False, type=bool, help='test cache results')
parser.add_argument('--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--prefix', default='weights/lr_5e4', type=str, help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-mstest', '--multi_scale_test', default=False, type=str2bool, help='multi scale test')
parser.add_argument('--model', default='512_ResNet_101', type=str, help='model name')
parser.add_argument('-mo', '--maxout', action="store_true", default=False, help='use maxout for the first detection layer')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def vis_detection(im, target, cls_dets, save_folder, target_size, i):
    h, w, _ = im.shape
    xr = target_size / w
    yr = target_size / h
    im_gt = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    for b in target:
        b[0] *= xr
        b[2] *= xr
        b[1] *= yr
        b[3] *= yr
        b = list(map(int, b))
        cv2.rectangle(im_gt, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        # cx = b[2]
        # cy = b[1]
        # text = "ship"
        # cv2.putText(im_gt, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
    boxes = cls_dets.copy()
    for b in boxes:
        b[0] *= xr
        b[2] *= xr
        b[1] *= yr
        b[3] *= yr
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        cv2.rectangle(im_gt, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[2]
        cy = b[1] + 12
        # text = "{:.2f}".format(b[4])
        # cv2.putText(im_gt, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
    # cv2.imshow('res', im_gt)
    # cv2.waitKey(0)
    save_gt_dir = os.path.join(save_folder, 'gt_im')
    if not os.path.exists(save_gt_dir):
        os.mkdir(save_gt_dir)
    cv2.imwrite(save_gt_dir + f'/{i}.png',im_gt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def im_detect(net, im, target_size):
    device = net.arm_conf[0].weight.device
    h, w, _ = im.shape
    scale = torch.Tensor([w, h, w, h])
    scale = scale.to(device)
    im_orig = im.astype(np.float32, copy=True)
    im = cv2.resize(im_orig, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    x = (im - MEANS).astype(np.float32)
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device)

    arm_loc, arm_conf, adm_loc, adm_conf, feat_sizes = net(x)
    priorbox = PriorBox(net.cfg, feat_sizes, (target_size, target_size), phase='test')
    priors = priorbox.forward()
    priors = priors.to(device)
    det = detect.forward(arm_loc, arm_conf, adm_loc, adm_conf, priors, scale)
    return det


def im_detect_ratio(net, im, target_size1, target_size2):
    device = net.arm_conf[0].weight.device
    h, w, _ = im.shape
    scale = torch.Tensor([w, h, w, h])
    scale = scale.to(device)
    im_orig = im.astype(np.float32, copy=True)
    if im_orig.shape[0] < im_orig.shape[1]:
        target_size1, target_size2 = target_size2, target_size1
    im = cv2.resize(im_orig, None, None, fx=float(target_size2)/float(w), fy=float(target_size1)/float(h), interpolation=cv2.INTER_LINEAR)
    x = (im - MEANS).astype(np.float32)
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device)

    arm_loc, arm_conf, adm_loc, adm_conf, feat_sizes = net(x)
    priorbox = PriorBox(net.cfg, feat_sizes, (target_size1, target_size2), phase='test')
    priors = priorbox.forward()
    priors = priors.to(device)
    det = detect.forward(arm_loc, arm_conf, adm_loc, adm_conf, priors, scale)
    return det
    

def flip_im_detect(net, im, target_size):
    im_f = cv2.flip(im, 1)
    det_f = im_detect(net, im_f, target_size)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = im.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = im.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    det_t[:, 5] = det_f[:, 5]

    return det_t


def flip_im_detect_ratio(net, im, target_size1, target_size2):
    im_f = cv2.flip(im, 1)
    det_f = im_detect_ratio(net, im_f, target_size1, target_size2)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = im.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = im.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]
    det_t[:, 4] = det_f[:, 4]
    det_t[:, 5] = det_f[:, 5]

    return det_t


def bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets


def soft_bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= args.confidence_threshold)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((soft_det_accu, det_accu_sum))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets


def multi_scale_test_net(target_size, save_folder, net, num_classes, dataset, detect, AP_stats=None):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    det_file = os.path.join(save_folder, 'detections.pkl')
    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        dataset.evaluate_detections(all_boxes, save_folder)
        return

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    
    for i in range(num_images):
        im, target = dataset.pull_image(i)
        _t['im_detect'].tic()
        # ori and flip
        det0 = im_detect(net, im, target_size)
        det0_f = flip_im_detect(net, im, target_size)
        det0 = np.row_stack((det0, det0_f))

        det_r = im_detect_ratio(net, im, target_size, int(0.75*target_size))
        det_r_f = flip_im_detect_ratio(net, im, target_size, int(0.75*target_size))
        det_r = np.row_stack((det_r, det_r_f))

        # shrink: only detect big object
        det_s1 = im_detect(net, im, int(0.5*target_size))
        det_s1_f = flip_im_detect(net, im, int(0.5*target_size))
        det_s1 = np.row_stack((det_s1, det_s1_f))

        det_s2 = im_detect(net, im, int(0.75*target_size))
        det_s2_f = flip_im_detect(net, im, int(0.75*target_size))
        det_s2 = np.row_stack((det_s2, det_s2_f))

        # #enlarge: only detect small object
        det3 = im_detect(net, im, int(1.75*target_size))
        det3_f = flip_im_detect(net, im, int(1.75*target_size))
        det3 = np.row_stack((det3, det3_f))
        index = np.where(np.minimum(det3[:, 2] - det3[:, 0] + 1, det3[:, 3] - det3[:, 1] + 1) < 128)[0]
        det3 = det3[index, :]

        det4 = im_detect(net, im, int(1.5*target_size))
        det4_f = flip_im_detect(net, im, int(1.5*target_size))
        det4 = np.row_stack((det4, det4_f))
        index = np.where(np.minimum(det4[:, 2] - det4[:, 0] + 1, det4[:, 3] - det4[:, 1] + 1) < 192)[0]
        det4 = det4[index, :]

        # More scales make coco get better performance
        if 'coco' in dataset.name:
            det5 = im_detect(net, im, int(1.25*target_size))
            det5_f = flip_im_detect(net, im, int(1.25*target_size))
            det5 = np.row_stack((det5, det5_f))
            index = np.where(np.minimum(det5[:, 2] - det5[:, 0] + 1, det5[:, 3] - det5[:, 1] + 1) < 224)[0]
            det5 = det5[index, :]

            det6 = im_detect(net, im, int(2*target_size))
            det6_f = flip_im_detect(net, im, int(2*target_size))
            det6 = np.row_stack((det6, det6_f))
            index = np.where(np.minimum(det6[:, 2] - det6[:, 0] + 1, det6[:, 3] - det6[:, 1] + 1) < 96)[0]
            det6 = det6[index, :]

            det7 = im_detect(net, im, int(2.25*target_size))
            det7_f = flip_im_detect(net, im, int(2.25*target_size))
            det7 = np.row_stack((det7, det7_f))
            index = np.where(np.minimum(det7[:, 2] - det7[:, 0] + 1, det7[:, 3] - det7[:, 1] + 1) < 64)[0]
            det7 = det7[index, :]
            det = np.row_stack((det0, det_r, det_s1, det_s2, det3, det4, det5, det6, det7))
        else:
            det = np.row_stack((det0, det_r, det_s1, det_s2, det3, det4))
        _t['im_detect'].toc()

        for j in range(1, num_classes):
            inds = np.where(det[:, -1] == j)[0]
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                if 'coco' in dataset.name:
                    cls_dets = soft_bbox_vote(cls_dets)
                elif 'sar' in dataset.name:
                    cls_dets = soft_bbox_vote(cls_dets)
                else:
                    cls_dets = bbox_vote(cls_dets)
                all_boxes[j][i] = cls_dets
                if args.show_image:
                    vis_detection(im, target, cls_dets, save_folder, target_size, i)

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('\nFPS: {} {} \n'.format(1 / (_t['im_detect'].average_time), 1 / _t['im_detect'].average_time))
    print('Evaluating detections')
    stats = dataset.evaluate_detections(all_boxes, save_folder)
    AP_stats['ap'].append(stats[0])
    AP_stats['ap50'].append(stats[1])
    AP_stats['ap75'].append(stats[2])
    AP_stats['ap_small'].append(stats[3])
    AP_stats['ap_medium'].append(stats[4])
    AP_stats['ap_large'].append(stats[5])


def single_scale_test_net(target_size, save_folder, net, num_classes, dataset, detect, AP_stats=None):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    det_file = os.path.join(save_folder, 'detections.pkl')
    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        dataset.evaluate_detections(all_boxes, save_folder)
        return

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    
    for i in range(num_images):
        im, target = dataset.pull_image(i)
        _t['im_detect'].tic()
        det = im_detect(net, im, target_size)
        _t['im_detect'].toc()

        for j in range(1, num_classes):
            inds = np.where(det[:, -1] == j)[0]
            if inds.shape[0] > 0:
                cls_dets = det[inds, :-1].astype(np.float32)
                if dataset.name == 'coco2017':
                    keep = soft_nms(cls_dets, sigma=0.5, Nt=0.30, threshold=args.confidence_threshold, method=1)
                    cls_dets = cls_dets[keep, :]
                all_boxes[j][i] = cls_dets
                if args.show_image:
                    vis_detection(im, target, cls_dets, save_folder, target_size, i)

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('\nFPS: {} {} \n'.format(1 / (_t['im_detect'].average_time), 1 / _t['im_detect'].average_time))
    print('Evaluating detections')
    stats = dataset.evaluate_detections(all_boxes, save_folder)
    AP_stats['ap'].append(stats[0])
    AP_stats['ap50'].append(stats[1])
    AP_stats['ap75'].append(stats[2])
    AP_stats['ap_small'].append(stats[3])
    AP_stats['ap_medium'].append(stats[4])
    AP_stats['ap_large'].append(stats[5])


if __name__ == '__main__':
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    prefix = args.prefix
    # prefix = 'weights/tmp'
    # prefix = 'weights/at_2e3'
    # prefix = 'weights/at_4e3'
    # prefix = 'weights/at1_4e3_01'
    # prefix = 'weights/at1_4e3_05'
    # prefix = 'weights/at1_mh_4e3_1'
    # prefix = 'weights/at1_mh_4e3_01'  # sigma 0.2
    # prefix = 'weights/at1_mh_4e3_01_5125vggbn'  # sigma 0.2
    # prefix = 'weights/at1_mh_4e3_01_sigma1'
    # prefix = 'weights/at1_mh_4e3_1_ce_sigma1'
    # prefix = 'weights/at1_mh_4e3_1_ce_sigma02'
    # prefix = 'weights/at1_mh2_4e3_1'
    # prefix = 'weights/at2_mh_4e3_03'
    # prefix = 'weights/at2_mh_4e3_01'
    # prefix = 'weights/at2_4e3_03'
    # prefix = 'weights/at2_4e3_01'
    save_folder = os.path.join(args.save_folder, prefix.split('/')[-1])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    maxout = args.maxout
    model = args.model
    # model = '512_vggbn'
    # model = '5125_vggbn'
    # model = '640_vggbn'
    # model = '512_ResNet_101'
    # model = '512_ResNet_50'
    # model = '1024_ResNet_101'
    # model = '1024_ResNeXt_152'
    if model == '512_ResNet_50':
        from sardet.refinedet_res import build_refinedet
        args.input_size = str(512)
        backbone_dict = dict(type='ResNet',depth=50, frozen_stages=-1)
    if model == '512_ResNet_101':
        from sardet.refinedet_res import build_refinedet
        args.input_size = str(512)
        backbone_dict = dict(type='ResNet',depth=101, frozen_stages=-1)
    elif model == '1024_ResNet_101':
        from sardet.refinedet_res import build_refinedet
        args.input_size = str(1024)
        backbone_dict = dict(type='ResNet',depth=101, frozen_stages=-1)
    elif model == '1024_ResNeXt_152':
        from sardet.refinedet_res import build_refinedet
        args.input_size = str(1024)
        backbone_dict = dict(type='ResNeXt',depth=152, frozen_stages=-1)
    elif model == '512_vggbn':
        from sardet.refinedet_bn_at1_mh import build_refinedet
        args.input_size = str(512)
        backbone_dict = dict(bn=True)
    elif model == '5125_vggbn':
        from sardet.refinedet_bn_at1_mh import build_refinedet
        args.input_size = str(5125)
        backbone_dict = dict(bn=True)
    elif model == '5126_vggbn':
        if maxout:
            from sardet.refinedet_bn_at1_mh_mxo import build_refinedet
        else:
            from sardet.refinedet_bn_at1_mh import build_refinedet
        args.input_size = str(5126)
        backbone_dict = dict(bn=True)
    elif model == '640_vggbn':
        if maxout:
            from sardet.refinedet_bn_at1_mh_mxo import build_refinedet
        else:
            from sardet.refinedet_bn_at1_mh import build_refinedet
        args.input_size = str(640)
        backbone_dict = dict(bn=True)

    cfg = coco_refinedet[args.input_size]
    target_size = cfg['min_dim']
    seg_num_grids = cfg['feature_maps']  # [64, 32, 16, 8]
    num_classes = cfg['num_classes']
    objectness_threshold = 0.01
    args.nms_threshold = 0.49  # nms
    # args.nms_threshold = 0.45  # softnms
    args.confidence_threshold = 0.01
    args.top_k = 1000
    args.keep_top_k = 500
    args.vis_thres = 0.3
    # args.multi_scale_test = True
    # args.show_image = True

    # load data
    dataset = COCODetection(COCOroot, [('sarship', 'test')], None)
    # dataset = COCODetection(COCOroot, [('sarship', 'test_inshore')], None)
    # dataset = COCODetection(COCOroot, [('sarship', 'test_offshore')], None)

    # load net
    torch.set_grad_enabled(False)
    load_to_cpu = not args.cuda
    cudnn.benchmark = True
    device = torch.device('cuda' if args.cuda else 'cpu')
    detect = Detect_RefineDet(num_classes, int(args.input_size), 0, objectness_threshold, confidence_threshold=args.confidence_threshold, nms_threshold=args.nms_threshold, top_k=args.top_k, keep_top_k=args.keep_top_k)
    net = build_refinedet('test', int(args.input_size), num_classes, seg_num_grids, backbone_dict) 

    # test multi models, to filter out the best model.
    # start_epoch = 10; step = 10
    start_epoch = 200; step = 5
    ToBeTested = []
    ToBeTested = [prefix + f'/RefineDet{args.input_size}_COCO_epoches_{epoch}.pth' for epoch in range(start_epoch, 300, step)]
    ToBeTested.append(prefix + f'/RefineDet{args.input_size}_COCO_final.pth') 
    # ToBeTested.append(prefix + '/RefineDet512_COCO_epoches_245.pth') 
    # ToBeTested *= 5
    ap_stats = {"ap": [], "ap50": [], "ap75": [], "ap_small": [], "ap_medium": [], "ap_large": [], "epoch": []}
    for index, model_path in enumerate(ToBeTested):
        args.trained_model = model_path
        net = load_model(net, args.trained_model, load_to_cpu)
        net.eval()
        print('Finished loading model!')
        # print(net)
        net = net.to(device)
        # evaluation
        ap_stats['epoch'].append(start_epoch + index * step)
        print("evaluating epoch: {}".format(ap_stats['epoch'][-1]))
        if not args.multi_scale_test:
            single_scale_test_net(target_size, save_folder, net, num_classes, dataset, detect, AP_stats=ap_stats)
        else:
            multi_scale_test_net(target_size, save_folder, net, num_classes, dataset, detect, AP_stats=ap_stats)

    # print the best model.
    max_idx = np.argmax(np.asarray(ap_stats['ap50']))
    print('Best ap50: {:.4f} at epoch {}'.format(ap_stats['ap50'][max_idx], ap_stats['epoch'][max_idx]))
    print('ap: {:.4f}, ap50: {:.4f}, ap75: {:.4f}, ap_s: {:.4f}, ap_m: {:.4f}, ap_l: {:.4f}'.\
        format(ap_stats['ap'][max_idx], ap_stats['ap50'][max_idx], ap_stats['ap75'][max_idx], ap_stats['ap_small'][max_idx], ap_stats['ap_medium'][max_idx], ap_stats['ap_large'][max_idx]))
    max_idx = np.argmax(np.asarray(ap_stats['ap']))
    print('Best ap  : {:.4f} at epoch {}'.format(ap_stats['ap'][max_idx], ap_stats['epoch'][max_idx]))
    print('ap: {:.4f}, ap50: {:.4f}, ap75: {:.4f}, ap_s: {:.4f}, ap_m: {:.4f}, ap_l: {:.4f}'.\
        format(ap_stats['ap'][max_idx], ap_stats['ap50'][max_idx], ap_stats['ap75'][max_idx], ap_stats['ap_small'][max_idx], ap_stats['ap_medium'][max_idx], ap_stats['ap_large'][max_idx]))
    res_file = os.path.join(save_folder, 'ap_stats.json')
    print('Writing ap stats json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
        json.dump(ap_stats, fid)
    
    # plot curves
    fig_name = 'ap.png'
    metrics = ['ap', 'ap75', 'ap50', 'ap_small', 'ap_medium', 'ap_large']
    legend  = ['ap', 'ap75', 'ap50', 'ap_small', 'ap_medium', 'ap_large']
    plot_map(save_folder, ap_stats, metrics, legend, fig_name)
    txt_log = prefix + '/log.txt'
    plot_loss(save_folder, txt_log)
"""
refinedet
lr_2e3
Best ap50: 0.9802 at epoch 240
ap: 0.6022, ap50: 0.9802, ap75: 0.6750, ap_s: 0.5550, ap_m: 0.6715, ap_l: 0.6515
Best ap  : 0.6091 at epoch 290
ap: 0.6091, ap50: 0.9783, ap75: 0.6921, ap_s: 0.5646, ap_m: 0.6713, ap_l: 0.6569
inshore 
Best ap50: 0.9400 at epoch 270
ap: 0.5124, ap50: 0.9400, ap75: 0.5157, ap_s: 0.4715, ap_m: 0.5679, ap_l: 0.5575
Best ap  : 0.5171 at epoch 250
ap: 0.5171, ap50: 0.9365, ap75: 0.5157, ap_s: 0.4693, ap_m: 0.5835, ap_l: 0.5242
offshore 
Best ap50: 0.9893 at epoch 225
ap: 0.6393, ap50: 0.9893, ap75: 0.7521, ap_s: 0.5869, ap_m: 0.7125, ap_l: 0.7697
Best ap  : 0.6500 at epoch 275
ap: 0.6500, ap50: 0.9888, ap75: 0.7740, ap_s: 0.6026, ap_m: 0.7166, ap_l: 0.7605

align 4e3 b16
Best ap50: 0.9725 at epoch 230
ap: 0.6147, ap50: 0.9725, ap75: 0.6975, ap_s: 0.5665, ap_m: 0.6831, ap_l: 0.6665
Best ap  : 0.6225 at epoch 255
ap: 0.6225, ap50: 0.9714, ap75: 0.7194, ap_s: 0.5729, ap_m: 0.6922, ap_l: 0.6703

at0 4e3 
Best ap50: 0.9801 at epoch 250
ap: 0.6113, ap50: 0.9801, ap75: 0.7057, ap_s: 0.5639, ap_m: 0.6778, ap_l: 0.6822
Best ap  : 0.6221 at epoch 280
ap: 0.6221, ap50: 0.9734, ap75: 0.7235, ap_s: 0.5784, ap_m: 0.6831, ap_l: 0.7019
Best ap50: 0.9741 at epoch 215
ap: 0.6166, ap50: 0.9741, ap75: 0.7045, ap_s: 0.5813, ap_m: 0.6688, ap_l: 0.6718
Best ap  : 0.6291 at epoch 290
ap: 0.6291, ap50: 0.9740, ap75: 0.7438, ap_s: 0.5869, ap_m: 0.6914, ap_l: 0.6600

at1 4e3 aw01
Best ap50: 0.9744 at epoch 295
ap: 0.6220, ap50: 0.9744, ap75: 0.7126, ap_s: 0.5714, ap_m: 0.6980, ap_l: 0.6666
Best ap  : 0.6220 at epoch 295
ap: 0.6220, ap50: 0.9744, ap75: 0.7126, ap_s: 0.5714, ap_m: 0.6980, ap_l: 0.6666
at1 4e3 mh aw 1
Best ap50: 0.9771 at epoch 205
ap: 0.6110, ap50: 0.9771, ap75: 0.6949, ap_s: 0.5689, ap_m: 0.6765, ap_l: 0.6506
Best ap  : 0.6229 at epoch 245
ap: 0.6229, ap50: 0.9695, ap75: 0.7137, ap_s: 0.5765, ap_m: 0.6881, ap_l: 0.6845

################### at1 4e3 mh aw 01 #############
Best ap50: 0.9797 at epoch 235
ap: 0.6160, ap50: 0.9797, ap75: 0.6952, ap_s: 0.5645, ap_m: 0.6962, ap_l: 0.6520
Best ap  : 0.6278 at epoch 285
ap: 0.6278, ap50: 0.9728, ap75: 0.7241, ap_s: 0.5823, ap_m: 0.6969, ap_l: 0.6628

Best ap50: 0.9747 at epoch 300
ap: 0.6248, ap50: 0.9747, ap75: 0.7381, ap_s: 0.5703, ap_m: 0.6999, ap_l: 0.6892
Best ap  : 0.6251 at epoch 280
ap: 0.6251, ap50: 0.9744, ap75: 0.7130, ap_s: 0.5697, ap_m: 0.7006, ap_l: 0.6861

Best ap50: 0.9758 at epoch 210
ap: 0.6215, ap50: 0.9758, ap75: 0.7015, ap_s: 0.5809, ap_m: 0.6847, ap_l: 0.6423
Best ap  : 0.6234 at epoch 215
ap: 0.6234, ap50: 0.9724, ap75: 0.7107, ap_s: 0.5837, ap_m: 0.6824, ap_l: 0.6613

at1 4e3 mh aw 01 sigma1
Best ap50: 0.9824 at epoch 205
ap: 0.6175, ap50: 0.9824, ap75: 0.7053, ap_s: 0.5722, ap_m: 0.6790, ap_l: 0.6811
Best ap  : 0.6233 at epoch 300
ap: 0.6233, ap50: 0.9735, ap75: 0.7259, ap_s: 0.5781, ap_m: 0.6857, ap_l: 0.6608

at1_mh_4e3_aw1_ce_sigma1
Best ap50: 0.9828 at epoch 255
ap: 0.6265, ap50: 0.9828, ap75: 0.7237, ap_s: 0.5864, ap_m: 0.6886, ap_l: 0.6457
Best ap  : 0.6265 at epoch 255
ap: 0.6265, ap50: 0.9828, ap75: 0.7237, ap_s: 0.5864, ap_m: 0.6886, ap_l: 0.6457
at1_mh_4e3_aw1_ce_sigma02
Best ap50: 0.9836 at epoch 255
ap: 0.6216, ap50: 0.9836, ap75: 0.7045, ap_s: 0.5745, ap_m: 0.6882, ap_l: 0.6606
Best ap  : 0.6273 at epoch 245
ap: 0.6273, ap50: 0.9831, ap75: 0.7252, ap_s: 0.5802, ap_m: 0.6960, ap_l: 0.6691
Best ap50: 0.9749 at epoch 205
ap: 0.6120, ap50: 0.9749, ap75: 0.6851, ap_s: 0.5603, ap_m: 0.6873, ap_l: 0.6589
Best ap  : 0.6234 at epoch 275
ap: 0.6234, ap50: 0.9634, ap75: 0.7250, ap_s: 0.5745, ap_m: 0.6935, ap_l: 0.6747

at1 4e3 mh2
Best ap50: 0.9781 at epoch 275
ap: 0.6104, ap50: 0.9781, ap75: 0.7071, ap_s: 0.5590, ap_m: 0.6805, ap_l: 0.6920
Best ap  : 0.6182 at epoch 250
ap: 0.6182, ap50: 0.9779, ap75: 0.7257, ap_s: 0.5702, ap_m: 0.6827, ap_l: 0.6856

at2 4e3 aw01
Best ap50: 0.9756 at epoch 230
ap: 0.6201, ap50: 0.9756, ap75: 0.7068, ap_s: 0.5725, ap_m: 0.6914, ap_l: 0.6551
Best ap  : 0.6213 at epoch 255
ap: 0.6213, ap50: 0.9750, ap75: 0.7192, ap_s: 0.5717, ap_m: 0.6898, ap_l: 0.6774
at2 4e3 mh aw01
Best ap50: 0.9806 at epoch 225
ap: 0.6234, ap50: 0.9806, ap75: 0.7251, ap_s: 0.5744, ap_m: 0.6887, ap_l: 0.7107
Best ap  : 0.6274 at epoch 275
ap: 0.6274, ap50: 0.9719, ap75: 0.7242, ap_s: 0.5747, ap_m: 0.7016, ap_l: 0.7138
Best ap50: 0.9730 at epoch 300
ap: 0.6185, ap50: 0.9730, ap75: 0.7088, ap_s: 0.5649, ap_m: 0.6963, ap_l: 0.6784
Best ap  : 0.6235 at epoch 280
ap: 0.6235, ap50: 0.9656, ap75: 0.7284, ap_s: 0.5730, ap_m: 0.6984, ap_l: 0.6743
Best ap50: 0.9793 at epoch 210
ap: 0.6042, ap50: 0.9793, ap75: 0.6967, ap_s: 0.5611, ap_m: 0.6702, ap_l: 0.6921
Best ap  : 0.6233 at epoch 280
ap: 0.6233, ap50: 0.9726, ap75: 0.7242, ap_s: 0.5766, ap_m: 0.6900, ap_l: 0.7055
Best ap50: 0.9835 at epoch 205
ap: 0.6115, ap50: 0.9835, ap75: 0.7005, ap_s: 0.5714, ap_m: 0.6764, ap_l: 0.6669
Best ap  : 0.6201 at epoch 255
ap: 0.6201, ap50: 0.9747, ap75: 0.7178, ap_s: 0.5766, ap_m: 0.6838, ap_l: 0.6519

1.0==cps solo 2e3 bs16 g12(cps, complementary sampling), solo has less grid number than fcos
Best ap50: 0.9826 at epoch 265
ap: 0.6140, ap50: 0.9826, ap75: 0.7032, ap_s: 0.5641, ap_m: 0.6851, ap_l: 0.6542
Best ap  : 0.6212 at epoch 295
ap: 0.6212, ap50: 0.9809, ap75: 0.7185, ap_s: 0.5688, ap_m: 0.6948, ap_l: 0.6764
inshore 
Best ap50: 0.9449 at epoch 280
ap: 0.5335, ap50: 0.9449, ap75: 0.5323, ap_s: 0.4896, ap_m: 0.5979, ap_l: 0.5640
Best ap  : 0.5391 at epoch 300
ap: 0.5391, ap50: 0.9413, ap75: 0.5515, ap_s: 0.4916, ap_m: 0.6108, ap_l: 0.5388
offshore 
Best ap50: 0.9895 at epoch 200
ap: 0.5956, ap50: 0.9895, ap75: 0.6853, ap_s: 0.5501, ap_m: 0.6572, ap_l: 0.7790
Best ap  : 0.6562 at epoch 280
ap: 0.6562, ap50: 0.9893, ap75: 0.7934, ap_s: 0.6000, ap_m: 0.7357, ap_l: 0.8126

1.0==cps fcos
Best ap50: 0.9798 at epoch 300
ap: 0.6238, ap50: 0.9798, ap75: 0.7139, ap_s: 0.5805, ap_m: 0.6876, ap_l: 0.6612
Best ap : 0.6246 at epoch 295
ap: 0.6246, ap50: 0.9796, ap75: 0.7154, ap_s: 0.5821, ap_m: 0.6862, ap_l: 0.6693
inshore 
Best ap50: 0.9345 at epoch 235
ap: 0.5218, ap50: 0.9345, ap75: 0.5302, ap_s: 0.4627, ap_m: 0.6103, ap_l: 0.5061
Best ap : 0.5418 at epoch 295
ap: 0.5418, ap50: 0.9334, ap75: 0.5716, ap_s: 0.4998, ap_m: 0.6064, ap_l: 0.5539
offshore
Best ap50: 0.9897 at epoch 210
ap: 0.6415, ap50: 0.9897, ap75: 0.7580, ap_s: 0.5853, ap_m: 0.7127, ap_l: 0.8235
Best ap : 0.6594 at epoch 240
ap: 0.6594, ap50: 0.9896, ap75: 0.7994, ap_s: 0.6058, ap_m: 0.7275, ap_l: 0.7844

in ap:   cps > cs, fcos > solo
in ap50: cps > cs, fcos = solo
best solution: cps + fcos grid number

2.0==cs solo(consistent sampling)
Best ap50: 0.9723 at epoch 250
ap: 0.6037, ap50: 0.9723, ap75: 0.6872, ap_s: 0.5528, ap_m: 0.6759, ap_l: 0.6628
Best ap  : 0.6109 at epoch 295
ap: 0.6109, ap50: 0.9707, ap75: 0.7024, ap_s: 0.5656, ap_m: 0.6742, ap_l: 0.6737
inshore 
Best ap50: 0.9217 at epoch 285
ap: 0.5187, ap50: 0.9217, ap75: 0.5607, ap_s: 0.4865, ap_m: 0.5697, ap_l: 0.5492
Best ap  : 0.5193 at epoch 275
ap: 0.5193, ap50: 0.9193, ap75: 0.5444, ap_s: 0.4824, ap_m: 0.5792, ap_l: 0.5443
offshore 
Best ap50: 0.9894 at epoch 210
ap: 0.6308, ap50: 0.9894, ap75: 0.7157, ap_s: 0.5792, ap_m: 0.6993, ap_l: 0.7628
Best ap  : 0.6491 at epoch 295
ap: 0.6491, ap50: 0.9890, ap75: 0.7612, ap_s: 0.5980, ap_m: 0.7160, ap_l: 0.8113

2.0==cs fcos
Best ap50: 0.9790 at epoch 240
ap: 0.6124, ap50: 0.9790, ap75: 0.7034, ap_s: 0.5725, ap_m: 0.6739, ap_l: 0.6302
Best ap  : 0.6135 at epoch 290
ap: 0.6135, ap50: 0.9734, ap75: 0.7075, ap_s: 0.5666, ap_m: 0.6798, ap_l: 0.6465
inshore 
Best ap50: 0.9363 at epoch 220
ap: 0.5110, ap50: 0.9363, ap75: 0.5118, ap_s: 0.4567, ap_m: 0.5925, ap_l: 0.4804
Best ap  : 0.5284 at epoch 250
ap: 0.5284, ap50: 0.9327, ap75: 0.5372, ap_s: 0.4880, ap_m: 0.5949, ap_l: 0.4907
offshore
Best ap50: 0.9896 at epoch 255
ap: 0.6485, ap50: 0.9896, ap75: 0.7613, ap_s: 0.5956, ap_m: 0.7226, ap_l: 0.7752
Best ap  : 0.6524 at epoch 290
ap: 0.6524, ap50: 0.9893, ap75: 0.7766, ap_s: 0.5994, ap_m: 0.7228, ap_l: 0.7984



"""