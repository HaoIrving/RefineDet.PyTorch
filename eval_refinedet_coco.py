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

from layers import Detect_RefineDet, Detect
from utils.nms_wrapper import nms, soft_nms
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
parser.add_argument('--model', default='640_vggbn', type=str, help='model name')
parser.add_argument('-woalign', '--wo_alignconv', action="store_true", default=False, help=' ')
parser.add_argument('-worefine', '--wo_refined_anchor', action="store_true", default=False, help=' ')
parser.add_argument('-wofuse', '--wo_fused_feature', action="store_true", default=False, help=' ')
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
    try:
        device = net.arm_conf[0].weight.device
    except:
        device = net.odm_conf[0].weight.device
    h, w, _ = im.shape
    scale = torch.Tensor([w, h, w, h])
    scale = scale.to(device)
    im_orig = im.astype(np.float32, copy=True)
    im = cv2.resize(im_orig, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    x = (im - MEANS).astype(np.float32)
    x = x.transpose(2, 0, 1)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(device)

    if args.wo_refined_anchor:
        adm_loc, adm_conf, feat_sizes = net(x)
    else:
        arm_loc, arm_conf, adm_loc, adm_conf, feat_sizes = net(x)
    priorbox = PriorBox(net.cfg, feat_sizes, (target_size, target_size), phase='test')
    priors = priorbox.forward()
    priors = priors.to(device)
    if args.wo_refined_anchor:
        det = detect.forward(adm_loc, adm_conf, priors, scale)
    else:
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
        # det = im_detect_ratio(net, im, target_size, int(0.75*target_size))
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
    # prefix = 'weights/align_2e3_512res50'
    # prefix = 'weights/align_1e3_512res101'
    # prefix = 'weights/align_4e3_640vggbn'
    
    # prefix = 'weights/align_4e3_5l'
    # prefix = 'weights/align_2e3'
    save_folder = os.path.join(args.save_folder, prefix.split('/')[-1])
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    from utils.logger import Logger
    sys.stdout = Logger(os.path.join(save_folder, 'eval.txt'))

    # args.show_image = True

    # args.wo_fused_feature = True
    wo_fused_feature = args.wo_fused_feature
    # args.wo_refined_anchor = True
    wo_refined_anchor = args.wo_refined_anchor
    # args.wo_alignconv = True
    wo_alignconv = args.wo_alignconv
    model = args.model
    # model = '768_vggbn'
    # model = '640_vggbn'
    # model = '512_vggbn'
    # model = '512_ResNet_101'
    # model = '512_ResNet_50'
    # model = '1024_ResNet_101'
    # model = '1024_ResNeXt_152'
    if model == '512_ResNet_50':
        from models.refinedet_res import build_refinedet
        args.input_size = str(512)
        backbone_dict = dict(type='ResNet',depth=50, frozen_stages=-1)
    if model == '512_ResNet_101':
        from models.refinedet_res import build_refinedet
        args.input_size = str(512)
        backbone_dict = dict(type='ResNet',depth=101, frozen_stages=-1)
    elif model == '1024_ResNet_101':
        from models.refinedet_res import build_refinedet
        args.input_size = str(1024)
        backbone_dict = dict(type='ResNet',depth=101, frozen_stages=-1)
    elif model == '1024_ResNeXt_152':
        from models.refinedet_res import build_refinedet
        args.input_size = str(1024)
        backbone_dict = dict(type='ResNeXt',depth=152, frozen_stages=-1)
    elif model == '512_vggbn':
        from models.refinedet_bn import build_refinedet
        args.input_size = str(512)
        backbone_dict = dict(bn=True)
    elif model == '768_vggbn':
        from models.refinedet_bn import build_refinedet
        args.input_size = str(768)
        backbone_dict = dict(bn=True)
    elif model == '640_vggbn':
        from models.refinedet_bn import build_refinedet
        if wo_alignconv:
            from models.refinedet_bn_wo_AlignConv import build_refinedet
        if wo_refined_anchor:
            from models.refinedet_bn_wo_AlignConv_RefinedAnchor import build_refinedet
        if wo_fused_feature:
            from models.refinedet_bn_wo_AlignConv_RefinedAnchor_FusedFeature import build_refinedet
            args.wo_refined_anchor = True
        args.input_size = str(640)
        backbone_dict = dict(bn=True)
    
    target_size = int(args.input_size)
    # target_size = 1024
    num_classes = 2 
    objectness_threshold = 0.01
    args.nms_threshold = 0.49  # nms
    # args.nms_threshold = 0.45  # softnms
    args.confidence_threshold = 0.01
    args.top_k = 1000
    args.keep_top_k = 500
    args.vis_thres = 0.3
    args.multi_scale_test = True

    # load data
    dataset = COCODetection(COCOroot, [('sarship', 'test')], None, dataset_name='sarship')
    # dataset = COCODetection(COCOroot, [('sarship', 'test_inshore')], None)
    # dataset = COCODetection(COCOroot, [('sarship', 'test_offshore')], None)

    # load net
    torch.set_grad_enabled(False)
    load_to_cpu = not args.cuda
    cudnn.benchmark = True
    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.wo_refined_anchor:
        detect = Detect(          num_classes, int(args.input_size), 0,                       confidence_threshold=args.confidence_threshold, nms_threshold=args.nms_threshold, top_k=args.top_k, keep_top_k=args.keep_top_k)
    else:
        detect = Detect_RefineDet(num_classes, int(args.input_size), 0, objectness_threshold, confidence_threshold=args.confidence_threshold, nms_threshold=args.nms_threshold, top_k=args.top_k, keep_top_k=args.keep_top_k)
    net = build_refinedet('test', int(args.input_size), num_classes, backbone_dict) 

    # test multi models, to filter out the best model.
    # start_epoch = 10; step = 10
    start_epoch = 200; step = 5
    ToBeTested = []
    # ToBeTested = [prefix + f'/RefineDet{args.input_size}_COCO_epoches_{epoch}.pth' for epoch in range(start_epoch, 300, step)]
    ToBeTested.append(prefix + f'/RefineDet{args.input_size}_COCO_final.pth') 
    # ToBeTested.append(prefix + '/RefineDet512_COCO_epoches_280.pth') 
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

lr_3e3
Best ap50: 0.9814 at epoch 240
ap: 0.6055, ap50: 0.9814, ap75: 0.6981, ap_s: 0.5580, ap_m: 0.6792, ap_l: 0.6230
Best ap  : 0.6094 at epoch 285
ap: 0.6094, ap50: 0.9797, ap75: 0.7029, ap_s: 0.5587, ap_m: 0.6797, ap_l: 0.6458
lr 4e3 bs16
Best ap50: 0.9826 at epoch 245
ap: 0.6111, ap50: 0.9826, ap75: 0.6637, ap_s: 0.5572, ap_m: 0.6933, ap_l: 0.6133
Best ap  : 0.6257 at epoch 290
ap: 0.6257, ap50: 0.9751, ap75: 0.7296, ap_s: 0.5739, ap_m: 0.7000, ap_l: 0.6462

align
4e3 b16
Best ap50: 0.9825 at epoch 230
ap: 0.6206, ap50: 0.9825, ap75: 0.7117, ap_s: 0.5782, ap_m: 0.6782, ap_l: 0.6902
Best ap  : 0.6294 at epoch 280
ap: 0.6294, ap50: 0.9735, ap75: 0.7296, ap_s: 0.5840, ap_m: 0.6960, ap_l: 0.6728
Best ap50: 0.9725 at epoch 230
ap: 0.6147, ap50: 0.9725, ap75: 0.6975, ap_s: 0.5665, ap_m: 0.6831, ap_l: 0.6665
Best ap  : 0.6225 at epoch 255
ap: 0.6225, ap50: 0.9714, ap75: 0.7194, ap_s: 0.5729, ap_m: 0.6922, ap_l: 0.6703

mstest less
Best ap50: 0.9764 at epoch 200
ap: 0.6455, ap50: 0.9764, ap75: 0.7564, ap_s: 0.6041, ap_m: 0.7067, ap_l: 0.7016
Best ap  : 0.6455 at epoch 200
ap: 0.6455, ap50: 0.9764, ap75: 0.7564, ap_s: 0.6041, ap_m: 0.7067, ap_l: 0.7016
mstest more 
Best ap50: 0.9755 at epoch 200
ap: 0.6458, ap50: 0.9755, ap75: 0.7565, ap_s: 0.6068, ap_m: 0.7031, ap_l: 0.6890
Best ap  : 0.6458 at epoch 200
ap: 0.6458, ap50: 0.9755, ap75: 0.7565, ap_s: 0.6068, ap_m: 0.7031, ap_l: 0.6890
mstest less softbv
Best ap50: 0.9757 at epoch 200
ap: 0.6621, ap50: 0.9757, ap75: 0.7898, ap_s: 0.6200, ap_m: 0.7240, ap_l: 0.7251
Best ap  : 0.6621 at epoch 200
ap: 0.6621, ap50: 0.9757, ap75: 0.7898, ap_s: 0.6200, ap_m: 0.7240, ap_l: 0.7251
mstest more softbv
Best ap50: 0.9785 at epoch 200
ap: 0.6609, ap50: 0.9785, ap75: 0.7901, ap_s: 0.6210, ap_m: 0.7211, ap_l: 0.7103
Best ap  : 0.6609 at epoch 200
ap: 0.6609, ap50: 0.9785, ap75: 0.7901, ap_s: 0.6210, ap_m: 0.7211, ap_l: 0.7103

softnms is suitable when single scale test.
Best ap50: 0.9791 at epoch 285
ap: 0.6464, ap50: 0.9791, ap75: 0.7664, ap_s: 0.6016, ap_m: 0.7116, ap_l: 0.6871
Best ap  : 0.6478 at epoch 280
ap: 0.6478, ap50: 0.9762, ap75: 0.7676, ap_s: 0.6036, ap_m: 0.7114, ap_l: 0.6815
mstest less snms 
Best ap50: 0.9755 at epoch 200
ap: 0.6427, ap50: 0.9755, ap75: 0.7499, ap_s: 0.6014, ap_m: 0.7041, ap_l: 0.6864
Best ap  : 0.6427 at epoch 200
ap: 0.6427, ap50: 0.9755, ap75: 0.7499, ap_s: 0.6014, ap_m: 0.7041, ap_l: 0.6864
mstest more snms
Best ap50: 0.9757 at epoch 200
ap: 0.6451, ap50: 0.9757, ap75: 0.7565, ap_s: 0.6068, ap_m: 0.7002, ap_l: 0.7059
Best ap  : 0.6451 at epoch 200
ap: 0.6451, ap50: 0.9757, ap75: 0.7565, ap_s: 0.6068, ap_m: 0.7002, ap_l: 0.7059
mstest less snms softbv
Best ap50: 0.9748 at epoch 200
ap: 0.6597, ap50: 0.9748, ap75: 0.7841, ap_s: 0.6184, ap_m: 0.7225, ap_l: 0.7086
Best ap  : 0.6597 at epoch 200
ap: 0.6597, ap50: 0.9748, ap75: 0.7841, ap_s: 0.6184, ap_m: 0.7225, ap_l: 0.7086
mstest more snms softbv
Best ap50: 0.9770 at epoch 200
ap: 0.6600, ap50: 0.9770, ap75: 0.7912, ap_s: 0.6206, ap_m: 0.7189, ap_l: 0.7233
Best ap  : 0.6600 at epoch 200
ap: 0.6600, ap50: 0.9770, ap75: 0.7912, ap_s: 0.6206, ap_m: 0.7189, ap_l: 0.7233

896 5l
Best ap50: 0.9745 at epoch 225
ap: 0.6259, ap50: 0.9745, ap75: 0.7196, ap_s: 0.5832, ap_m: 0.6835, ap_l: 0.6831
Best ap  : 0.6290 at epoch 290
ap: 0.6290, ap50: 0.9720, ap75: 0.7116, ap_s: 0.5918, ap_m: 0.6769, ap_l: 0.6896

res50 2e3 pt
Best ap50: 0.9646 at epoch 270
ap: 0.6016, ap50: 0.9646, ap75: 0.6886, ap_s: 0.5532, ap_m: 0.6803, ap_l: 0.7029
Best ap  : 0.6046 at epoch 265
ap: 0.6046, ap50: 0.9610, ap75: 0.6924, ap_s: 0.5603, ap_m: 0.6772, ap_l: 0.6758
res101 1e3 pt
Best ap50: 0.9770 at epoch 255
ap: 0.6044, ap50: 0.9770, ap75: 0.6731, ap_s: 0.5508, ap_m: 0.6880, ap_l: 0.7622
Best ap  : 0.6111 at epoch 220
ap: 0.6111, ap50: 0.9693, ap75: 0.7196, ap_s: 0.5610, ap_m: 0.6879, ap_l: 0.7283

"""