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
from data import COCOroot, COCODetection
import torch.utils.data as data

from sardet.refinedet import build_refinedet

from layers import Detect_RefineDet
from utils.nms_wrapper import nms

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
# parser.add_argument('--voc_root', default=VOC_ROOT,
#                     help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--input_size', default='512', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--prefix', default='weights/lr_5e4', type=str, help='File path to save results')

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

class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.float32(img), (self.resize, self.resize),interpolation = interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)


def test_net(save_folder, net, device, num_classes, dataset, transform, top_k, max_per_image=300, confidence_threshold=0.005, nms_threshold=0.4, AP_stats=None):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    det_file = os.path.join(save_folder, 'detections.pkl')
    
    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        dataset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img, target = dataset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        x = transform(img).unsqueeze(0)
        x = x.to(device)
        scale = scale.to(device)

        _t['im_detect'].tic()
        boxes, scores = net(x)
        boxes = boxes[0]
        scores=scores[0]

        # scale each detection back up to the image
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > confidence_threshold)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]

            # keep top-K before NMS
            order = c_scores.argsort()[::-1][:top_k]
            c_bboxes = c_bboxes[order]
            c_scores = c_scores[order]

            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, nms_threshold, force_cpu=(not args.cuda))
            c_dets = c_dets[keep, :]
            c_dets = c_dets[:max_per_image, :]
            all_boxes[j][i] = c_dets
        _t['im_detect'].toc()

        # print('im_detect: {:d}/{:d} forward_nms_time{:.4f}s'.format(i + 1, num_images, _t['im_detect'].average_time))
        if args.show_image:
            h, w, _ = img.shape
            xr = net.size / w
            yr = net.size / h
            img_gt = img.astype(np.uint8)
            img_gt = cv2.resize(img_gt, (net.size, net.size),interpolation=cv2.INTER_LINEAR)
            for b in target:
                b[0] *= xr
                b[2] *= xr
                b[1] *= yr
                b[3] *= yr
                b = list(map(int, b))
                cv2.rectangle(img_gt, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
                cx = b[2]
                cy = b[1]
                # text = "ship"
                # cv2.putText(img_gt, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
            # boxes = all_boxes[1][i][:]
            # for b in boxes:
            #     b[0] *= xr
            #     b[2] *= xr
            #     b[1] *= yr
            #     b[3] *= yr
            #     if b[4] < args.vis_thres:
            #         continue
            #     b = list(map(int, b))
            #     cv2.rectangle(img_gt, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)
            #     cx = b[2]
            #     cy = b[1] + 12
            #     # text = "{:.2f}".format(b[4])
            #     # cv2.putText(img_gt, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            # cv2.imshow('res', img_gt)
            # cv2.waitKey(0)
            save_gt_dir = os.path.join(save_folder, 'gt_img')
            if not os.path.exists(save_gt_dir):
                os.mkdir(save_gt_dir)
            cv2.imwrite(save_gt_dir + f'/{i}.png',img_gt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

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
    
    # args.trained_model = 'weights/lr_1e3/RefineDet512_COCO_final.pth'
    # args.trained_model = 'weights/lr_5e4/RefineDet512_COCO_final.pth'
    # args.cuda = False
    # args.retest = True
    args.show_image = True
    # prefix = args.prefix
    # prefix = 'weights/tmp'
    prefix = 'weights/at_2e3'

    save_folder = os.path.join(args.save_folder, prefix.split('/')[-1])

    nms_threshold = 0.49
    confidence_threshold = 0.01
    objectness_thre = 0.01
    # 'feature_maps': [64, 32, 16, 8]
    seg_num_grids = [64, 32, 16, 8]

    num_classes = 2 
    top_k = 1000
    keep_top_k = 500
    torch.set_grad_enabled(False)

    # load data
    rgb_means = (98.13131, 98.13131, 98.13131)
    # dataset = COCODetection(COCOroot, [('sarship', 'test')], None)
    dataset = COCODetection(COCOroot, [('sarship', 'test_inshore')], None)
    # dataset = COCODetection(COCOroot, [('sarship', 'test_offshore')], None)

    # load net
    detect = Detect_RefineDet(num_classes, int(args.input_size), 0, top_k, confidence_threshold, nms_threshold, objectness_thre, keep_top_k)
    net = build_refinedet('test', int(args.input_size), num_classes, seg_num_grids, detector=detect) 
    load_to_cpu = not args.cuda
    cudnn.benchmark = True
    device = torch.device('cuda' if args.cuda else 'cpu')

    ap_stats = {"ap": [], "ap50": [], "ap75": [], "ap_small": [], "ap_medium": [], "ap_large": [], "epoch": []}

    start_epoch = 10; step = 10
    # start_epoch = 200; step = 5
    ToBeTested = []
    ToBeTested = [prefix + f'/RefineDet512_COCO_epoches_{epoch}.pth' for epoch in range(start_epoch, 300, step)]
    ToBeTested.append(prefix + '/RefineDet512_COCO_final.pth') 
    # ToBeTested.append(prefix + '/RefineDet512_COCO_epoches_295.pth') 
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
        test_net(save_folder, net, device, num_classes, dataset, 
                BaseTransform(net.size, rgb_means, (2, 0, 1)), top_k, 
                keep_top_k, confidence_threshold=confidence_threshold, nms_threshold=nms_threshold, AP_stats=ap_stats)

    print(ap_stats)
    res_file = os.path.join(save_folder, 'ap_stats.json')

    max_idx = np.argmax(np.asarray(ap_stats['ap50']))
    print('Best ap50: {:.4f} at epoch {}'.format(ap_stats['ap50'][max_idx], ap_stats['epoch'][max_idx]))
    print('ap: {:.4f}, ap50: {:.4f}, ap75: {:.4f}, ap_s: {:.4f}, ap_m: {:.4f}, ap_l: {:.4f}'.\
        format(ap_stats['ap'][max_idx], ap_stats['ap50'][max_idx], ap_stats['ap75'][max_idx], ap_stats['ap_small'][max_idx], ap_stats['ap_medium'][max_idx], ap_stats['ap_large'][max_idx]))
    max_idx = np.argmax(np.asarray(ap_stats['ap']))
    print('Best ap  : {:.4f} at epoch {}'.format(ap_stats['ap'][max_idx], ap_stats['epoch'][max_idx]))
    print('ap: {:.4f}, ap50: {:.4f}, ap75: {:.4f}, ap_s: {:.4f}, ap_m: {:.4f}, ap_l: {:.4f}'.\
        format(ap_stats['ap'][max_idx], ap_stats['ap50'][max_idx], ap_stats['ap75'][max_idx], ap_stats['ap_small'][max_idx], ap_stats['ap_medium'][max_idx], ap_stats['ap_large'][max_idx]))

    import json
    print('Writing ap stats json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
        json.dump(ap_stats, fid)
    with open(res_file) as f:
        ap_stats = json.load(f)
    
    from plot_curve import plot_map, plot_loss
    fig_name = 'ap.png'
    # fig_name = 'ap_last10.png'
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

solo 2e3 bs16 g12
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

1.0 fcos
Best ap50: 0.9798 at epoch 300
ap: 0.6238, ap50: 0.9798, ap75: 0.7139, ap_s: 0.5805, ap_m: 0.6876, ap_l: 0.6612
Best ap  : 0.6246 at epoch 295
ap: 0.6246, ap50: 0.9796, ap75: 0.7154, ap_s: 0.5821, ap_m: 0.6862, ap_l: 0.6693
inshore 
Best ap50: 0.9345 at epoch 235
ap: 0.5218, ap50: 0.9345, ap75: 0.5302, ap_s: 0.4627, ap_m: 0.6103, ap_l: 0.5061
Best ap  : 0.5418 at epoch 295
ap: 0.5418, ap50: 0.9334, ap75: 0.5716, ap_s: 0.4998, ap_m: 0.6064, ap_l: 0.5539
offshore
Best ap50: 0.9897 at epoch 210
ap: 0.6415, ap50: 0.9897, ap75: 0.7580, ap_s: 0.5853, ap_m: 0.7127, ap_l: 0.8235
Best ap  : 0.6594 at epoch 240
ap: 0.6594, ap50: 0.9896, ap75: 0.7994, ap_s: 0.6058, ap_m: 0.7275, ap_l: 0.7844

2.0cs
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

2.0cs fcos
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