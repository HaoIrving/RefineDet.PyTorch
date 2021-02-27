from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import RefineDetMultiBoxLoss, AttentionFocalLoss
from sardet.refinedet_bn import build_refinedet
# from sardet.refinedet_bn_at2 import build_refinedet


import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from utils.logger import Logger
import math
import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--input_size', default='512', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
# parser.add_argument('--start_iter', default=0, type=int,
#                     help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')               
parser.add_argument('--ngpu', default=4, type=int, help='gpus')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

sys.stdout = Logger(os.path.join(args.save_folder, 'log.txt'))

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.system('CUDA_VISIBLE__DEVICES = 2')

# args.lr = 1e-5
# args.batch_size = 4
# args.ngpu = 2
# args.num_workers = 0
args.input_size = str(512)
args.max_epoch = 300

# 'feature_maps': [64, 32, 16, 8],
# seg_num_grids = [36, 24, 16, 12]
seg_num_grids = [64, 32, 16, 8]

# ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)) for stride from 4
# anchor [32, 64, 128, 256]
# scale_ranges = ((1, 96), (48, 192), (96, 384), (192, 768))
scale_ranges = ((1, 64), (32, 128), (64, 256), (128, 512))
att_loss_weight = 1

negpos_ratio = 3
initial_lr = args.lr

def train():
    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    
    print('Loading the dataset...')
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCOroot):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCOroot
        cfg = coco_refinedet[args.input_size]
        # dataset = COCODetection(root=args.dataset_root, image_set='train',
        #                         transform=SSDAugmentation(cfg['min_dim'],
        #                                                   MEANS))
        train_sets = [('sarship', 'train')]
        dataset = COCODetection(COCOroot, train_sets, SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'VOC':
        '''if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')'''
        cfg = voc_refinedet[args.input_size]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))
    print('Training RefineDet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    refinedet_net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'], seg_num_grids)
    net = refinedet_net
    print(net)

    if args.ngpu > 1 and args.cuda:
        net = torch.nn.DataParallel(refinedet_net, device_ids=list(range(args.ngpu)))
    cudnn.benchmark = True
    net = net.to(device)
    
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:
        print('Initializing weights...')
        # refinedet_net.init_weights(pretrained=args.basenet)
        refinedet_net.init_weights(pretrained=None)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(                 2, 0.5, True, 0, True, negpos_ratio, 0.5,
                             False, args.cuda)
    odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, negpos_ratio, 0.5,
                             False, args.cuda, use_ARM=True)
    # attention criterion 
    loss_cate = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0)
    attention_criterion = AttentionFocalLoss(cfg['num_classes'], cfg['min_dim'], loss_cate, seg_num_grids, scale_ranges)

    net.train()
    # loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    epoch = 0 + args.resume_epoch

    epoch_size = math.ceil(len(dataset) / args.batch_size)
    max_iter = args.max_epoch * epoch_size
    
    stepvalues = (args.max_epoch * 2 // 3 * epoch_size, args.max_epoch * 8 // 9 * epoch_size, args.max_epoch * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    if args.visdom:
        vis_title = 'RefineDet.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot(viz, 'Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot(viz, 'Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            if args.visdom and iteration != 0:
                update_vis_plot(viz, epoch, arm_loc_loss, arm_conf_loss, epoch_plot, None,
                                'append', epoch_size)
                # reset epoch loss counters
                arm_loc_loss = 0
                arm_conf_loss = 0
                odm_loc_loss = 0
                odm_conf_loss = 0
            # create batch iterator
            batch_iterator = iter(data_loader)
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 ==0 and epoch > 200):
                torch.save(net.state_dict(), args.save_folder+'RefineDet'+ args.input_size +'_'+args.dataset + '_epoches_'+
                           repr(epoch) + '.pth')
            epoch += 1

        t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [ann.to(device) for ann in targets]
        for an in targets:
            for instance in an:
                for cor in instance[:-1]:
                    if cor < 0 or cor > 1:
                        raise StopIteration

        # forward
        attention_maps, arm_loc, arm_conf, odm_loc, odm_conf, priors = net(images)
        out = (arm_loc, arm_conf, odm_loc, odm_conf, priors)

        # backprop
        optimizer.zero_grad()
        arm_loss_l, arm_loss_c = arm_criterion(out, targets)
        odm_loss_l, odm_loss_c = odm_criterion(out, targets)
        arm_loss = arm_loss_l + arm_loss_c
        odm_loss = odm_loss_l + odm_loss_c

        attention_loss = attention_criterion(attention_maps, targets)
        loss = arm_loss + odm_loss + attention_loss * att_loss_weight
        
        loss.backward()
        optimizer.step()
        arm_loc_loss += arm_loss_l.item()
        arm_conf_loss += arm_loss_c.item()
        odm_loc_loss += odm_loss_l.item()
        odm_conf_loss += odm_loss_c.item()
        t1 = time.time()
        batch_time = t1 - t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || ARM_L Loss: {:.4f} ARM_C Loss: {:.4f} ODM_L Loss: {:.4f} ODM_C Loss: {:.4f} ATT Loss: {:.4f} loss: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.\
            format(epoch, args.max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item(), attention_loss.item(), loss.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        # if iteration % 10 == 0:
        #     print('Batch time: %.4f sec. ||' % (batch_time) + 'Eta: {}'.format(str(datetime.timedelta(seconds=eta))))
        #     print('iter ' + repr(iteration) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f loss: %.4f ||' \
        #     % (arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item(), loss.item()), end=' ')

        if args.visdom:
            update_vis_plot(viz, iteration, arm_loss_l.item(), arm_loss_c.item(),
                            iter_plot, epoch_plot, 'append')

    torch.save(refinedet_net.state_dict(), args.save_folder + '/RefineDet{}_{}_final.pth'.format(args.input_size, args.dataset))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = 5
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(viz, _xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(viz, iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
