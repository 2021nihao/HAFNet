import os
import shutil
import json
import time

# from apex import amp
from torch.cuda import amp
import apex
import copy

import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from toolbox import MscCrossEntropyLoss
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger
from toolbox import setup_seed
from toolbox.losses import pytorch_iou
from toolbox.losses.pytorch_iou.ciou_loss2 import ciou
from toolbox.losses.edge_loss import EdgeHoldLoss
# from toolbox.losses.EdgeHoldLoss import EdgeHoldLoss
from toolbox.losses.GHM_loss import GHM_Loss, GHMC_Loss, GHMR_Loss
from toolbox.losses.focal_loss import BinaryFocalLoss
from toolbox.losses.dice_loss import BinaryDiceLoss
from toolbox.losses.pytorch_ssim.SSIM_loss import SSIMLoss, LOGSSIMLoss
from toolbox.losses.pytorch_ssim.ssim import SSIM
from backbone.RepVGG_main.repvgg import repvgg_model_convert
from toolbox import load_ckpt
from toolbox import group_weight_decay
# from toolbox import ber
from toolbox import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d, \
    LovaszSoftmax, LDAMLoss
# from metrics_mg.evaluator import Eval_thread
setup_seed(33)

class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.6238, 5.7808])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4241, 46.2611])).float()

        self.class_weight = class_weight
        self.LovaszSoftmax = LovaszSoftmax()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        self.dice_loss = BinaryDiceLoss()
        self.IOU = pytorch_iou.IOU(size_average=True).cuda()
        self.ciou = ciou
        self.BCE = nn.BCEWithLogitsLoss()
        self.edge_with_logits1 = EdgeHoldLoss().cuda()
        self.ghm_loss = GHMC_Loss(bins=10, alpha=0.5)
        self.focal_loss = BinaryFocalLoss()
        self.ssim_loss = SSIMLoss()
        self.logssim_loss = LOGSSIMLoss()
        self.ssim_f = SSIM(size_average=False)
        self.ssim_t = SSIM(size_average=True)


    def forward(self, inputs, targets):
        
        out1, out2, out3, out4, out5 = inputs
        semantic_gt = targets
        semantic_gt = torch.unsqueeze(semantic_gt, 1).float()
     
        loss1 = self.IOU(torch.sigmoid(out1), semantic_gt) + self.BCE(out1, semantic_gt) #+ self.edge_with_logits1(out1,semantic_gt)
        loss2 = self.IOU(torch.sigmoid(out2), semantic_gt) + self.BCE(out2, semantic_gt) #+ self.edge_with_logits1(out2,semantic_gt)
        loss3 = self.IOU(torch.sigmoid(out3), semantic_gt) + self.BCE(out3, semantic_gt) #+ self.edge_with_logits1(out3,semantic_gt)
        loss4 = self.IOU(torch.sigmoid(out4), semantic_gt) + self.BCE(out4, semantic_gt) #+ self.edge_with_logits(out4,semantic_gt)
        loss5 = self.IOU(torch.sigmoid(out5), semantic_gt) + self.BCE(out5, semantic_gt) #+ self.edge_with_logits(out5,semantic_gt)

        loss = 2*loss1 + loss2 + loss3 + loss4 + loss5 

        return loss


def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    # model
    model = get_model(cfg)
    device = torch.device(f'cuda:{args.cuda}')
    model.to(device)

    # dataloader
    trainset, testset = get_dataset(cfg)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True)

    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                              pin_memory=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])

    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()
    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])

    best_test = 100

    # amp.register_float_function(torch, 'sigmoid')
    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)

            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)

            with amp.autocast():
                predict = model(image, depth)
                loss = train_criterion(predict, label)

            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()

            reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()
            running_metrics_test.reset()
            test_loss_meter.reset()
            difficult = []
            avg_iou, img_num1 = 0.0, 0.0
            avg_ber, img_num2 = 0.0, 0.0
            avg_mae, img_num3 = 0.0, 0.0

            IOU = pytorch_iou.IOU()
            edge_loss = EdgeHoldLoss()

            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)[0]
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)[0]

                label = torch.unsqueeze(label, 1).float()
                loss = IOU(torch.sigmoid(predict), label) + F.binary_cross_entropy_with_logits(predict, label) + edge_loss(predict, label)
                test_loss_meter.update(loss.item())

                gt1 = gt2 = gt3 = label.cuda()
                pre1 = pre2 = pre3 = torch.sigmoid(predict).cuda()


                #iou
                pre1 = (pre1 >= 0.5)
                gt1 = (gt1 >= 0.5)
                iou = torch.sum((pre1 & gt1)) / torch.sum((pre1 | gt1))
                if iou == iou:  # for Nan
                    avg_iou += iou
                    img_num1 += 1.0

                #ber
                pre2 = (pre2 >= 0.5)
                gt2 = (gt2 >= 0.5)
                N_p = torch.sum(gt2) + 1e-20
                N_n = torch.sum(torch.logical_not(gt2)) + 1e-20  
                TP = torch.sum(pre2 & gt2)
                TN = torch.sum(torch.logical_not(pre2) & torch.logical_not(gt2))
                ber = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

                if ber == ber:  # for Nan
                    avg_ber += ber
                    img_num2 += 1.0

                #mae
                pre3 = torch.where(pre3 >= 0.5, torch.ones_like(pre3), torch.zeros_like(pre3))
                gt3 = torch.where(gt3 >= 0.5, torch.ones_like(gt3), torch.zeros_like(gt3))
                mea = torch.abs(pre3 - gt3).mean()

                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num3 += 1.0

        #iou结果
        avg_iou /= img_num1
        test_iou = avg_iou.item()
        # print(avg_iou.item())

        # ber结果
        avg_ber /= img_num2
        test_ber = avg_ber.item() * 100
        # print(avg_ber.item() * 100)

        # mae结果
        avg_mae /= img_num3
        test_mae = avg_mae.item()
        # print(avg_mae.item())

        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg

        test_avg = (test_ber + test_mae) / 2
        logger.info(
        f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] loss={train_loss:.3f}/{test_loss:.3f}, iou={test_iou:.5f}, ber={test_ber:.5f}, mae={test_mae:.5f}')
    
        if test_avg < best_test and test_iou >= 0.500 and test_ber <= 16.50 and test_mae <= 0.119:
            best_test = test_avg
            save_ckpt(logdir, model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="/home/noone/桌面/HAFNet/configs/mirror.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=0, help="set cuda device id")

    args = parser.parse_args()

    run(args)
