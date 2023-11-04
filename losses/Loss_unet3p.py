# -*- coding: utf-8 -*-
# @Time    : 2021/3/11 10:17 下午
# @Author  : Haonan Wang
# @File    : Losses.py
# @Software: PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class WeightedBCE(nn.Module):

    def __init__(self, weights=None):
        super(WeightedBCE, self).__init__()
        if weights is None:
            weights = [0.2, 0.8]
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.contiguous().view(-1)
        truth = truth_pixel.contiguous().view(-1)
        assert (logit.shape == truth.shape)
        loss = F.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth > 0.5).float()
        neg = (truth < 0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0] * pos * loss / pos_weight + self.weights[1] * neg * loss / neg_weight).sum()

        return loss


class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=None):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        if weights is None:
            weights = [0.2, 0.8]
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (logit.shape == truth.shape)
        p = logit.view(batch_size, -1)
        t = truth.view(batch_size, -1)
        w = truth.detach()
        w = w * (self.weights[1] - self.weights[0]) + self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w * (p)
        t = w * (t)
        intersection = (p * t).sum(-1)
        union = (p * p).sum(-1) + (t * t).sum(-1)
        dice = 1 - (2 * intersection + smooth) / (union + smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss


class WeightedDiceBCE(nn.Module):
    def __init__(self, dice_weight=0.7, BCE_weight=0.3):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE
        # print "dice_focal",dice_focal_loss
        return dice_BCE_loss


class LBTW_Loss(nn.Module):

    def __init__(self, Loss):
        super(LBTW_Loss, self).__init__()
        self.criteria = Loss
        try:
            self.__name__ = "LBTW_" + self.criteria.__name__
        except:
            self.__name__ = "LBTW_" + self.criteria._get_name()

    def _get_name(self):
        super(LBTW_Loss, self)._get_name()
        return "LBTW_" + self.criteria._get_name()

    def forward(self, out, comb3, comb2, comb1, comb0, target):
        outs = [out, comb3, comb2, comb1, comb0]
        sub_loss = []
        for i in range(len(outs)):
            pred = outs[i]
            # 多任务学习
            curr_loss = 0
            for index in range(5):
                curr_loss += self.criteria(pred[..., index], target[..., index])
            # 单任务
            # curr_loss = self.criteria(pred, target)
            sub_loss.append(curr_loss / 5)
        # print sub_loss[0].data, sub_loss[1].data, sub_loss[2].data, sub_loss[3].data
        return sub_loss, sub_loss[0], sub_loss[1], sub_loss[2], sub_loss[3], sub_loss[4]


class LBTW_algorithm():
    def __init__(self):
        self.initial_taskout_loss_list = 0
        self.initial_task3_loss_list = 0
        self.initial_task2_loss_list = 0
        self.initial_task1_loss_list = 0
        self.weights_out_save = []
        self.weights_c3_save = []
        self.weights_c2_save = []
        self.weights_c1_save = []
        self.weights_c0_save = []

    def __call__(self, batch_num, out_loss, c3_loss, c2_loss, c1_loss, c0_loss, alpha=0.5):
        if batch_num == 1:
            self.initial_taskout_loss_list = out_loss.item()
            self.initial_task3_loss_list = c3_loss.item()
            self.initial_task2_loss_list = c2_loss.item()
            self.initial_task1_loss_list = c1_loss.item()
            self.initial_task0_loss_list = c0_loss.item()

        out_loss_ratio = out_loss.item() / self.initial_taskout_loss_list
        c3_loss_ratio = c3_loss.item() / self.initial_task3_loss_list
        c2_loss_ratio = c2_loss.item() / self.initial_task2_loss_list
        c1_loss_ratio = c1_loss.item() / self.initial_task1_loss_list
        c0_loss_ratio = c0_loss.item() / self.initial_task0_loss_list

        out_loss_weight = pow(out_loss_ratio, alpha)
        c3_loss_weight = pow(c3_loss_ratio, alpha)
        c2_loss_weight = pow(c2_loss_ratio, alpha)
        c1_loss_weight = pow(c1_loss_ratio, alpha)
        c0_loss_weight = pow(c0_loss_ratio, alpha)

        weights_sum = out_loss_weight + c3_loss_weight + c2_loss_weight + c1_loss_weight + c0_loss_weight
        out_loss_weight = out_loss_weight / weights_sum * 4
        c3_loss_weight = c3_loss_weight / weights_sum * 4
        c2_loss_weight = c2_loss_weight / weights_sum * 4
        c1_loss_weight = c1_loss_weight / weights_sum * 4
        c0_loss_weight = c0_loss_weight / weights_sum * 4

        self.weights_out_save.append(out_loss_weight)
        self.weights_c3_save.append(c3_loss_weight)
        self.weights_c2_save.append(c2_loss_weight)
        self.weights_c1_save.append(c1_loss_weight)
        self.weights_c0_save.append(c0_loss_weight)

        w_out = self.weights_out_save
        w3 = self.weights_c3_save
        w2 = self.weights_c2_save
        w1 = self.weights_c1_save
        w0 = self.weights_c0_save
        losses = out_loss * out_loss_weight \
                 + c3_loss * c3_loss_weight \
                 + c2_loss * c2_loss_weight \
                 + c1_loss * c1_loss_weight \
                 + c0_loss * c0_loss_weight
        return losses / 5.0, w_out, w3, w2, w1, w0
