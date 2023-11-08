import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss


class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.4, 0.6]):
        super(WeightedBCE, self).__init__()
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
    def __init__(self, weights=[0.2, 0.8]):  # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.contiguous().view(batch_size, -1)
        truth = truth.contiguous().view(batch_size, -1)
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
    def __init__(self, dice_weight=1, BCE_weight=0.5):
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

    def forward(self, out, comb3, comb2, comb1, target):
        outs = [out, comb3, comb2, comb1]
        sub_loss = []
        for i in range(4):
            pred = outs[i]
            # 多任务学习
            curr_loss = 0
            for index in range(1, 5):
                curr_loss += self.criteria(pred[..., index], target[..., index])
            sub_loss.append(curr_loss / 4)
        # print sub_loss[0].data, sub_loss[1].data, sub_loss[2].data, sub_loss[3].data
        return sub_loss, sub_loss[0], sub_loss[1], sub_loss[2], sub_loss[3]


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

    def __call__(self, batch_num, out_loss, c3_loss, c2_loss, c1_loss, alpha=0.5):
        if batch_num == 1:
            self.initial_taskout_loss_list = out_loss.item()
            self.initial_task3_loss_list = c3_loss.item()
            self.initial_task2_loss_list = c2_loss.item()
            self.initial_task1_loss_list = c1_loss.item()
        out_loss_ratio = out_loss.item() / self.initial_taskout_loss_list
        c3_loss_ratio = c3_loss.item() / self.initial_task3_loss_list
        c2_loss_ratio = c2_loss.item() / self.initial_task2_loss_list
        c1_loss_ratio = c1_loss.item() / self.initial_task1_loss_list

        out_loss_weight = pow(out_loss_ratio, alpha)
        c3_loss_weight = pow(c3_loss_ratio, alpha)
        c2_loss_weight = pow(c2_loss_ratio, alpha)
        c1_loss_weight = pow(c1_loss_ratio, alpha)

        weights_sum = out_loss_weight + c3_loss_weight + c2_loss_weight + c1_loss_weight
        out_loss_weight = out_loss_weight / weights_sum * 4
        c3_loss_weight = c3_loss_weight / (weights_sum) * 4
        c2_loss_weight = c2_loss_weight / weights_sum * 4
        c1_loss_weight = c1_loss_weight / weights_sum * 4

        self.weights_out_save.append(out_loss_weight)
        self.weights_c3_save.append(c3_loss_weight)
        self.weights_c2_save.append(c2_loss_weight)
        self.weights_c1_save.append(c1_loss_weight)

        w0 = self.weights_out_save
        w3 = self.weights_c3_save
        w2 = self.weights_c2_save
        w1 = self.weights_c1_save
        losses = out_loss * out_loss_weight \
                 + c3_loss * c3_loss_weight \
                 + c2_loss * c2_loss_weight \
                 + c1_loss * c1_loss_weight
        return losses / 4.0, w0, w3, w2, w1


# MultiClass Loss
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0

        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class MultiClassDiceCE(nn.Module):
    def __init__(self, num_classes=9):
        super(MultiClassDiceCE, self).__init__()
        self.CE_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.CE_weight = 0.5
        self.dice_weight = 0.5

    def _show_dice(self, inputs, targets, softmax=True):
        dice = 1.0 - self.dice_loss(inputs, targets, softmax=softmax)
        return dice

    def forward(self, inputs, targets, softmax=True):
        dice = self.dice_loss(inputs, targets, softmax=softmax)
        CE = self.CE_loss(inputs, targets)
        dice_CE_loss = self.dice_weight * dice + self.CE_weight * CE
        return dice_CE_loss


class LBTW_Loss_Multiclass(nn.Module):

    def __init__(self, Loss):
        super(LBTW_Loss_Multiclass, self).__init__()
        self.criteria = Loss
        try:
            self.__name__ = "LBTW_" + self.criteria.__name__
        except:
            self.__name__ = "LBTW_" + self.criteria._get_name()

    def _get_name(self):
        super(LBTW_Loss_Multiclass, self)._get_name()
        return "LBTW_" + self.criteria._get_name()

    def forward(self, out, comb3, comb2, comb1, target):
        outs = [out, comb3, comb2, comb1]
        sub_loss = []
        for i in range(4):
            sub_loss.append(self.criteria(outs[i], target))
        return sub_loss, sub_loss[0], sub_loss[1], sub_loss[2], sub_loss[3]
