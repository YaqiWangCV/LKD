import matplotlib as mpl
mpl.use('Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    # x = x.to(device)
    return x


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x


def tensor_to_image(x):
    '''Returns an array of shape CxHxW from a given tensor with shape HxWxC'''

    x = np.rollaxis(x.int().detach().cpu().numpy(), 0, 3)
    return x


def plot(image, masks=None, pred_masks=None):
    '''plots for a given image the ground truth mask and the corresponding predicted mask
      masks: tensor of shape (n_tasks, 512, 512)
    '''
    fig, ax = plt.subplots(1, 3, gridspec_kw={'wspace': 0.15, 'hspace': 0.2,
                                              'top': 0.85, 'bottom': 0.1,
                                              'left': 0.05, 'right': 0.95})
                                              

    ax[0].imshow(image.int().permute(1, 2, 0).detach().cpu().numpy())
    # ax[0].imshow(tensor_to_image(image))
    ax[0].axis('off')

    if masks is not None:
        # masks = np.argmax(masks, axis=0)
        ax[1].imshow(masks[0], cmap='gray')
        ax[1].axis('off')

    if pred_masks is not None:
        # pred_masks = np.argmax(pred_masks, axis=0)
        # Thresholding mask
        thresh = 0.1
        prediction = pred_masks[0].detach().cpu().numpy()
        max_prob = np.max(prediction)
        img_pred = np.zeros(prediction.shape)
        img_pred[prediction >= thresh * max_prob] = 1
        ax[2].imshow(img_pred, cmap='gray')
        ax[2].axis('off')

    ax[0].set_title('Original Image')
    ax[1].set_title('Ground Truth')
    ax[2].set_title('Predicted Segmentation map')

    fig.canvas.draw()

    return fig


def AUPR(mask, prediction):
    '''Computes the Area under Precision-Recall Curve for a given ground-truth mask and predicted mask'''
    threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # list of thresholds
    precisions = []
    recalls = []

    for thresh in threshold_list:
        # thresholding the predicted mask
        thresh_pred = np.zeros(prediction.shape)
        thresh_pred[prediction >= thresh] = 1

        # Computing the True and False Positives
        P = np.count_nonzero(mask)
        TP = np.count_nonzero(mask * thresh_pred)
        FP = np.count_nonzero(thresh_pred - (mask * thresh_pred))

        if (P > 0) and (TP + FP > 0):  # avoid division by 0
            precision = TP * 1.0 / (TP + FP)
            recall = TP * 1.0 / P
        else:
            precision = 1
            recall = 0

        # print "precison", precision
        # print "recall", recall

        precisions.append(precision)
        recalls.append(recall)
    precisions.append(1)
    recalls.append(0)

    return auc(recalls, precisions)


def aupr_on_batch(masks, pred):
    '''Computes the mean AUPR over a batch during training'''
    auprs = []
    auprs_old = []
    for i in range(pred.shape[0]):
        prediction = pred[i].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        # 如果全是背景区域，不评估该样本
        all_black = True
        for out in mask.flatten():
            if out:
                all_black = False
        if all_black:
            continue
        auprs_old.append(AUPR(mask, prediction))
    return np.mean(auprs_old)


def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[0]):
        prediction = pred[i].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        all_black = True
        for out in mask.flatten():
            if out:
                all_black = False
        if all_black:
            continue
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)


def L2(f_):
    return (((f_ ** 2).sum(dim=1)) ** 0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(feat.shape[0], feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S)) ** 2) / ((f_T.shape[-1] * f_T.shape[-2]) ** 2) / f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        """inter pair-wise loss from inter feature maps"""
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)

        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                               ceil_mode=True)  # change

        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss
    

# intra-image distillation 
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap

class CriterionIntra(nn.Module):

    def __init__(self, norm_type='channel', divergence='kl', temperature=1.0):

        super(CriterionIntra, self).__init__()

        # define normalize function
        if norm_type == 'channel':
            self.normalize = ChannelNorm()
        elif norm_type == 'spatial':
            self.normalize = nn.Softmax(dim=1)
        elif norm_type == 'channel_mean':
            self.normalize = lambda x: x.view(x.size(0), x.size(1), -1).mean(-1)
        else:
            self.normalize = None
        self.norm_type = norm_type

        self.temperature = 1.0

        # define loss function
        if divergence == 'mse':
            self.criterion = nn.MSELoss(reduction='sum')
        elif divergence == 'kl':
            self.criterion = nn.KLDivLoss(reduction='sum')
            self.temperature = temperature
        self.divergence = divergence

    def forward(self, preds_S, preds_T):
        if preds_S.shape[2] != preds_T.shape[2]:
            preds_S = F.interpolate(preds_S, preds_T.size()[-2:], mode='bilinear')

        n, c, h, w = preds_S.shape
        if self.normalize is not None:
            norm_s = self.normalize(preds_S / self.temperature)
            norm_t = self.normalize(preds_T.detach() / self.temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        if self.divergence == 'kl':
            norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)
