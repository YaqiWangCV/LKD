import os
import math
import numpy as np
import torch.optim
from torch import nn
from DataSet import load_ddr_train_val, load_train_val_data
from nets import U2Net, UNet, unetpp
from nets.CAUNet import CAUNet
from utils import auc_on_batch, aupr_on_batch

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def dice(pred, truth, smooth=1e-5):
    batch_size = len(pred)
    p = pred.view(batch_size,-1)
    t = truth.view(batch_size,-1)
    p = p.cpu().detach().numpy()
    t = t.cpu().detach().numpy()
    intersection = (p * t).sum(-1)
    union = (p * p).sum(-1) + (t * t).sum(-1)
    dice = (2*intersection + smooth) / (union +smooth)   
    return dice

def evaluate():
    EX_auc, EX_aupr, EX_dice = [], [], []
    HE_auc, HE_aupr, HE_dice = [], [], []
    MA_auc, MA_aupr, MA_dice = [], [], []
    SE_auc, SE_aupr, SE_dice = [], [], []
    for (i, sample) in enumerate(val_loader, 1):
        images, masks = sample['image'], sample['masks'].permute(0, 2, 3, 1)
        img = images[0].permute(1, 2, 0).detach().numpy()
        # Take variable and put them to GPU
        images = images.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        if model_type == 'u2net':
            output, d1, d2, d3, d4, d5, d6 = model(images)
        elif model_type == 'mcaunet':
            # output, _, _, _, feature = model(images)
            features, output = model(images, all_features=True)
        elif model_type == 'unet' or 'uctransnet' or 'udtransnet':
            # output = model(images)
            features, output = model(images, all_features=True)
        elif model_type == 'unet++':
            outputs = model(images)
            output, pred_comb3, pred_comb2, pred_comb1 = outputs[-1], outputs[-2], outputs[-3], outputs[-4]

        # output
        EX_masks, EX_output = masks[..., 1], output[..., 1]
        HE_masks, HE_output = masks[..., 2], output[..., 2]
        MA_masks, MA_output = masks[..., 3], output[..., 3]
        SE_masks, SE_output = masks[..., 4], output[..., 4]

        # auc aupr
        EX_cur_auc, EX_cur_aupr, EX_cur_dice = auc_on_batch(EX_masks, EX_output), aupr_on_batch(EX_masks, EX_output), dice(EX_masks, EX_output)
        HE_cur_auc, HE_cur_aupr, HE_cur_dice  = auc_on_batch(HE_masks, HE_output), aupr_on_batch(HE_masks, HE_output), dice(HE_masks, HE_output)
        MA_cur_auc, MA_cur_aupr, MA_cur_dice  = auc_on_batch(MA_masks, MA_output), aupr_on_batch(MA_masks, MA_output), dice(MA_masks, MA_output)
        SE_cur_auc, SE_cur_aupr, SE_cur_dice  = auc_on_batch(SE_masks, SE_output), aupr_on_batch(SE_masks, SE_output), dice(SE_masks, SE_output)

        if not math.isnan(EX_cur_auc):
            EX_auc.append(EX_cur_auc), EX_aupr.append(EX_cur_aupr), EX_dice.append(EX_cur_dice)
        if not math.isnan(HE_cur_auc):
            HE_auc.append(HE_cur_auc), HE_aupr.append(HE_cur_aupr), HE_dice.append(HE_cur_dice)
        if not math.isnan(MA_cur_auc):
            MA_auc.append(MA_cur_auc), MA_aupr.append(MA_cur_aupr), MA_dice.append(MA_cur_dice)
        if not math.isnan(SE_cur_auc):
            SE_auc.append(SE_cur_auc), SE_aupr.append(SE_cur_aupr), SE_dice.append(SE_cur_dice)
        
        torch.cuda.empty_cache()

    return np.mean(EX_aupr), np.mean(HE_aupr), np.mean(MA_aupr), np.mean(SE_aupr), np.mean(EX_dice), np.mean(HE_dice), np.mean(MA_dice), np.mean(SE_dice)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    tasks = ['EX', 'MA', 'HE', 'SE']
    model_type = 'unet'
    checkpoints = None  # /path/to/checkpoints

    n_labels = len(tasks) + 1
    n_channels = 3 

    train_loader, val_loader = load_train_val_data(batch_size=1)   # for idrid
    # train_loader, val_loader = load_ddr_train_val(batch_size=1)    # for ddr

    if model_type == 'u2net':
        model = U2Net.U2NET(n_channels, n_labels)
    elif model_type == 'mcaunet':
        model = CAUNet(n_channels, n_labels)
    elif model_type == 'unet':
        model = UNet.UNet(n_channels, n_labels)
    elif model_type == 'unet++':
        model = unetpp.NestedUNet(n_channels, n_labels, deepsupervision=True)

    model = nn.DataParallel(model)
    model = model.to(device)
    checkpoints = torch.load(checkpoints[model_type], map_location=device)
    model.load_state_dict(checkpoints['state_dict'], strict=False)

    print("Model Loaded!")
    print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    res = evaluate()
    print("AUPR EX:{}, HE:{}, MA:{}, SE:{}".format(res[0], res[1], res[2], res[3]))
    print("Dice EX:{}, HE:{}, MA:{}, SE:{}".format(res[4], res[5], res[6], res[7]))

