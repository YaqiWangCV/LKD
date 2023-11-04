import logging
import math
import os
import time
import warnings
import numpy as np
import torch.optim
from tensorboardX import SummaryWriter
from torch import nn
import argparse
from losses import Losses_mcau, Losses_u2net, Losses_unet, Loss_unetpp, Loss_unet3p
from DataSet import load_train_val_data, load_ddr_train_val
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from nets import U2Net, UNet, unetpp, ENet
from nets.CAUNet import CAUNet
from nets.UCTransNet import UCTransNet, get_CTranS_config
from nets.UDTransNet import UDTransNet, get_model_config
from utils import auc_on_batch, aupr_on_batch, plot
from nets.attention_unet import AttU_Net
from nets.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from nets.UNet3Plus import UNet3Plus_DeepSup
from nets.attention_unet import AttU_Net
from nets.res_unet_plus import ResUnetPlusPlus

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
warnings.filterwarnings("ignore")
os.environ['QTQPAPLATFORM'] = 'offscreen'


def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr


def print_summary_whole(epoch, i, nb_batch, loss, loss_name, batch_time,
                        average_loss, average_time, mode, lr, ex, ex_mean, he, he_mean, ma, ma_mean, se, se_mean):
    summary = '   [' + str(mode) + '] Epoch: [{0}][{1}/{2}]  '.format(
        epoch, i, nb_batch)
    string = ''
    string += '{}: {:.3f} '.format(loss_name, loss)
    string += '(Avg {:.4f}) '.format(average_loss)
    if not math.isnan(ex):
        string += 'EX_AUPR {:.3f} '.format(ex)
        string += '(Avg {:.4f}) '.format(ex_mean)
    if not math.isnan(he):
        string += 'HE_AUPR {:.3f} '.format(he)
        string += '(Avg {:.4f}) '.format(he_mean)
    if not math.isnan(ma):
        string += 'MA_AUPR {:.3f} '.format(ma)
        string += '(Avg {:.4f}) '.format(ma_mean)
    if not math.isnan(se):
        string += 'SE_AUPR {:.3f} '.format(se)
        string += '(Avg {:.4f}) '.format(se_mean)
    if mode == 'Train':
        string += 'LR {:.2e}   '.format(lr)
    string += 'Time {:.1f} '.format(batch_time)
    string += '(Avg {:.1f})   '.format(average_time)
    summary += string
    logger.info(summary)


def save_checkpoint(state, save_path):
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type
    loss = state['loss']  # loss name

    if best_model:
        filename = save_path + '/' + \
                   'best_model.{}--{}.pth.tar'.format(loss, model)
    else:
        filename = save_path + '/' + \
                   'model.{}--{}--{:02d}.pth.tar'.format(loss, model, epoch)

    torch.save(state, filename)


# Train One Epoch
def train_one_epoch(loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type, lbtw_algorithm,
                    batch_size):
    logging_mode = 'Train' if model.training else 'Val'
    end = time.time()
    time_sum, loss_sum = 0, 0
    EX_auc, EX_aupr = [], []
    HE_auc, HE_aupr = [], []
    MA_auc, MA_aupr = [], []
    SE_auc, SE_aupr = [], []
    for (i, sample) in enumerate(loader, 1):

        images, masks = sample['image'], sample['masks'].permute(0, 2, 3, 1)

        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        images = images.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()

        if model_type == 'u2net':
            if model.training:
                optimizer.zero_grad()
                output, d1, d2, d3, d4, d5, d6 = model(images)
                out_loss, loss = criterion(output, d1, d2, d3, d4, d5, d6, masks)
                loss.backward()
                optimizer.step()
            else:
                output, d1, d2, d3, d4, d5, d6 = model(images)
                out_loss, loss = criterion(output, d1, d2, d3, d4, d5, d6, masks)
        elif model_type == 'mcaunet':
            output, pred_comb2, pred_comb3, pred_comb1 = model(images)
            losses, out_loss, c3_loss, c2_loss, c1_loss = criterion(output, pred_comb3, pred_comb2, pred_comb1, masks)
            if model.training:
                loss, w0, w3, w2, w1 = lbtw_algorithm(i, out_loss, c3_loss, c2_loss, c1_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif model_type == 'unet3p':
            outputs = model(images)
            output, pred_comb3, pred_comb2, pred_comb1, pred_comb0 = outputs[0], outputs[1], outputs[2], outputs[3], \
                                                                     outputs[4]
            losses, out_loss, c3_loss, c2_loss, c1_loss, c0_loss = criterion(output, pred_comb3, pred_comb2, pred_comb1,
                                                                             pred_comb0, masks)
            if model.training:
                loss, w0, w3, w2, w1, w0 = lbtw_algorithm(i, out_loss, c3_loss, c2_loss, c1_loss, c0_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        elif model_type == 'unet++':
            outputs = model(images)
            output, pred_comb3, pred_comb2, pred_comb1 = outputs[-1], outputs[-2], outputs[-3], outputs[-4]
            losses, out_loss, c3_loss, c2_loss, c1_loss = criterion(output, pred_comb3, pred_comb2, pred_comb1, masks)
            if model.training:
                loss, w0, w3, w2, w1 = lbtw_algorithm(i, out_loss, c3_loss, c2_loss, c1_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        elif model_type == 'unet' or 'uctransnet' or 'udtransnet' or 'attenunet' or 'resunetpp' or 'transunet' or 'enet':
            output = model(images)
            loss = criterion(output, masks)
            out_loss = loss
            if model.training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # auc aupr for single lesion
        EX_masks, EX_output = masks[..., 1], output[..., 1]
        HE_masks, HE_output = masks[..., 2], output[..., 2]
        MA_masks, MA_output = masks[..., 3], output[..., 3]
        SE_masks, SE_output = masks[..., 4], output[..., 4]
        EX_cur_auc, EX_cur_aupr = auc_on_batch(EX_masks, EX_output), aupr_on_batch(EX_masks, EX_output)
        HE_cur_auc, HE_cur_aupr = auc_on_batch(HE_masks, HE_output), aupr_on_batch(HE_masks, HE_output)
        MA_cur_auc, MA_cur_aupr = auc_on_batch(MA_masks, MA_output), aupr_on_batch(MA_masks, MA_output)
        SE_cur_auc, SE_cur_aupr = auc_on_batch(SE_masks, SE_output), aupr_on_batch(SE_masks, SE_output)
        if not math.isnan(EX_cur_auc):
            EX_auc.append(EX_cur_auc), EX_aupr.append(EX_cur_aupr)
        if not math.isnan(HE_cur_auc):
            HE_auc.append(HE_cur_auc), HE_aupr.append(HE_cur_aupr)
        if not math.isnan(MA_cur_auc):
            MA_auc.append(MA_cur_auc), MA_aupr.append(MA_cur_aupr)
        if not math.isnan(SE_cur_auc):
            SE_auc.append(SE_cur_auc), SE_aupr.append(SE_cur_aupr)

        # measure elapsed time
        batch_time = time.time() - end
        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss

        if i == len(loader):
            average_loss = loss_sum / (batch_size * (i - 1) + len(images))
            average_time = time_sum / (batch_size * (i - 1) + len(images))

        else:
            average_loss = loss_sum / (i * batch_size)
            average_time = time_sum / (i * batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % print_frequency == 0:
            print_summary_whole(epoch + 1, i, len(loader), out_loss, loss_name, batch_time,
                                average_loss, average_time, logging_mode,
                                min(g["lr"] for g in optimizer.param_groups),
                                EX_cur_aupr, np.mean(EX_aupr), HE_cur_aupr, np.mean(HE_aupr),
                                MA_cur_aupr, np.mean(MA_aupr), SE_cur_aupr, np.mean(SE_aupr))

        if tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, out_loss.item(), step)

            # plot metrics in tensorboard
            if not math.isnan(EX_cur_auc):
                writer.add_scalar(logging_mode + '_EX_auc', EX_cur_auc, step)
                writer.add_scalar(logging_mode + '_EX_aupr', EX_cur_aupr, step)
            if not math.isnan(HE_cur_auc):
                writer.add_scalar(logging_mode + '_HE_auc', HE_cur_auc, step)
                writer.add_scalar(logging_mode + '_HE_aupr', HE_cur_aupr, step)
            if not math.isnan(MA_cur_auc):
                writer.add_scalar(logging_mode + '_MA_auc', MA_cur_auc, step)
                writer.add_scalar(logging_mode + '_MA_aupr', MA_cur_aupr, step)
            if not math.isnan(SE_cur_auc):
                writer.add_scalar(logging_mode + '_SE_auc', SE_cur_auc, step)
                writer.add_scalar(logging_mode + '_SE_aupr', SE_cur_aupr, step)

        torch.cuda.empty_cache()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return np.mean(EX_aupr), np.mean(HE_aupr), np.mean(MA_aupr), np.mean(SE_aupr)


def main_loop(batch_size, model_type, tensorboard=True, task_name=None, dataset=None):
    # Load train and val data
    global model, criterion
    tasks = task_name
    n_labels = len(tasks) + 1
    lr = learning_rate
    n_channels = 3

    if dataset == 'idrid':
        train_loader, val_loader = load_train_val_data(batch_size=batch_size)
    elif dataset == 'ddr':
        train_loader, val_loader = load_ddr_train_val(batch_size=batch_size)

    if model_type == 'u2net':
        model = U2Net.U2NET(n_channels, n_labels)
        criterion = Losses_u2net.LBTW_Loss(Losses_u2net.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'mcaunet':
        model = CAUNet(n_channels, n_labels)
        criterion = Losses_mcau.LBTW_Loss(Losses_mcau.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Choose optimize
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Losses_mcau.LBTW_algorithm()

    elif model_type == 'unet':
        model = UNet.UNet(n_channels, n_labels)
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'enet':
        model = ENet.ENet(n_labels)
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'unet++':
        model = unetpp.NestedUNet(n_channels, n_labels, deepsupervision=True)
        criterion = Loss_unetpp.LBTW_Loss(Loss_unetpp.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Loss_unetpp.LBTW_algorithm()

    elif model_type == 'uctransnet':
        config = get_CTranS_config()
        model = UCTransNet(config, n_channels, n_labels, img_size=640)
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'udtransnet':
        config = get_model_config()
        model = UDTransNet(config, n_channels, n_labels, img_size=640)
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'attenunet':
        model = AttU_Net(n_channels, n_labels)
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'resunetpp':
        model = ResUnetPlusPlus(n_channels)
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    elif model_type == 'unet3p':
        model = UNet3Plus_DeepSup(n_channels, n_labels)
        criterion = Loss_unet3p.LBTW_Loss(Loss_unet3p.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)
        lbtw_algorithm = Loss_unet3p.LBTW_algorithm()

    elif model_type == 'transunet':
        parser = argparse.ArgumentParser()
        parser.add_argument('--img_size', type=int,
                            default=640, help='input patch size of network input')
        parser.add_argument('--n_skip', type=int,
                            default=3, help='using number of skip-connect, default is num')
        parser.add_argument('--vit_name', type=str,
                            default='R50-ViT-B_16', help='select one vit model')
        parser.add_argument('--vit_patches_size', type=int,
                            default=16, help='vit_patches_size, default is 16')
        args = parser.parse_args()
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = 5
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            config_vit.patches.grid = (
                int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)
        model.load_from(weights=np.load(config_vit.pretrained_path))
        criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        lbtw_algorithm = Losses_u2net.LBTW_algorithm()

    model = nn.DataParallel(model)
    model = model.to(device)
    if checkpoint:
        checkpoints = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])
        print("Model Loaded!")
    print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    loss_name = criterion._get_name()

    if tensorboard:
        log_dir = tensorboard_folder + session_name + '/'
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

    EX_best, HE_best, MA_best, SE_best, mean_best = 0, 0, 0, 0, 0
    EX_best_epoch, HE_best_epoch, MA_best_epoch, SE_best_epoch, mean_best_epoch = 0, 0, 0, 0, 0
    logger.info('Train Start')
    for epoch in range(epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, epochs))
        logger.info(session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))

        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, lr_scheduler, model_type,
                        lbtw_algorithm,
                        batch_size=batch_size)

        if epoch >= 100:
            with torch.no_grad():
                model.eval()
                ex_aupr, he_aupr, ma_aupr, se_aupr = train_one_epoch(val_loader, model, criterion, optimizer,
                                                                     writer,
                                                                     epoch, None,
                                                                     model_type, lbtw_algorithm,
                                                                     batch_size=batch_size)
            save_flag = False
            if ex_aupr > EX_best:
                EX_best = ex_aupr
                EX_best_epoch = epoch
                save_flag = True
            if he_aupr > HE_best:
                HE_best = he_aupr
                HE_best_epoch = epoch
                save_flag = True
            if ma_aupr > MA_best:
                MA_best = ma_aupr
                MA_best_epoch = epoch
                save_flag = True
            if se_aupr > SE_best:
                SE_best = se_aupr
                SE_best_epoch = epoch
                save_flag = True
            current_mean = (ex_aupr + he_aupr + ma_aupr + se_aupr) / 4
            if current_mean > mean_best:
                mean_best = current_mean
                mean_best_epoch = epoch
                save_flag = True

            logger.info('Best model: EX:{} \t HE:{} \t MA:{} \t SE:{} \n mAUPR:{}'.format(EX_best_epoch, HE_best_epoch,
                                                                                          MA_best_epoch, SE_best_epoch,
                                                                                          mean_best_epoch))
            logger.info(
                'Best AUPR: EX:{} \t HE:{} \t MA:{} \t SE:{} \n mAUPR:{}'.format(EX_best, HE_best, MA_best, SE_best,
                                                                                 mean_best))

            if (save_flag and epoch >= 250) or (save_flag and checkpoint):
                save_checkpoint({'epoch': epoch,
                                 'best_model': False,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': '123',
                                 'loss': loss_name,
                                 'optimizer': optimizer.state_dict()}, model_path)


if __name__ == '__main__':

    task_names = ['EX', 'MA', 'HE', 'SE']
    # opitons: ['mcaunet', 'u2net', 'unet', 'unet++', 'uctransnet', 'udtransnet', 'attenunet', 'resunetpp', 'unet3p', 'transunet', 'enet']
    model_name = 'unet'  
    dataset = 'idrid' # idrid or ddr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = None
    learning_rate = 5e-3
    epochs = 400
    batch_size = 2
    print_frequency = 1
    save_frequency = 10
    save_model = True
    tensorboard = True
    save_path = os.path.join('./single_network_log', model_name)
    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
    model_path = save_path + 'models/' + session_name + '/'
    tensorboard_folder = save_path + 'tensorboard_logs/'
    logger_path = save_path + 'log_file/'
    if not os.path.isdir(logger_path):
        os.makedirs(logger_path)

    logger_path = save_path + 'log_file/' + session_name + ".log"

    logger = logger_config(log_path=logger_path)

    main_loop(model_type=model_name, tensorboard=True,
              task_name=task_names, batch_size=batch_size, dataset=dataset)
