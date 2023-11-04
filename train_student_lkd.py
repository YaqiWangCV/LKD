import os
import math
import time
import logging
import warnings
import torch.optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from nets.CAUNet import CAUNet
from nets import U2Net, UNet, unetpp, ENet
from nets.attention_unet import AttU_Net
from losses import Losses_unet
from DataSet import load_train_val_data, load_ddr_train_val
from utils import auc_on_batch, aupr_on_batch, CriterionIntra
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from InterLKD import InterLKD

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings("ignore")

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
    val_loss = state['val_loss']  # loss value
    model = state['model']  # model type
    loss = state['loss']  # loss name

    if best_model:
        filename = save_path + '/' + \
                   'best_model.{}--{}.pth.tar'.format(loss, model)
    else:
        filename = save_path + '/' + \
                   'model.{}--{}--{:02d}.pth.tar'.format(loss, model, epoch)

    torch.save(state, filename)

def train_one_epoch(loader, student, teacher, criterion, optimizer, writer, epoch, lr_scheduler,
                    criterion_cwd, criterion_memory_contrast,
                    batch_size, student_type):
    
    logging_mode = 'Train' if student.training else 'Val'

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
        images = images.type(torch.FloatTensor)
        masks = masks.type(torch.FloatTensor)

        if torch.cuda.is_available():
            images, masks = Variable(images.cuda(), requires_grad=False), Variable(masks.cuda(), requires_grad=False)
        else:
            images, masks = Variable(images, requires_grad=False), Variable(masks, requires_grad=False)

        # teacher output
        with torch.no_grad():
            teacher_features, teacher_ouputs = teacher(images, all_features=True)

        # student output
        student_features, student_output = student(images, all_features=True)

        # Inter-LKD
        if student.training:
            targets, teacher_out, student_out = masks.permute(0, 3, 1, 2), teacher_ouputs.permute(0, 3, 1, 2), student_output.permute(0, 3, 1, 2)
            inter_loss = criterion_memory_contrast(student_features[-1], teacher_features[-1].detach(), targets, teacher_out.detach(), student_out.detach(), epoch=epoch)
        else:
            inter_loss = 0

        # Fill the lesion embedding queue in the first 10mini-batches without updating the model parameters.
        if epoch < 10:
            print("warmup:", epoch)
            continue

        # hard loss
        hard_loss = criterion(student_output, masks)

        # Intra-LKD
        intra_prediction = criterion_cwd(student_output.permute(0, 3, 1, 2), teacher_ouputs.permute(0, 3, 1, 2))

        # Inter-LKD
        if student == 'enet':
            intra_intermidiate = criterion_cwd(student_features[-1], teacher_features[-1]) + \
                        criterion_cwd(student_features[0], teacher_features[0]) + \
                        criterion_cwd(student_features[2], teacher_features[2]) + \
                        criterion_cwd(student_features[4], teacher_features[4]) 
        else:
            intra_intermidiate = criterion_cwd(student_features[-1], teacher_features[-1]) + \
                                criterion_cwd(student_features[0], teacher_features[0]) + \
                                criterion_cwd(student_features[2], teacher_features[2]) + \
                                criterion_cwd(student_features[4], teacher_features[4])

        intra_loss = 25 * intra_prediction + 0.07 * intra_intermidiate
        loss = hard_loss + intra_loss + 0.01* inter_loss
        
        if student.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        output = student_output
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
        loss_sum += len(images) * loss

        if i == len(loader):
            average_loss = loss_sum / (batch_size * (i - 1) + len(images))
            average_time = time_sum / (batch_size * (i - 1) + len(images))

        else:
            average_loss = loss_sum / (i * batch_size)
            average_time = time_sum / (i * batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % print_frequency == 0:
            print_summary_whole(epoch + 1, i, len(loader), loss, loss_name, batch_time,
                                average_loss, average_time, logging_mode,
                                min(g["lr"] for g in optimizer.param_groups),
                                EX_cur_aupr, np.mean(EX_aupr), HE_cur_aupr, np.mean(HE_aupr),
                                MA_cur_aupr, np.mean(MA_aupr), SE_cur_aupr, np.mean(SE_aupr))

        if tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, loss.item(), step)

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


def main_loop(batch_size, model_type, tensorboard=True, task_name=None, student_type=None, dataset=None):

    tasks = task_name
    n_labels = len(tasks) + 1
    n_channels = 3 

    if dataset == 'idrid':
        train_loader, val_loader = load_train_val_data(batch_size=batch_size)
    elif dataset == 'ddr':
        train_loader, val_loader = load_ddr_train_val(batch_size=batch_size)

    # load teacher model
    if model_type == 'u2net':
        teacher = U2Net.U2NET(n_channels, n_labels)
    elif model_type == 'mcaunet':
        teacher = CAUNet(n_channels, n_labels)
    elif model_type == 'unet++':
        teacher = unetpp.NestedUNet(n_channels, n_labels, deepsupervision=True)
    teacher = nn.DataParallel(teacher)
    teacher = teacher.to(device)
    # teacher_pretrain = torch.load(teacher_checkpoints, map_location=device)
    # teacher.load_state_dict(teacher_pretrain['state_dict'], strict=False)
    print("Successfully loaded teacher model!")
    print("Let's use {0} GPUs!".format(torch.cuda.device_count()))

    # create student model
    if student_type == 'unet':
        print('student unet')
        student = UNet.UNet(n_channels, n_labels)
    elif student_type == 'attentionunet':
        print('student attentionunet')
        student = AttU_Net(n_channels, n_labels)
    elif student_type == 'enet':
        print('student enet')
        student = ENet.ENet(n_labels)
    student = nn.DataParallel(student)
    student = student.to(device)
    if checkpoint is not None:
        student_pretrain = torch.load(checkpoint, map_location=device)
        student.load_state_dict(student_pretrain['state_dict'], strict=False)
        print("Successfully loaded student model!")

    # Segmentation loss
    criterion = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).to(device)
    loss_name = criterion._get_name()
    # Intra loss
    criterion_intra = CriterionIntra().to(device)
    # Inter LKD Module
    criterion_memory_contrast = InterLKD(num_classes=3, queue_size=4096, contrast_size=512, contrast_kd_temperature=1.0, contrast_temperature=0.1)

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=5e-5)

    if tensorboard:
        log_dir = tensorboard_folder + session_name + '/'
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

    logger.info('Train Start')
    logger.info('Training with batch size : {}'.format(batch_size))
    EX_best, HE_best, MA_best, SE_best, mean_best = 0, 0, 0, 0, 0
    EX_best_epoch, HE_best_epoch, MA_best_epoch, SE_best_epoch, mean_best_epoch = 0, 0, 0, 0, 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, epochs))
        logger.info(session_name)
        student.train()
        teacher.eval()
        train_one_epoch(train_loader, student, teacher, criterion, optimizer, writer, epoch, lr_scheduler, criterion_intra,
                        criterion_memory_contrast, batch_size=batch_size, student_type=student_type)

        if epoch >= 0:

            with torch.no_grad():
                student.eval()
                ex_aupr, he_aupr, ma_aupr, se_aupr = train_one_epoch(val_loader, student, teacher,
                                                                     criterion, optimizer, writer, epoch, None, criterion_intra,
                                                                     criterion_memory_contrast, batch_size=batch_size,
                                                                     student_type=student_type)
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
                                 'state_dict': student.state_dict(),
                                 'val_loss': '123',
                                 'loss': loss_name,
                                 'optimizer': optimizer.state_dict()}, model_path)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lr = 5e-3
    epochs = 500
    batch_size = 2
    print_frequency = 1
    save_frequency = 10
    save_model = True
    tensorboard = True
    dataset = 'idrid' # idrid or ddr
    main_path = './idrid/' 
    save_path = './stuent/'
    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
    model_path = save_path + 'models/' + session_name + '/'
    tensorboard_folder = save_path + 'tensorboard_logs/'
    logger_path = save_path + 'log_file/'
    if not os.path.isdir(logger_path):
        os.makedirs(logger_path)

    logger_path = save_path + 'log_file/' + session_name + ".log"

    logger = logger_config(log_path=logger_path)

    task_names = ['EX', 'MA', 'HE', 'SE']
    # opitons: ['mcaunet', 'u2net', 'unet', 'unet++', 'uctransnet', 'attenunet', 'resunetpp', 'unet3p', 'transunet', 'enet']
    teacher = 'mcaunet'     # teacher model 
    student = 'unet'        # student model
    checkpoint = None       # student checkpoints
    # teacher checkpoints
    if teacher == 'mcaunet':
        teacher_checkpoints = "/path/of/mcaunet"
    if teacher == 'unet++':
        teacher_checkpoints = "/path/of/mcaunet"
    if teacher == 'u2net':
        teacher_checkpoints = "/path/of/mcaunet"

    main_loop(student_type=student, model_type=teacher, tensorboard=True,
              task_name=task_names, batch_size=batch_size, dataset=dataset)
