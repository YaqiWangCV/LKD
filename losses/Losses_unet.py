import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import KLDivLoss
from torch.autograd import Variable
import numpy as np


class WeightedBCE(nn.Module):

    def __init__(self, weights=None):
        super(WeightedBCE, self).__init__()
        if weights is None:
            weights = [0.6, 0.4]
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
    def __init__(self, dice_weight=1, BCE_weight=0.5):
        super(WeightedDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.3, 0.7])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE
        return dice_BCE_loss


# 分割损失
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

    def forward(self, output, target):
        # 多任务学习
        loss = 0
        for index in range(1, 5):
            loss += self.criteria(output[..., index], target[..., index])
        return loss / 4


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


"""
pair loss
"""


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
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                               ceil_mode=True)  # change
        feat_S = maxpool(feat_S)
        feat_T = maxpool(feat_T)

        loss = self.criterion(feat_S, feat_T)
        return loss


"""
SimKD
"""


class SimKD(nn.Module):
    def __init__(self, *, s_n, t_n, factor=1):
        super(SimKD, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False,
                             groups=groups)

        # A bottleneck design to reduce extra parameters
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n // factor, t_n // factor),
            # depthwise convolution
            conv3x3(t_n // factor, t_n // factor, groups=t_n // factor),
            nn.BatchNorm2d(t_n // factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n // factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))

    def forward(self, feat_s, feat_t, cls_t):
        # Spatial Dimension Alignment
        # s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        # if s_H > t_H:
        #    source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
        #    target = feat_t
        # else:
        #    source = feat_s
        #    target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        #
        # trans_feat_t = target

        # Channel Alignment
        # trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        # temp_feat = self.avg_pool(trans_feat_s)
        # temp_feat = temp_feat.view(temp_feat.size(0), -1)

        # with torch.no_grad():
        # pred_feat_s = cls_t(trans_feat_s)

        # return trans_feat_s, trans_feat_t, pred_feat_s

        pred_feat_s = cls_t(feat_s)
        return feat_s, feat_t, pred_feat_s


"""
SemCKD
"""


class SemCKDLoss(nn.Module):
    """Cross-Layer Distillation with Semantic Calibration, AAAI2021"""

    def __init__(self):
        super(SemCKDLoss, self).__init__()
        # 原始
        # self.crit = nn.MSELoss(reduction='none')

        self.crit = nn.MSELoss()

    def forward(self, s_value, f_target, weight):
        # 原版本
        # bsz, num_stu, num_tea = weight.shape
        # ind_loss = torch.zeros(bsz, num_stu, num_tea)
        # ind_loss = torch.zeros(bsz, num_stu, num_tea).cuda()
        # for i in range(num_stu):
        #     for j in range(num_tea):
        #         # [2, 24, 640, 640]) -> [2, 9830400] -> [2]
        #         ind_loss[:, i, j] = self.crit(s_value[i][j], f_target[i][j]).reshape(bsz, -1).mean(-1)
        # loss = (weight * ind_loss).sum() / (1.0 * bsz * num_stu)
        # return loss

        # 只选一个的版本
        # 感觉原文中是每个样本都求一次相似度，所以不能以batch为单位计算了。
        bsz, num_stu = weight.shape
        loss = 0
        # for i in range(num_stu):
        #     for batch in range(bsz):
        #         loss += self.crit(s_value[i][weight[batch][i]][batch],
        #                           f_target[i][weight[batch][i]][batch])
        # return loss / (1.0 * num_stu * bsz)

        for i in range(num_stu):
            loss += self.crit(s_value[i], f_target[i])
        return loss / (1.0 * num_stu)


class SelfA(nn.Module):
    """Cross-layer Self Attention"""

    def __init__(self, feat_dim, s_n, t_n, soft, factor=4):
        super(SelfA, self).__init__()

        self.soft = soft
        self.s_len = len(s_n)
        self.t_len = len(t_n)
        self.feat_dim = feat_dim

        # query and key mapping
        for i in range(self.s_len):
            setattr(self, 'query_' + str(i), MLPEmbed(feat_dim, feat_dim))
        for i in range(self.t_len):
            setattr(self, 'key_' + str(i), MLPEmbed(feat_dim, feat_dim))

        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, 'regressor' + str(i) + str(j), Proj(s_n[i], t_n[j]))

    def forward(self, feat_s, feat_t):
        sim_s = list(range(self.s_len))
        sim_t = list(range(self.t_len))
        bsz = self.feat_dim
        # similarity matrix
        for i in range(self.s_len):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(self.t_len):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())

        # calculate student query
        proj_query = self.query_0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, self.s_len):
            temp_proj_query = getattr(self, 'query_' + str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)

        # calculate teacher key
        proj_key = self.key_0(sim_t[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, self.t_len):
            temp_proj_key = getattr(self, 'key_' + str(i))(sim_t[i])
            proj_key = torch.cat([proj_key, temp_proj_key[:, :, None]], 2)

        # attention weight: batch_size X No. stu feature X No.tea feature
        energy = torch.bmm(proj_query, proj_key) / self.soft
        attention = F.softmax(energy, dim=-1)

        sort_indexes = attention.sort(descending=True).indices

        # 6.1 添加 筛选出encoder和decoder阶段各自对应最相似的feature map层
        encoder_layer_num = 5
        selected_layers = []
        for i in range(len(sort_indexes)):
            selected_layers.append([])
            for j in range(len(sort_indexes[i])):
                if j < encoder_layer_num:
                    for tmp in sort_indexes[i][j]:
                        if tmp < encoder_layer_num:
                            break
                    selected_layers[i].append(tmp.item())
                else:
                    for tmp in sort_indexes[i][j]:
                        if tmp >= encoder_layer_num:
                            break
                    selected_layers[i].append(tmp.item())
        # 只加载需要的层
        proj_value_stu = []
        value_tea = []
        for i in range(self.s_len):
            s_H, t_H = feat_s[i].shape[2], feat_t[i].shape[2]
            if s_H > t_H:
                source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                target = feat_t[i]
            elif s_H <= t_H:
                source = feat_s[i]
                target = F.adaptive_avg_pool2d(feat_t[i], (s_H, s_H))
            proj_value_stu.append(getattr(self, 'regressor' + str(i) + str(i))(source))
            value_tea.append(target)
        return proj_value_stu, value_tea, np.array(selected_layers)

        # feature dimension alignment(原始代码)
        # proj_value_stu = []
        # value_tea = []
        # for i in range(self.s_len):
        #     proj_value_stu.append([])
        #     value_tea.append([])
        #     for j in range(self.t_len):
        #         s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
        #         if s_H > t_H:
        #             source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
        #             target = feat_t[j]
        #         elif s_H <= t_H:
        #             source = feat_s[i]
        #             target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
        #
        #         proj_value_stu[i].append(getattr(self, 'regressor' + str(i) + str(j))(source))
        #         value_tea[i].append(target)
        #
        # return proj_value_stu, value_tea, np.array(selected_layers)


class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""

    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear mapping for attention calculation"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_out),
            self.l2norm,
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim_out, dim_out),
            self.l2norm,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))

        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


# channel wise distillation
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, h, w = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):

    def __init__(self, norm_type='channel', divergence='kl', temperature=1.0):

        super(CriterionCWD, self).__init__()

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

        # item_loss = [round(self.criterion(norm_t[0][0].log(),norm_t[0][i]).item(),4) for i in range(c)]
        # import pdb;pdb.set_trace()
        if self.norm_type == 'channel' or self.norm_type == 'channel_mean':
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (self.temperature ** 2)


# EMKD
def prediction_map_distillation(y, teacher_scores, T=4):
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 5)
    q = q.view(-1, 5)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    if s.shape[2] != t.shape[2]:
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


def region_contrast(x, gt):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    region = []
    for i in range(gt.shape[1]):
        current_mask = gt[:, i].unsqueeze(1)
        if i == 0:
            region.append(torch.sum(x * current_mask, dim=(2, 3)) / torch.sum(current_mask, dim=(2, 3)))
        else:
            region.append(torch.sum(x * current_mask, dim=(2, 3)) / (torch.sum(current_mask, dim=(2, 3)) + 1.0))
    similarity = 0
    add = 0
    for i in range(gt.shape[1]):
        for j in range(i):
            similarity += F.cosine_similarity(region[i], region[j], dim=1)
            add += 1
    return similarity / add


def region_affinity_distillation(s, t, gt):
    """
    region affinity distillation KD loss
    :param s: student feature
    :param t: teacher feature
    :return: loss value
    """
    gt = F.interpolate(gt, s.size()[2:])
    return (region_contrast(s, gt) - region_contrast(t, gt)).pow(2).mean()


# spkd
# https://github.com/DongGeun-Yoon/SPKD
def spkd(s, t):
    # 限定bc = 2, 否则每两个样本之间都要做
    l2 = nn.MSELoss()
    student_sim = batch_similarity(s)
    teacher_sim = batch_similarity(t)
    loss = l2(student_sim, teacher_sim)
    return loss


# batch similarity
def batch_similarity(fm):
    fm = fm.view(fm.size(0), -1)
    Q = torch.mm(fm, fm.transpose(0, 1))
    normalized_Q = Q / torch.norm(Q, 2, dim=1).unsqueeze(1).expand(Q.shape)
    return normalized_Q


# IFVD
# self.criterion_ifv = CriterionIFV(classes=args.num_classes).cuda()
# temp = args.lambda_ifv*self.criterion_ifv(self.preds_S, self.preds_T, self.labels)
class CriterionIFV(nn.Module):
    def __init__(self, classes):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes

    def forward(self, feat_S, feat_T, target):
        feat_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        target = target.permute(0, 3, 1, 2)
        target = torch.argmax(target, dim=1)
        tar_feat_S = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode='nearest')(target.unsqueeze(1).float()).expand(feat_T.size())
        center_feat_S = feat_S.clone()
        center_feat_T = feat_T.clone()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * (
                    (mask_feat_S * feat_S).sum(-1).sum(-1) / (mask_feat_S.sum(-1).sum(-1) + 1e-6)).unsqueeze(
                -1).unsqueeze(-1)
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * (
                    (mask_feat_T * feat_T).sum(-1).sum(-1) / (mask_feat_T.sum(-1).sum(-1) + 1e-6)).unsqueeze(
                -1).unsqueeze(-1)

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return loss
