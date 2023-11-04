import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from losses import Losses_unet

__all__ = ['StudentSegContrast']


class InterLKD(nn.Module):
    def __init__(self, num_classes, queue_size, contrast_size,
                 contrast_kd_temperature, contrast_temperature):
        super(InterLKD, self).__init__()
        self.base_temperature = 0.1
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        self.window_size = 8
        self.dim = 32*self.window_size*self.window_size 
        self.num_classes = num_classes
        self.queue_size = queue_size  
        self.multi_queue_size = [2048, 2048, 4096] 
        self.contrast_size = contrast_size 
        self.warmup_epoch = 10
        self.hard_loss = Losses_unet.LBTW_Loss(Losses_unet.WeightedDiceBCE()).cuda()
        self.register_buffer("region_embedding_queue", torch.randn(self.num_classes, self.queue_size, self.dim))
        self.region_embedding_queue = nn.functional.normalize(self.region_embedding_queue, p=2, dim=2)
        self.register_buffer("queue_ptr", torch.zeros(self.num_classes, dtype=torch.long))
        self.register_buffer("queue_scores", torch.zeros(self.num_classes, self.queue_size))
        self.register_buffer("student_scores", torch.zeros(self.num_classes, self.queue_size))
        self.register_buffer("student_ptr", torch.zeros(self.num_classes, dtype=torch.long))

    def _sample_negative(self, Q, index):
        class_num, cache_size, feat_size = Q.shape
        contrast_size = index.size(1)
        X_ = torch.zeros((class_num * contrast_size, feat_size)).float().cuda()  # 5 * contrast_size
        y_ = torch.zeros((class_num * contrast_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            this_q = Q[ii, index[ii], :]
            X_[sample_ptr:sample_ptr + contrast_size, ...] = this_q
            y_[sample_ptr:sample_ptr + contrast_size, ...] = ii
            sample_ptr += contrast_size

        return X_, y_

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits / self.contrast_kd_temperature, dim=1)
        p_t = F.softmax(t_logits / self.contrast_kd_temperature, dim=1)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature ** 2
        return sim_dis
    
    def screen_lesion_regions(self, labels, stride=1, threshold = 0.5):
        lesions = ['', 'EX', 'HE', 'MA', 'SE']
        window_size = self.window_size
        boxes_list = [[] for _ in range(labels.shape[0])]
        for bc in range(labels.shape[0]):
            for lesion in range(1, labels.shape[1]-1): # ignore SE
                # list version
                current_map = labels[bc][lesion].cpu().detach().numpy()               
                height, width = current_map.shape
                rectangles = []
                for y in range(0, height - window_size + 1, stride):
                    for x in range(0, width - window_size + 1, stride):
                        region = current_map[y: y + window_size, x: x + window_size]
                        nonzero_ratio = np.count_nonzero(region) / (window_size ** 2)
                        if lesions[lesion] == 'EX':
                            if nonzero_ratio > 0.8:
                                rectangles.append((x, y, x + window_size, y + window_size, nonzero_ratio))
                        elif lesions[lesion] == 'HE':
                            if nonzero_ratio > 0.8:
                                rectangles.append((x, y, x + window_size, y + window_size, nonzero_ratio))
                        elif lesions[lesion] == 'MA':
                            if nonzero_ratio > 0.6:
                                rectangles.append((x, y, x + window_size, y + window_size, nonzero_ratio))
                        elif lesions[lesion] == 'SE':
                            if nonzero_ratio > 0.6:
                                rectangles.append((x, y, x + window_size, y + window_size, nonzero_ratio))
                        
                rectangles.sort(key=lambda rect: rect[4], reverse=True)

                final_rectangles = []
                for rect in rectangles:
                    overlapping = False
                    for final_rect in final_rectangles:
                        x_overlap = max(0, min(rect[2], final_rect[2]) - max(rect[0], final_rect[0]))
                        y_overlap = max(0, min(rect[3], final_rect[3]) - max(rect[1], final_rect[1]))
                        if x_overlap * y_overlap > 0:
                            overlapping = True
                            break
                    if not overlapping:
                        final_rectangles.append(rect)
                boxes_list[bc].append(final_rectangles)
                # visualize
                # import cv2
                # import matplotlib.pyplot as plt
                # import time
                # current_time = time.time()
                # plt.imshow(current_map, cmap='gray')
                # plt.savefig('./tmp/' + str(current_time) + lesions[lesion] + '.jpg')
                # plt.cla()
                # for rect in final_rectangles:
                #      x1, y1, x2, y2, _ = rect        
                #      cv2.rectangle(current_map, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # plt.imshow(current_map, cmap='gray')
                # plt.savefig('./tmp/' + str(current_time) + lesions[lesion] + '_boxes' + '.jpg')
                # plt.cla()
        return boxes_list

    def forward(self, s_feats, t_feats, labels=None, teacher_predict=None, student_predict=None, epoch=None, max_region_num=1024):

        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = F.normalize(s_feats, p=2, dim=1)

        # screening lesion region
        boxes_list = self.screen_lesion_regions(labels)

        all_student_regions = []
        all_teacher_regions = []
        batch_size = s_feats.shape[0]
        for i in range(batch_size):
            for lesion in range(len(boxes_list[i])):
                for j in range(min(len(boxes_list[i][lesion]), max_region_num)):
                    x_start, y_start, x_end, y_end, _ = boxes_list[i][lesion][j]
                    current_student_region = s_feats[i, :, y_start:y_end, x_start:x_end]
                    current_teacher_region = t_feats[i, :, y_start:y_end, x_start:x_end]
                    teacher_patch_mask = teacher_predict[i, lesion+1, y_start:y_end, x_start:x_end]
                    student_patch_mask = student_predict[i, lesion+1, y_start:y_end, x_start:x_end]
                    gt_patch = labels[i, lesion+1, y_start:y_end, x_start:x_end]
                    with torch.no_grad():
                        teacher_patch_loss = self.hard_loss(teacher_patch_mask, gt_patch)   
                        student_patch_loss = self.hard_loss(student_patch_mask, gt_patch)   
                    # torch.Size([5, 128, 32768])
                    current_teacher_region = current_teacher_region.flatten()
                    # Add teacher lesion region to the queue
                    self.region_embedding_queue[lesion, int(self.queue_ptr[lesion]), :] = nn.functional.normalize(current_teacher_region.view(-1), p=2, dim=0)
                    # Add score corresponding to teacher lesion region to score queue
                    self.queue_scores[lesion, int(self.queue_ptr[lesion])] = teacher_patch_loss
                    self.queue_ptr[lesion] = (self.queue_ptr[lesion] + 1) % self.multi_queue_size[lesion] 
                    self.student_scores[lesion, int(self.student_ptr[lesion])] = student_patch_loss       
                    self.student_ptr[lesion] = (self.student_ptr[lesion] + 1) % self.multi_queue_size[lesion] 
                    if epoch >= self.warmup_epoch:
                        # Self-paced strategy
                        # sorted_student_scores, _ = torch.sort(self.student_scores, dim=1)
                        # current_process = epoch // 100
                        # min_criterion, max_criterion = sorted_student_scores[lesion][100*current_process], sorted_student_scores[lesion][(100*current_process)+100]
                        # if min_criterion <= student_patch_loss <= max_criterion:
                        #     all_student_regions.append(current_student_region)
                        #     all_teacher_regions.append(current_teacher_region)
                        # Hard-sample strategy
                        sorted_student_scores, _ = torch.sort(self.student_scores, dim=1)
                        min_criterion, max_criterion = sorted_student_scores[lesion][int(len(sorted_student_scores[0])*0.6)], sorted_student_scores[lesion][-1]
                        if min_criterion <= student_patch_loss <= max_criterion:
                            all_student_regions.append(current_student_region)
                            all_teacher_regions.append(current_teacher_region)                    

        if epoch < self.warmup_epoch or len(all_student_regions) == 0:
            return 1e-9

        # Self-paced hard sample learning strategy
        start_index1, start_index2 = int(2048 * 0.6), int(4096 * 0.6)
        end_index1, end_index2= 2048 - 1, 4096-1
        length1 = (end_index1 - start_index1) + 1
        length2 = (end_index2 - start_index2) + 1
        step1 = (length1 - self.contrast_size) / 500  
        step2 = (length2 - self.contrast_size) / 500 
        start_index1, start_index2 = start_index1+int(step1*epoch), start_index2+int(step2*epoch)
        end_index1, end_index2 = start_index1 + self.contrast_size, start_index2 + self.contrast_size
        # Get queue scores and sort them
        EX, HE, MA = self.queue_scores[0][:2048], self.queue_scores[1][:2048], self.queue_scores[2]
        _, EX_indexes = torch.sort(EX)
        _, HE_indexes = torch.sort(HE)
        _, MA_indexes = torch.sort(MA)
        EX_soted_indexes, HE_sorted_indexes, MA_sorted_indexes = EX_indexes[start_index1: end_index1], HE_indexes[start_index1: end_index1], MA_indexes[start_index2: end_index2]
        soted_indexes = torch.stack([EX_soted_indexes, HE_sorted_indexes, MA_sorted_indexes])
        t_X_region_contrast, _ = self._sample_negative(self.region_embedding_queue, soted_indexes)
        teacher_regions, student_regions = torch.stack(all_teacher_regions).cuda(), torch.stack(all_student_regions).cuda() 
        number_samples = teacher_regions.shape[0] 
        # Calculate Similarity Matirx
        t_region_logits = torch.div(torch.mm(teacher_regions.view(number_samples, -1), t_X_region_contrast.T), self.contrast_temperature)
        s_region_logits = torch.div(torch.mm(student_regions.view(number_samples, -1), t_X_region_contrast.T), self.contrast_temperature)

        return self.contrast_sim_kd(s_region_logits, t_region_logits.detach())