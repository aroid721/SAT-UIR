import os

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from itertools import cycle
import torchvision
import torch.distributed as dist
from torch.optim import lr_scheduler
import PIL.Image as Image
from utils import *
from torch.autograd import Variable
from adamp import AdamP
from torchvision.models import vgg16
from loss.losses import *
from model import GetGradientNopadding
from loss.contrast import ContrastLoss
import pyiqa
from PIL import Image
import time



class Trainer:
    def __init__(self, model, tmodel, args, supervised_loader, unsupervised_loader, val_loader, iter_per_epoch, writer):

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader
        self.args = args
        self.iter_per_epoch = iter_per_epoch
        self.writer = open('log.txt', 'w')
        self.model = model
        self.tmodel = tmodel
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.num_epochs
        self.save_period = 20
        self.loss_unsup = nn.L1Loss().cuda()
        self.loss_str = MyLoss().cuda()
        self.bceloss = nn.BCELoss().cuda()
        self.loss_match = nn.MSELoss()
        self.loss_grad = nn.L1Loss().cuda()
        self.loss_cr = ContrastLoss().cuda()
        self.consistency = 0.2
        self.consistency_rampup = 100.0
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        self.curiter = 0
        self.model.cuda()
        self.tmodel.cuda()
        self.device, available_gpus = self._get_available_devices(self.args.gpus)
        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
        self.score_his = {}
        # set optimizer and learning rate
        self.optimizer_s = AdamP(self.model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[200], gamma=0.5)

    @torch.no_grad()
    def update_teachers(self, teacher, itera, keep_rate=0.996, epochs = 100):
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def predict_with_out_grad(self, image, image_l):
        with torch.no_grad():
            predict_target_ul, _, ssim_ul, fea_ul = self.tmodel(image, image_l)

        return predict_target_ul, ssim_ul

    def freeze_teachers_parameters(self):
        for p in self.tmodel.parameters():
            p.requires_grad = False

    def get_reliable(self, teacher_predict, student_predict, positive_list, p_name, ssim_score_ul, epoch):

        N = teacher_predict.shape[0]
        scores = ssim_score_ul.clone()
        positive_sample = positive_list.clone()
        negative_sample = positive_list.clone()
        for idx in range(0, N):
            score_his = self.score_his[p_name[idx]] if p_name[idx] in self.score_his else 0.0
            if ssim_score_ul[idx] > score_his:
                positive_sample[idx] = student_predict[idx] 
                # update the reliable bank
                temp_c = np.transpose(student_predict[idx].detach().cpu().numpy(), (1, 2, 0))
                temp_c = np.clip(temp_c, 0, 1)
                arr_c = (temp_c*255).astype(np.uint8)
                arr_c = Image.fromarray(arr_c)
                arr_c.save('%s' % p_name[idx])
                self.score_his[p_name[idx]] = ssim_score_ul[idx].item()
            else:
                scores[idx] = score_his
                negative_sample[idx] = negative_sample[idx]
        del N, teacher_predict, student_predict, positive_list
        return positive_sample, negative_sample, scores

    def train(self):
        self.freeze_teachers_parameters()
        if self.start_epoch == 1:
            initialize_weights(self.model)
        else:
            checkpoint = torch.load(self.args.resume_path)
            self.model.load_state_dict(checkpoint['state_dict'])

        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_ave, psnr_train = self._train_epoch(epoch)
            loss_val = loss_ave.item() / self.args.crop_size * self.args.train_batchsize
            train_psnr = sum(psnr_train) / len(psnr_train)
            psnr_val = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)

            print('[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f' % (
                epoch, loss_val, train_psnr, val_psnr, self.lr_scheduler_s.get_last_lr()[0]))

            # Save checkpoint
            if epoch % self.save_period == 0 and self.args.local_rank <= 0:
                state = {'arch': type(self.tmodel).__name__,
                         'epoch': epoch,
                         'state_dict': self.tmodel.state_dict(),
                         'optimizer_dict': self.optimizer_s.state_dict()}
                ckpt_name = str(self.args.save_path) + 'model_e{}.pth'.format(str(epoch))
                print("Saving a checkpoint: {} ...".format(str(ckpt_name)))
                torch.save(state, ckpt_name)

    def _train_epoch(self, epoch):
        sup_loss = AverageMeter()
        unsup_loss = AverageMeter()
        loss_total_ave = 0.0
        psnr_train = []
        self.model.train()
        self.freeze_teachers_parameters()
        sup_loader = iter(self.supervised_loader)
        unsup_loader = iter(self.unsupervised_loader)
        tbar = range(len(self.supervised_loader))
        tbar = tqdm(tbar, ncols=130, leave=True)
        for i in tbar:
            (img_data, label, img_la), (unpaired_data_w, unpaired_data_s, unpaired_la, p_list, p_name) = next(sup_loader), next(unsup_loader)
            img_data = Variable(img_data).cuda(non_blocking=True)
            label = Variable(label).cuda(non_blocking=True)
            img_la = Variable(img_la).cuda(non_blocking=True)
            unpaired_data_s = Variable(unpaired_data_s).cuda(non_blocking=True)
            unpaired_data_w = Variable(unpaired_data_w).cuda(non_blocking=True)
            unpaired_la = Variable(unpaired_la).cuda(non_blocking=True)

            p_list = Variable(p_list).cuda(non_blocking=True)
            # teacher output
            predict_target_u, ssim_score_ul = self.predict_with_out_grad(unpaired_data_w, unpaired_la)
            origin_predict = predict_target_u.detach().clone()
            # student output
            outputs_l, outputs_g, ssim_score_l,fea_l = self.model(img_data, img_la)
           
            ssim_l, _ = compute_psnr_ssim_1(outputs_l.detach(), label.detach())
           
            ssim_l = torch.FloatTensor(ssim_l).cuda()
            pseudo_loss =  self.bceloss(ssim_score_l, ssim_l)

            structure_loss =  self.loss_str(outputs_l, label)
            perpetual_loss =  self.loss_per(outputs_l, label)
           
            get_grad = GetGradientNopadding().cuda()
            gradient_loss = self.loss_grad(get_grad(outputs_l), get_grad(label)) + self.loss_grad(outputs_g, get_grad(label))
            loss_sup = structure_loss + 0.5 * perpetual_loss + 0.1 * gradient_loss + 0.01 * pseudo_loss 
            sup_loss.update(loss_sup.mean().item())
            p_sample, n_sample, scores = self.get_reliable(predict_target_u, predict_target_u, p_list, p_name, ssim_score_ul, epoch)

        
            outputs_ul, _, ssim_score_ul_s,fea_ul_s = self.model(unpaired_data_s, unpaired_la)
           


            loss_unsu = self.loss_cr(outputs_ul, p_sample, unpaired_data_w)
            loss_unsu2 =  self.loss_unsup(outputs_ul, p_sample)

            loss_unsu3 =  torch.mean(torch.max(ssim_score_ul_s - ssim_score_ul,torch.tensor(0.0)))

            
            unsup_loss.update((loss_unsu + loss_unsu2 + 0.01*loss_unsu3).mean().item())
            consistency_weight = self.get_current_consistency_weight(epoch)
            total_loss = loss_sup + consistency_weight*(loss_unsu + loss_unsu2 + 0.01*loss_unsu3)
            total_loss = total_loss.mean()
            psnr_train.extend(to_psnr(outputs_l, label))
            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            tbar.set_description('Train-Student Epoch {} | Ls {:.4f} Lu {:.4f}|'
                                 .format(epoch, sup_loss.avg, unsup_loss.avg))

            del img_data, label, unpaired_data_w, n_sample, unpaired_data_s, img_la, unpaired_la

            with torch.no_grad():
                self.update_teachers(teacher=self.tmodel, itera=self.curiter, epochs = epoch)
                self.curiter = self.curiter + 1

        loss_total_ave = loss_total_ave + total_loss

        return loss_total_ave, psnr_train

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.tmodel.eval()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        total_loss_val = AverageMeter()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for i, (val_data, val_label, val_la) in enumerate(tbar):
                val_data = Variable(val_data).cuda()
                val_label = Variable(val_label).cuda()
                val_la = Variable(val_la).cuda()
                # forward
                val_output, _,_,_ = self.tmodel(val_data, val_la)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)
                val_psnr.update(temp_psnr, N)
                val_ssim.update(temp_ssim, N)
                psnr_val.extend(to_psnr(val_output, val_label))
                tbar.set_description('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}|'.format(
                    "Eval-Student", epoch, val_psnr.avg, val_ssim.avg))

            self.writer.write('{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f} + \n'.format("Eval-Student", epoch, val_psnr.avg, val_ssim.avg))
            self.writer.flush()
            del val_output, val_label, val_data
            return psnr_val

    def _get_available_devices(self, n_gpu):
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu > sys_gpu:
            print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu
        device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
        available_gpus = list(range(n_gpu))
        return device, available_gpus

    def get_current_consistency_weight(self, epoch):
        return self.consistency * self.sigmoid_rampup(epoch, self.consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
