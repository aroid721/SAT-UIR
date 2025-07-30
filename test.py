import os
import cv2
import torch
from utils import *
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from adamp import AdamP
from model import AIMnet
from dataset_all import TestData, ValLabeled_unresize
import skimage.metrics
import tqdm
from thop import profile
import time


bz = 1
model_root = 'model/ckpt/model_e100.pth'
save_path  = 'experiment/testR'
input_path = 'data/benchmark/test'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
checkpoint = torch.load(model_root)

val_dataset = TestData(dataroot=input_path)
val_loader = DataLoader(val_dataset, batch_size=bz, sampler=None)

model = AIMnet().cuda()
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
model.eval()

print('START!')

with torch.no_grad():
    for i, (val_data, val_la) in enumerate(val_loader):
        val_data = Variable(val_data).cuda()
        val_la = Variable(val_la).cuda()
        val_output, _, ssim_predict, _ = model(val_data, val_la)
        name = val_dataset.A_paths[i].split('/')[-1]
        result = val_output.detach().cpu().numpy()
        result = result[0].transpose(1, 2, 0)
        result = (result * 255).astype(np.uint8)
        temp_res = Image.fromarray(result)
        temp_res.save('%s/%s' % (save_path, name))

print("over!")
