import os
import torch
from glob import glob
from os.path import join
from torchvision.transforms import transforms
from PIL import Image

# initialize the reliable bank

input_dir = 'data/unlabeled/input'
result_dir = 'data/unlabeled/candidate'
input_lists = glob(join(input_dir, '*.*'))
for gen_path in zip(input_lists):
    img = Image.open(gen_path[0]).convert("RGB")
    img_name = gen_path[0].split('/')[3]
    print(img_name)
    img = img.resize((256, 256), Image.ANTIALIAS)    
    res = img
    res.save(os.path.join(result_dir, img_name))
