# A demo run on one image

import torch
import os
from skimage import io, transform
import numpy as np
import time
import datetime
from model.model import WSL
import data_utils.load_voc as load_voc
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spn_codes.models import SPNetWSL
from evaluate.rst_for_corloc import rst_for_corloc
from evaluate.corloc_eval import corloc
from .get_attention_map import process_one


data_dir = '/home/zhangyu/data/VOC2007_test/'
imgDir = os.path.join(data_dir, 'JPEGImages')
ck_pt = '/disk3/zhangyu/WeaklyDetection/spn_new/\
checkpt/best_model/best_checkpoint_epoch20.pth.tar'

demo_img = os.path.join(imgDir, '000182.jpg')

img = io.imread(demo_img)
img_sz = np.array(list(img.shape[:2]))

input = img #TODO

num_class = 20
net = WSL(num_class)
load_pretrained(model=net, ck_pt)
net.eval()
cls_scores, ft = model.get_att_map(input)
lr_weigth = model.classifier[1].weight.cpu().data.numpy()

atten_maps = process_one(ft, lr_weigth)
