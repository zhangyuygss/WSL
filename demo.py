# A demo run on one image

import torch
import os
from skimage import io, transform
import numpy as np
import time
import datetime
from model.model import WSL, load_pretrained
import data_utils.load_voc as load_voc
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from spn_codes.models import SPNetWSL
from evaluate.rst_for_corloc import rst_for_corloc
from evaluate.corloc_eval import corloc
from evaluate.get_attention_map import process_one


data_dir = '/home/zhangyu/data/VOC2007_test/'
imgDir = os.path.join(data_dir, 'JPEGImages')
xml_files = os.path.join(data_dir, 'Annotations')
ck_pt = '/disk3/zhangyu/WeaklyDetection/spn_new/\
checkpt/best_model/best_checkpoint_epoch20.pth.tar'
gpuID = 0

img_name = '000182'

demo_img = os.path.join(imgDir, '{}.jpg'.format(img_name))
demo_xml = os.path.join(xml_files, '{}.xml'.format(img_name))

img = io.imread(demo_img)
img_sz = np.array(list(img.shape[:2]))
trans = transforms.Compose([
    load_voc.Rescale((224,224)),
    load_voc.ToTensor(),
    load_voc.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])])
cls = load_voc.load_class(demo_xml)
sample = {'filename':demo_img, 'sz': img_sz, 'image': img, 'class': cls}
sample = trans(sample)
sample['image'] = sample['image'].unsqueeze(0).float()
input_var = torch.autograd.Variable(sample['image'], volatile=True).cuda(gpuID)

num_class = 20
net = WSL(num_class)
load_pretrained(net, ck_pt)
net.eval()
net.cuda(gpuID)
cls_scores, ft = net.get_att_map(input_var)
lr_weigth = net.classifier[1].weight.cpu().data.numpy()
ft = ft.cpu().data.numpy()

atten_maps = process_one(ft[0,:,:,:], lr_weigth, None)
print('Process finished!')
