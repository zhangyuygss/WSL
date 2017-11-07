# A demo run on one image

import torch
import os
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

data_dir = '/home/zhangyu/data/VOC2007_test/'
imgDir = os.path.join(data_dir, 'JPEGImages')

demo_img = os.path.join