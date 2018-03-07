"""
Main code for weakly supervised object localization
===================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import os
import numpy as np
import time
import datetime
from model.model import WSL, load_pretrained
import data_utils.load_voc as load_voc
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from evaluate.rst_for_corloc import spn_torch_rst
from evaluate.corloc_eval import corloc


parser = argparse.ArgumentParser(description='test torch version of spn')
parser.add_argument('--batch_size', default=64, type=int, metavar='BT',
                    help='batch size')

data_dir = '/home/zhangyu/data/VOC2007_test/'
root_dir = '/disk3/zhangyu/WeaklyLoc/spn_torch/'
attention_maps_h5 = os.path.join(root_dir, 'rst/h5/attention_maps.h5')
imgDir = os.path.join(data_dir, 'JPEGImages')
train_annos = os.path.join(data_dir, 'train_annos')
trainval_annos = os.path.join(data_dir, 'Annotations')
att_map_dir = os.path.join(root_dir, 'rst/h5big/')
cls_number = 20

save_file = os.path.join(att_map_dir, 'predict{}.csv'.format(
    datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))


def main():
    global args
    args = parser.parse_args()
    test_loader = prepare_data(trainval_annos)

    test(test_loader)
    corloc_rst = corloc(save_file, trainval_annos)
    print('Corloc results: {}'.format(corloc_rst))


def test(test_loader):
    for i, data in enumerate(test_loader):
        print('Testing: [{0}/{1}] '.format(i, len(test_loader)))
        batch_names = data['filename']
        img_szs = data['sz'].numpy()
        target = data['class'].float().numpy()
        # generate results
        spn_torch_rst(batch_names, target, img_szs, att_map_dir, save_file)


def prepare_data(annos_path):
    # prepare dataloader for training and validation
    train_dataset = load_voc.VOCDataset(
        xmlsPath=annos_path, imgDir=imgDir,
        transform=transforms.Compose([
            load_voc.Rescale((224, 224)),
            load_voc.ToTensor(),
            load_voc.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=1, drop_last=True)
    return train_loader


if __name__ == '__main__':
    main()
