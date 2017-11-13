"""
Model definition for weakly supervised object localization with pytorch
=====================================================================
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import torch
import torch.nn as nn
import numpy as np
import os
# import shutil
import torchvision.models as models
from spn.modules import SoftProposal
import spn_codes.spatialpooling as spatialpooling


class WSL(nn.Module):
    def __init__(self, num_classes=20, num_maps=1024):
        super(WSL, self).__init__()
        model = models.vgg16(pretrained=True)
        num_features = model.features[28].out_channels
        self.features = nn.Sequential(*list(model.features.children())[:-1])
        # self.spatial_pooling = pooling
        self.addconv = nn.Conv2d(num_features, num_maps, kernel_size=3,
                                 stride=1, padding=1, groups=2, bias=True)
        self.maps = nn.ReLU()
        self.sp = SoftProposal()
        self.sum = spatialpooling.SpatialSumOverMap()

        # classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_maps, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.addconv(x)
        x = self.maps(x)
        sp = self.sp(x)
        x = self.sum(sp)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_att_map(self, x):
        x = self.features(x)
        x = self.addconv(x)
        x = self.maps(x)
        sp = self.sp(x)
        x = self.sum(sp)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, sp

    # def load_pretrained_vgg(self, fname):
    #     vgg_param = np.load(fname, encoding='latin1').item()  # vgg16
    #     net_param = self.state_dict()
    #     para_keys = list(net_param.keys())
    #     for idx in range(26):
    #         name = para_keys[idx]
    #         val = net_param[name]
    #         i, j = int(name[4]), int(name[6]) + 1
    #         ptype = 'weights' if name[-1] == 't' else 'biases'
    #         key = 'conv{}_{}'.format(i, j)
    #         param = torch.from_numpy(vgg_param[key][ptype])
    #         if ptype == 'weights':
    #             param = param.permute(3, 2, 0, 1)
    #         val.copy_(param)

    def load_checkpoint(self, fname):
        if os.path.isfile(fname):
            print('loading checkpoint {}'.format(fname))
            checkpt = torch.load(fname)
            self.load_state_dict(checkpt['state_dict'])
        else:
            print('{} not found'.format(fname))


class ConvReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz, stride=1, relu=True, pd=True, bn=False):
        super(ConvReLU, self).__init__()
        padding = int((kernel_sz - 1) / 2) if pd else 0  # same spatial size by default
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_sz, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ClsConv(nn.Module):
    """docstring for ClsConv"""

    def __init__(self, in_ch=512, bn=False):
        super(ClsConv, self).__init__()
        self.conv_layers = nn.Sequential(ConvReLU(in_ch, 256, 3, pd=True, bn=bn),
                                         ConvReLU(256, 128, 3, pd=True, bn=bn),
                                         ConvReLU(128, 64, 3, pd=True, bn=bn),
                                         nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, feature):
        return self.conv_layers(feature)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_pretrained(model, fname, optimizer=None):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch for train or only model for test
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer, checkpoint['epoch']
        else:
            return model
    else:
        raise Exception("=> no checkpoint found at '{}'".format(fname))



