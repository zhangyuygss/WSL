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
from torch.autograd import Variable
from bboxgenerator.gen_bbox import gen_bbox


class FeatureEnhance(torch.autograd.Function):
    """
    Autograd function for feature enhancement with box
    TODO: try simplify this
    """
    def __init__(self):
        super(FeatureEnhance, self).__init__()
        self.mask = None
        self.valid_box = None
        self.gpuID = None 

    def forward(self, feature, box):
        batchsize, ch, sp_sz, _ = feature.size()
        pos = torch.FloatTensor([sp_sz, sp_sz, sp_sz, sp_sz])
        mask = torch.zeros(feature.size())
        valid_box = torch.zeros(batchsize, 4)
        # pos, mask, valid_box = Variable(pos), Variable(mask), Variable(valid_box)
        if feature.is_cuda:
            self.gpuID = torch.cuda.device_of(feature)
            pos, valid_box = pos.cuda(self.gpuID.idx), valid_box.cuda(self.gpuID.idx)
            mask = mask.cuda(self.gpuID.idx)
        pos = pos.expand_as(box)
        pos = torch.round(torch.mul(box, pos)).type(torch.IntTensor)
        # TODO: try to remove for loop for efficiency
        for bs_idx in range(batchsize):
            if pos[bs_idx, 3] > pos[bs_idx, 1] and pos[bs_idx, 2] > pos[bs_idx, 0]:
                valid_box[bs_idx, :] = 1
                mask[bs_idx, :, pos[bs_idx, 1]:pos[bs_idx, 3], 
                     pos[bs_idx, 0]:pos[bs_idx, 2]] = 1
        self.mask = mask
        self.valid_box = valid_box
        enhanced_ft = torch.mul(feature, mask)
        return enhanced_ft

    def backward(self, grad_output):
        return self.mask, self.valid_box


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


class WSDR(nn.Module):
    num_classes = 20

    def __init__(self, bn=False):
        super(WSDR, self).__init__()
        # Network based on VGG16
        self.conv1 = nn.Sequential(ConvReLU(3, 64, 3, pd=True, bn=bn),
                                   ConvReLU(64, 64, 3, pd=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(ConvReLU(64, 128, 3, pd=True, bn=bn),
                                   ConvReLU(128, 128, 3, pd=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(ConvReLU(128, 256, 3, pd=True, bn=bn),
                                   ConvReLU(256, 256, 3, pd=True, bn=bn),
                                   ConvReLU(256, 256, 3, pd=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(ConvReLU(256, 512, 3, pd=True, bn=bn),
                                   ConvReLU(512, 512, 3, pd=True, bn=bn),
                                   ConvReLU(512, 512, 3, pd=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(ConvReLU(512, 512, 3, pd=True, bn=bn),
                                   ConvReLU(512, 512, 3, pd=True, bn=bn),
                                   ConvReLU(512, 512, 3, pd=True, bn=bn))

        # self.feature_gap = nn.Sequential(ConvReLU(512, 256, 3, pd=True, bn=bn),
        #                                  ConvReLU(256, 128, 3, pd=True, bn=bn),
        #                                  ConvReLU(128, 64, 3, pd=True, bn=bn),
        #                                  nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.cls_convs = nn.ModuleList([ClsConv() for i in range(self.num_classes)])

        self.feature_fwd = nn.Sequential(ConvReLU(512, 256, 3, pd=True, bn=bn),
                                         ConvReLU(256, 128, 3, pd=True, bn=bn),
                                         ConvReLU(128, 64, 3, pd=True, bn=bn),
                                         nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

        self.gap = nn.AvgPool2d(kernel_size=14, stride=14)

        self.box = nn.Sigmoid()

    def forward(self, im_data):
        """
        im_data: input image
        pre_maps: class feature maps from last layer of the network
        """
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv5 = self.conv5(x)

        features = conv5
        batch_sz, ch, sp_sz, _ = features.size()
        cls_maps = cls_scores = None
        fwd_map = self.feature_fwd(features)
        for cls_idx in range(self.num_classes):
            bwd_map = self.cls_convs[cls_idx](features)
            for rnn_idx in range(3):
                fused_map = torch.add(bwd_map, fwd_map)
                fused_map = self.box(fused_map)
                mask_feature = fused_map.expand_as(conv5)
                mask_feature = torch.mul(mask_feature, conv5)
                enhanced_ft = torch.add(mask_feature, conv5)
                bwd_map = self.cls_convs[cls_idx](enhanced_ft)

            cls_maps = bwd_map if cls_idx == 0 \
                else torch.cat((cls_maps, bwd_map), 1)
            cls_score = self.gap(bwd_map).view(batch_sz, -1)
            cls_scores = cls_score if cls_idx == 0 \
                else torch.cat((cls_scores, cls_score), 1)
        return cls_scores, cls_maps 

    def forward_no_recurrent(self, im_data):
        """
        Forward path of the network without recurrent part, used for initializing
        the 'pre_maps' part of the training process
        Change architecture, this function is useless now
        =========================================================================
        im_data: input image
        """
        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv5 = self.conv5(x)
        cls_features = []
        cls_scores = []
        for cls_idx in range(self.num_classes):
            cls_feature = self.feature_gap(conv5)
            cls_features = cls_feature if cls_idx == 0 \
                else torch.cat((cls_features, cls_feature), 1)
            # cls_score = self.gap(cls_feature)
            # cls_scores.append(cls_score)
        return cls_features

    def load_pretrained_vgg(self, fname):
        vgg_param = np.load(fname, encoding='latin1').item()  # vgg16
        net_param = self.state_dict()
        para_keys = list(net_param.keys())
        for idx in range(26):
            name = para_keys[idx]
            val = net_param[name]
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}'.format(i, j)
            param = torch.from_numpy(vgg_param[key][ptype])
            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)
            val.copy_(param)

    def load_checkpoint(self, fname):
        if os.path.isfile(fname):
            print('loading checkpoint {}'.format(fname))
            checkpt = torch.load(fname)
            self.load_state_dict(checkpt['state_dict'])
        else:
            print('{} not found'.format(fname))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)





