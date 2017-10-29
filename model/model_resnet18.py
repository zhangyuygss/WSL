import torch
import torch.nn as nn
import numpy as np
import os
from model.resnet import ResNet, BasicBlock
from torch.nn.parameter import Parameter


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


class WSDR(nn.Module):
    num_classes = 20

    def __init__(self, bn=False):
        super(WSDR, self).__init__()
        self.resnet_feature = ResNet(BasicBlock, [2, 2, 2, 2])

        self.feature_gap = nn.Sequential(ConvReLU(512, 256, 3, pd=True, bn=bn),
                                         ConvReLU(256, 128, 3, pd=True, bn=bn),
                                         nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.feature_fwd = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        self.sing_flow = nn.Conv2d(512, 20, kernel_size=3, stride=1, padding=1)
        self.sing_flow_2 = nn.Sequential(ConvReLU(512, 256, 3, pd=True, bn=bn),
                                         nn.Conv2d(256, 20, kernel_size=3, stride=1, padding=1))
        # self.feature_fwd = nn.Sequential(ConvReLU(512, 1, 3, pd=True, bn=bn))
        self.gap = nn.AvgPool2d(kernel_size=7, stride=7)
        self.box = nn.Sigmoid()

    def forward(self, im_data):
        features = self.resnet_feature(im_data)
        batch_sz, ch, sp_sz, _ = features.size()
        cls_maps = cls_scores = None
        feature_map = self.feature_fwd(features)
        for cls_idx in range(self.num_classes):
            bwd_map = self.feature_gap(features)
            #TODO: add rnn loop
            cls_maps = bwd_map if cls_idx == 0 \
                else torch.cat((cls_maps, bwd_map), 1)
            cls_score = self.gap(bwd_map).view(batch_sz, -1)
            cls_scores = cls_score if cls_idx == 0 \
                else torch.cat((cls_scores, cls_score), 1)
        return cls_scores

    # def forward(self, im_data):
    #     ft = self.resnet_feature(im_data)
    #     maps = self.sing_flow_2(ft)
    #     scores = self.gap(maps).squeeze()
    #     return scores

    def load_resnet(self, resnet_file='/home/zhangyu/data/resnet18-5c106cde.pth'):
        resnet_dic = torch.load(resnet_file)
        net_paras = self.resnet_feature.state_dict()
        for name in net_paras.keys():
            if name.find('fc.') >= 0 or name.find('myconv2.') >= 0:
                continue
            value = net_paras[name]
            param = resnet_dic[name]
            if isinstance(param, Parameter):
                param = param.data
            value.copy_(param)

    def load_checkpoint(self, fname):
        if os.path.isfile(fname):
            print('loading checkpoint {}'.format(fname))
            checkpt = torch.load(fname)
            self.load_state_dict(checkpt['state_dict'])
        else:
            print('{} not found'.format(fname))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


