"""
Data loading script for pascal voc2007
=====================================
Load VOC dataset to pytorch torch.utils.data.Dataset class for further
training and processing.
*Author*: Yu Zhang, Northwestern Polytechnical University
"""

import os
import shutil
import pandas as pd
import torch
import numpy as np
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import random
import torchvision.transforms as transforms


def select_validation(val_size):
    """Select val_size images as validation set from VOC2007 val.txt file"""

    root_dir = '/home/zhangyu/data/VOC2007/'
    train_annos = os.path.join(root_dir, 'train_annos')
    val_annos = os.path.join(root_dir, 'val_annos')
    os.mkdir(val_annos)
    xmls_dir = os.path.join(root_dir, 'Annotations')
    txtfile = '/home/zhangyu/data/VOC2007/ImageSets/Main/val.txt'
    df = pd.read_csv(txtfile, dtype=str, names=['filename'])
    mask = random.sample(range(df.shape[0]), min(val_size, df.shape[0]))
    val_files = df['filename'].values[mask]
    shutil.copytree(xmls_dir, train_annos)
    for filename in val_files:
        xmlname = filename + '.xml'
        shutil.move(os.path.join(train_annos, xmlname),
                    os.path.join(val_annos, xmlname))


def training_set_feature(set_dir):
    """Computer image mean and std for training set"""


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        return None
    else:
        ext = allfiles[0].split('.')[-1]
        filelist = [fname.replace(''.join(['.', ext]), '') for fname in allfiles]
        return ext, filelist


def load_annotation(xmlFile):
    """
    Read annotations from for a image from xml file and return a dictionary of
    the objects and their locations
    """
    with open(xmlFile) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    anno = BeautifulSoup(xml, "html5lib")
    anno_dic = {}
    fname = anno.findChild('filename').contents[0]
    anno_dic['filename'] = fname
    objs = anno.findAll('object')
    # print('Number of objects:', len(objs))
    objects = []
    for obj in objs:
        obj_name = obj.findChild('name').contents[0]
        bbox = obj.findChildren('bndbox')[0]
        xmin = int(bbox.findChildren('xmin')[0].contents[0])
        ymin = int(bbox.findChildren('ymin')[0].contents[0])
        xmax = int(bbox.findChildren('xmax')[0].contents[0])
        ymax = int(bbox.findChildren('ymax')[0].contents[0])
        obj_dic = {'object_name': obj_name,
                   'location': np.array([xmin, ymin, xmax, ymax])}
        objects.append(obj_dic)
    anno_dic['annotation'] = objects
    return anno_dic


def get_cls_cord_from_anno(anno_dic, cls_name):
    """can have multiple cords for one class in one image"""
    objs = anno_dic['annotation']
    cords = None
    for obj in objs:
        if obj['object_name'] == cls_name:
            cords = obj['location'] if cords is None \
                    else np.vstack((cords, obj['location']))
    return cords


def load_class(xmlFile):
    """
    Read only class information from xml file, no bbox GT
    """
    all_cls = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike',
                          'person', 'pottedplant', 'sheep', 'sofa',
                          'train', 'tvmonitor'])
    with open(xmlFile) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    anno = BeautifulSoup(xml, "html5lib")
    objs = anno.findAll('object')
    # print('Number of objects:', len(objs))
    classes = np.zeros(all_cls.size)
    for obj in objs:
        obj_name = obj.findChild('name').contents[0]
        classes[np.where(all_cls == obj_name)[0][0]] = 1
    return classes


class VOCDataset(Dataset):
    """PASCAL VOC2007 dataset"""

    def __init__(self, xmlsPath, imgDir, transform=None):
        """
        Args
         xmlsPath: Path to xml files with image annotations, one xml file per image
         imgDir: Directory with all the images
         transform:
        """
        _, self.imgList = fold_files(xmlsPath)
        self.xmlsPath = xmlsPath
        self.imgDir = imgDir
        self.transform = transform

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        imgName = self.imgList[idx] + '.jpg'
        img = io.imread(os.path.join(self.imgDir, imgName))
        im_sz = np.array(list(img.shape[:2]))
        cls = load_class(os.path.join(
            self.xmlsPath, ''.join([self.imgList[idx], '.xml'])))
        sample = {'filename':imgName, 'sz': im_sz, 'image': img, 'class': cls}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size, sample without bbox GT
    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # sample = deepcopy(sample)
        image, cls, filename, sz = sample['image'], sample['class'], \
                                   sample['filename'], sample['sz']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w), mode='constant')

        return {'filename': filename, 'image': image, 'class': cls, 'sz': sz}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors, no bbox"""

    def __call__(self, sample):
        image, cls, filename, sz = sample['image'], sample['class'], \
                                   sample['filename'], sample['sz']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'filename': filename, 'sz': sz,
                'image': torch.from_numpy(image),
                'class': torch.from_numpy(cls)}


class Normalize(object):
    """Normalize images"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, cls, filename, sz = sample['image'], sample['class'], \
                                   sample['filename'], sample['sz']
        normal = transforms.Normalize(self.mean, self.std)
        image = normal(image)
        return {'filename': filename, 'image': image, 'class': cls, 'sz': sz}


class Augmentation(object):
    """image augmentation with flip and crop"""
    def __call__(self, sample):
        image, cls, filename, sz = sample['image'], sample['class'], \
                                   sample['filename'], sample['sz']
        # randomly choose whether do augmentation
        if random.random() < 0.7:
            h, w = image.shape[:2]

            # convert to PIL.Image for crop and flip
            topil = transforms.ToPILImage()
            image = topil(image)

            crop_size = random.randint(20, 30)
            pad_size = random.randint(0, 4)
            crop = transforms.RandomCrop(size=[h-crop_size, w-crop_size],
                                         padding=pad_size)
            image = crop(image)
            flip = transforms.RandomHorizontalFlip()
            image = flip(image)
            image = np.array(image.convert('RGB'))
        return {'filename': filename, 'image': image, 'class': cls, 'sz': sz}


class RescaleBox(object):
    """Rescale the image in a sample to a given size, sample with bbox GT

    Args:
        output_size (int or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # sample = deepcopy(sample)
        image, annos = sample['image'], sample['info']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w), mode='constant')
        objs = annos['annotation']
        for index, obj in enumerate(objs):
            obj_loc = obj['location']
            obj_loc[0] = max(int(obj_loc[0] * (new_w / w)), 1)
            obj_loc[2] = min(int(obj_loc[2] * (new_w / w)), new_w)
            obj_loc[1] = max(int(obj_loc[1] * (new_h / h)), 1)
            obj_loc[3] = min(int(obj_loc[3] * (new_h / h)), new_h)
            annos['annotation'][index]['location'] = obj_loc

        return {'image': image, 'info': annos}


class ToTensorBox(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annos = sample['image'], sample['info']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        objs = annos['annotation']
        for index, obj in enumerate(objs):
            annos['annotation'][index]['location'] = torch.from_numpy(obj['location'])

        return {'image': torch.from_numpy(image), 'info': annos}


