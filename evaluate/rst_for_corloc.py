"""
Get attention maps, bounding box for a batch
result including: attention maps for all images of 
gt classes
All numerical inputs should be in numpy form
"""
import numpy as np
import cv2
import skimage
import skimage.io as imio
import skimage.transform as imtrans
import os
from pandas import DataFrame, read_csv
from .get_attention_map import get_attention_map
from .gen_bbox import gen_bbox
import warnings


def rst_for_corloc(batch_names, targets, img_szs, cls_scores, conv_ft,
                  lr_weight, att_map_dir, img_list_file, proposals=None):
    batch_size, cls_number = cls_scores.shape
    atten_maps = get_attention_map(conv_ft, lr_weight, proposals)
    df_columns = ['name', 'aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike',
                          'person', 'pottedplant', 'sheep', 'sofa',
                          'train', 'tvmonitor', 'cord']
    if not os.path.isfile(img_list_file):
        df = DataFrame(columns=df_columns)
    else:
        df = read_csv(img_list_file)
        pass
    for bs in range(batch_size):
        img_name = batch_names[bs]
        # print('img {}/{} {}'.format(bs, batch_size-1, img_name))
        img_sz = img_szs[bs, :]
        map_save_dir = os.path.join(att_map_dir, img_name.split('.')[0])
        os.makedirs(map_save_dir, exist_ok=True)
        img_data = (img_name, )
        maps = atten_maps[bs, :, :, :]
        indexes = np.where(targets[bs] == 1)[0]
        cls = np.zeros(cls_number)
        cords = None
        for cls_idx in indexes:
            cls[cls_idx] = 1
            cls_map = maps[:, :, cls_idx]
            cls_map_norm = cv2.normalize(cls_map, None, 0.0, 0.99, cv2.NORM_MINMAX)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls_map = skimage.img_as_float(cls_map)
                cls_map = imtrans.resize(cls_map, img_sz, mode='reflect')
                cls_map_norm = imtrans.resize(cls_map_norm, img_sz, mode='reflect')
                cls_map_file = map_save_dir + '/' + img_name.split('.')[0] + \
                               '_{}.png'.format(cls_idx)
                imio.imsave(cls_map_file, cls_map_norm)
            box = gen_bbox(cls_map)
            cord = np.array([cls_idx, box[0], box[1], box[2], box[3]])
            cords = cord if cords is None else np.hstack((cords, cord))
        img_data += tuple(cls)
        img_data += (cords, )
        dft = DataFrame(data=[img_data], columns=df_columns)
        df = df.append(dft)
    df.to_csv(img_list_file, index=False)
