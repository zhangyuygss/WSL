"""
Get attention maps, bounding box for a batch
result including: attention maps for all images of 
all classes, top5 predictions 
All numerical inputs should be in numpy form
"""
import numpy as np 
import cv2
import skimage 
import skimage.io as imio
import skimage.transform as imtrans
import os
from .get_attention_map import get_attention_map
from .gen_bbox import gen_bbox
import warnings


def get_batch_rst(batch_names, img_szs, cls_scores, conv_ft, proposals, lr_weight,
                    att_map_dir, top5_save_file):
    img_list_file = top5_save_file.split('.txt')[0] + '_img_list.txt'
    batch_size, cls_number = cls_scores.shape
    atten_maps = get_attention_map(conv_ft, lr_weight, proposals)
    for bs in range(batch_size):
        img_name = batch_names[bs]
        with open(img_list_file, 'a+') as i_list:
            i_list.write(img_name + '\n')
        print('img {}/{} {}'.format(bs, batch_size-1, img_name))
        img_sz = img_szs[bs, :]
        map_save_dir = os.path.join(att_map_dir, img_name.split('.')[0])
        img_box_file = os.path.join(map_save_dir,
                                    '{}.txt'.format(img_name.split('.')[0]))
        # if ~os.path.isdir(map_save_dir):
        os.makedirs(map_save_dir, exist_ok=True)

        cls_score = cls_scores[bs, :]
        indexes = (-cls_score).argsort()[0:5]
        img_boxes = None
        maps = atten_maps[bs, :, :, :]
        for cls_idx in range(cls_number):
            cls_map = maps[:, :, cls_idx]
            cls_map = cv2.normalize(cls_map, None, 0.0, 0.99, cv2.NORM_MINMAX)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cls_map = skimage.img_as_float(cls_map)
                cls_map = imtrans.resize(cls_map, img_sz, mode='reflect')
                cls_map_file = map_save_dir + '/' + img_name.split('.')[0] + \
                            '_{}.jpg'.format(cls_idx)
                imio.imsave(cls_map_file, cls_map)
            box = gen_bbox(cls_map)
            img_boxes = box if cls_idx==0 else np.vstack((img_boxes, box))
            tmp_str = '{} {} {} {} {}'.format(cls_idx, box[0], box[1], box[2], box[3])
            with open(img_box_file, 'a+') as im_file:
                im_file.write(tmp_str + '\n')
        # record predict to top5 file
        top5_str = ''
        for idx in range(5):
            top5_str += '{} {} {} {} {} '.format(indexes[idx],
                        img_boxes[indexes[idx]][0], img_boxes[indexes[idx]][1],
                        img_boxes[indexes[idx]][2], img_boxes[indexes[idx]][3])
        top5_str += '\n'
        with open(top5_save_file, 'a+') as top5_f:
            top5_f.write(top5_str)
