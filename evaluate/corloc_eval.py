# Evaluate correct localization score
import numpy as np
import os
import sys
from pandas import read_csv
sys.path.append('../data_utils')
sys.path.append('./data_utils')
from load_voc import load_annotation, get_cls_cord_from_anno


def get_cord(cords, cls_idx):
    """get cord corresponding to the class"""
    cords = np.reshape(cords, (-1, 5))
    for idx in range(cords.shape[0]):
        if cords[idx, 0] == cls_idx:
            return cords[idx, 1:]
    return None  # if not found


def compute_iou(pred_cord, gt_cord):
    """
    Compute iou between predict cords and gt cords as intersection/union
    Note that gt_cord can have multiple pair of cords, consider correct
    localization if any cord is correctly prefdicted
    """
    iou_vector = []
    gt_cord = np.reshape(gt_cord, (-1, 4))
    for cord_idx in range(gt_cord.shape[0]):
        cord = gt_cord[cord_idx, :]
        ci = [max(cord[0], pred_cord[0]), max(cord[1], pred_cord[1]),
              min(cord[2], pred_cord[2]), min(cord[3], pred_cord[3])]
        iw = ci[2] - ci[0]
        ih = ci[3] - ci[1]
        iou = -1
        if iw > 0 and ih > 0:
            ua = (cord[2] - cord[0]) * (cord[3] - cord[1]) +\
                (pred_cord[2] - pred_cord[0]) * (pred_cord[3] - pred_cord[1]) -\
                iw * ih
            iou = iw * ih / ua
        iou_vector.append(iou)
    return max(iou_vector)


def corloc(predict_file, voc_path):
    all_cls = np.asarray(['aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike',
                          'person', 'pottedplant', 'sheep', 'sofa',
                          'train', 'tvmonitor'])
    df = read_csv(predict_file)
    cls_number = len(all_cls)
    cls_cor = np.zeros(cls_number)
    for cls_idx in range(cls_number):
        cls = all_cls[cls_idx]
        print('Evaluating class {}/{}: {}'.format(cls_idx+1, cls_number, cls))
        cls_imgs = df[df[cls] == 1]
        cls_num = cls_imgs.shape[0]
        correct_cnt = 0
        for idx, row in cls_imgs.iterrows():
            img_name = row['name']
            # TODO: improve datafram to supprot array save
            tmp_str = row['cord'].replace('[', '').replace(']', '')
            cords = np.fromstring(tmp_str, int, sep=' ')
            cord = get_cord(cords, cls_idx)
            gt_anno = load_annotation(os.path.join(voc_path,
                      '{}.xml'.format(img_name.split('.')[0])))
            gt_cord = get_cls_cord_from_anno(gt_anno, cls)
            iou = compute_iou(cord, gt_cord)
            if iou > 0.5:
                correct_cnt += 1
        cur_cls_score = correct_cnt / cls_num
        print('Current class score: {}'.format(cur_cls_score))
        cls_cor[cls_idx] = cur_cls_score
    return cls_cor

if __name__ == '__main__':
    clsloc = corloc(sys.argv[1], sys.argv[2])
    print(clsloc)
