import os
import numpy as np


def gen_bbox(attenMapFile, thresh, bboxFile):
    """
    A python wrap to generate bbox from a attention(heat) map
    :param attenMapFile: map file
    :param thresh: threshold para of dt_box function
    :param bboxFile: file for dt_box to write the result
    :return:
    """
    tmp_file = bboxFile.split('.')[0] + '_tmp.txt'
    thresh = thresh.astype(str)
    argStr = 'bboxgenerator/./dt_box ' + attenMapFile
    argStr = argStr + ' ' + thresh[0] + ' ' + thresh[1] + ' ' + thresh[2] + ' '
    argStr = argStr + tmp_file
    os.system(argStr)

    with open(tmp_file) as f:
        for line in f:
            items = [int(x) for x in line.strip().split()]

    boxData1 = np.array(items[0::4]).T
    boxData2 = np.array(items[1::4]).T
    boxData3 = np.array(items[2::4]).T
    boxData4 = np.array(items[3::4]).T

    boxData_formulate = np.array([boxData1, boxData2, boxData1 + boxData3, boxData2 + boxData4]).T

    col1 = np.min(np.array([boxData_formulate[:, 0], boxData_formulate[:, 2]]), axis=0)
    col2 = np.min(np.array([boxData_formulate[:, 1], boxData_formulate[:, 3]]), axis=0)
    col3 = np.max(np.array([boxData_formulate[:, 0], boxData_formulate[:, 2]]), axis=0)
    col4 = np.max(np.array([boxData_formulate[:, 1], boxData_formulate[:, 3]]), axis=0)

    boxes = np.array([col1, col2, col3, col4]).T
    # take the tightest box as attention box
    keep_idx = 0
    area = float('Inf')
    for idx in range(boxes.shape[0]):
        cur_eara = (boxes[idx, 2] - boxes[idx, 0])*(boxes[idx, 3] - boxes[idx, 1])
        if cur_eara < area:
            area = cur_eara
            keep_idx = idx
    return boxes[keep_idx, :]
