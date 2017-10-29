# generate bounding box using threshold
import numpy as np
from skimage.measure import regionprops 


def gen_bbox(map):
    regions = get_region(map)
    if regions is not None:
        box = list(regions[0].bbox)
        # skimage:(top,left,bottom,right)python type
        # --> voc:(left,top,right,bottom)
        voc_box = box.copy()
        voc_box[0] = box[1]
        voc_box[1] = box[0]
        voc_box[2] = box[3] - 1
        voc_box[3] = box[2] - 1
        return voc_box
    return np.array([0, 0, map.shape[0], map.shape[1]])


def get_region(att_map):
    mean_map = att_map.mean()
    paras = [1, 0.8, 0.6]
    for para in paras:
        thresh = mean_map * para
        b_map = np.ndarray.astype(att_map >= thresh, int)
        regions = regionprops(b_map)
        if len(regions) > 0:
            return regions
    print('Bad attention map')
    return None
