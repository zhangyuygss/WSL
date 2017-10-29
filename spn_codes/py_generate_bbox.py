## Here is the code to generate the bounding box from the heatmap
# 
# to reproduce the ILSVRC localization result, you need to first generate
# the heatmap for each testing image by merging the heatmap from the
# 10-crops (it is exactly what the demo code is doing), then resize the merged heatmap back to the original size of
# that image. Then use this bbox generator to generate the bbox from the resized heatmap.
#
# The source code of the bbox generator is also released. Probably you need
# to install the correct version of OpenCV to compile it.
#
# Special thanks to Hui Li for helping on this code.
#
# Bolei Zhou, April 19, 2016

import os
import numpy as np
import cv2
from py_map2jpg import py_map2jpg

def im2double(im):
	return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

bbox_threshold = [20, 100, 110] # parameters for the bbox generator
curParaThreshold = str(bbox_threshold[0])+' '+str(bbox_threshold[1])+' '+str(bbox_threshold[2])+' '
curHeatMapFile = 'bboxgenerator/heatmap_6.jpg';
curImgFile = 'bboxgenerator/sample_6.jpg';
curBBoxFile = 'bboxgenerator/heatmap_6.txt';

os.system("bboxgenerator/./dt_box "+curHeatMapFile+' '+curParaThreshold+' '+curBBoxFile)

with open(curBBoxFile) as f:
	for line in f:
		items = [int(x) for x in line.strip().split()]

boxData1 = np.array(items[0::4]).T
boxData2 = np.array(items[1::4]).T
boxData3 = np.array(items[2::4]).T
boxData4 = np.array(items[3::4]).T

boxData_formulate = np.array([boxData1, boxData2, boxData1+boxData3, boxData2+boxData4]).T

col1 = np.min(np.array([boxData_formulate[:,0], boxData_formulate[:,2]]), axis=0)
col2 = np.min(np.array([boxData_formulate[:,1], boxData_formulate[:,3]]), axis=0)
col3 = np.max(np.array([boxData_formulate[:,0], boxData_formulate[:,2]]), axis=0)
col4 = np.max(np.array([boxData_formulate[:,1], boxData_formulate[:,3]]), axis=0)

boxData_formulate = np.array([col1, col2, col3, col4]).T

curHeatMap = cv2.imread(curHeatMapFile)
curImg = cv2.imread(curImgFile)

curHeatMap = im2double(curHeatMap)
curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
curHeatMap = im2double(curImg)*0.2+im2double(curHeatMap)*0.7

for i in range(boxData_formulate.shape[0]): # for each bbox
	print(boxData_formulate[i][:2])
	print(boxData_formulate[i][2:])
	cv2.rectangle(curHeatMap, tuple(boxData_formulate[i][:2]), tuple(boxData_formulate[i][2:]), (255,0,0), 3)
	cv2.imshow('bbox', curHeatMap)
	cv2.waitKey(0)