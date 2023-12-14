import pandas as pd
import numpy as np
import time
import os
import math
from tqdm import *
from itertools import product
from scipy.spatial import distance_matrix
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import skeletonize, dilation, binary_erosion, disk
from skimage.measure import label, regionprops

def is_close(target_item, item, thresh):
    return math.dist(target_item, item)<=thresh
    
def remove_close_neighbors(input_list, thresh, position):
    target_item = input_list[position]
    return [item for i, item in enumerate(input_list) if i == position or not is_close(target_item, item, thresh)]

def remove_close_neighbors(input_list, thresh, position):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    input_list : list
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    thresh : numeric
        Ratio for at which point to make the split, default is 33%
    position : int
        Ratio for at which point to make the split, default is 33%
    """
    target_item = input_list[position]
    return [item for i, item in enumerate(input_list) if i == position or not is_close(target_item, item, thresh)]

def clean_up_results(input_list, threshold):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    input_list : list
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    threshold : numeric
        Ratio for at which point to make the split, default is 33%
    """
    pos = 0
    while pos < len(input_list):
        input_list = remove_close_neighbors(input_list, threshold, pos)
        pos += 1
    return input_list

def superimpose_after_rolling(img_c2, img_c1, shift_):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    img_c1 : numpy array
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    imc_c2 : numpy array
        Ratio for at which point to make the split, default is 33%
    shift_ : int
        Ratio for at which point to make the split, default is 33%
    """
    sx, sy = shift_
    img_c1_s = np.roll(img_c1, shift=(sx, sy), axis=(0,1))
    return abs(img_c1_s - img_c2).sum()

def extract_ROI(fname_c1, fname_c2, reduction=200, 
         image_smoothing=1.5, image_threshold=.5, displacement=70, 
         node_smoothing = 3., edge_smoothing=1., area_threshold=30):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    fname_x : string
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    reduction : int
        Ratio for at which point to make the split, default is 33%
    image_smoothing : float
        Ratio for at which point to make the split, default is 33%
    image_threshold : float
        Ratio for at which point to make the split, default is 33%
    displacement : float
        Ratio for at which point to make the split, default is 33%
    node_smoothing : float
        Ratio for at which point to make the split, default is 33%
    edge_smoothing : float
        Ratio for at which point to make the split, default is 33%
    area_threshold : int
        Ratio for at which point to make the split, default is 33%
    """
    condition_1 = fname_c1[:-4]
    condition_2 = fname_c2[:-4]

    c1 = pd.read_csv(fname_c1).dropna()
    c1 = c1[c1.precisionz.notnull()][['x', 'y', 'precisionx', 'photon-count']]

    c2 = pd.read_csv(fname_c2).dropna()
    c2 = c2[c2.precisionz.notnull()][['x', 'y', 'precisionx', 'photon-count']]

    xmax, ymax = max(c2.x.max(), c1.x.max()), max(c2.y.max(), c1.y.max())
    xmax, ymax = xmax.astype(np.int64), ymax.astype(np.int64)

    img_c2 = np.zeros((xmax//reduction+1, ymax//reduction+1))
    img_c1 = np.zeros((xmax//reduction+1, ymax//reduction+1))

    for x, y in np.array(c1[['x', 'y']]).astype(np.int64):
        img_c1[x//reduction, y//reduction] +=1

    for x, y in np.array(c2[['x', 'y']]).astype(np.int64):
        img_c2[x//reduction, y//reduction] +=1

    img_c1 = gaussian(img_c1, image_smoothing)
    img_c1 = (img_c1 > image_threshold * threshold_otsu(img_c1)).astype(np.int64)

    img_c2 = binary_erosion(img_c2)
    img_c2 = gaussian(img_c2, image_smoothing)
    img_c2 = (img_c2 > image_threshold * threshold_otsu(img_c2)).astype(np.int64)

    s = list(product(range(-displacement, displacement, 1), range(-displacement, displacement, 1)))
    s_min = [superimpose_after_rolling(img_c1, img_c2, s_) for s_ in tqdm(s, leave=False, desc='Superimposing images')]
    shift = np.array(s[np.argmin(s_min)])
    shift *= reduction

    c1_shifted = c1.copy()                         
    c1_shifted.x = c1_shifted.x - shift[0]
    c1_shifted.y = c1_shifted.y - shift[1]             

    dist_thresh = 2.
    connectivity_thresh = 3.
    end_thresh = 1

    skel_c2 = skeletonize(img_c2)

    xo, yo = np.where(skel_c2 == 1)
    p = np.stack([xo, yo], axis=1)
    d_mat = distance_matrix(p, p)
    connectivity = []

    for i, row in enumerate(d_mat):
        row = np.delete(row, i)
        val = np.sum(row < dist_thresh)
        connectivity.append(val)

    pc = pd.DataFrame(p, columns={'x', 'y'})
    pc.loc[:, 'c'] = connectivity
    pc_ = pc[pc.c >= connectivity_thresh]

    input_list = np.array(pc_[['x', 'y']])
    processed_list = clean_up_results(input_list, 10)
    pc_rm = pd.DataFrame(processed_list, columns={'x', 'y'})

    node_out = np.zeros(skel_c2.shape)

    for _, row in pc_rm.iterrows():
        xi, yi = row[['x', 'y']]
        node_out[xi, yi] +=1

    node_out_smooth = gaussian(node_out, node_smoothing)
    node_out_pl = gaussian(node_out, 3)
    node_out_smooth = (node_out_smooth > threshold_otsu(node_out_smooth)).astype(np.int32)
    node_out_pl = (node_out_pl > threshold_otsu(node_out_pl)).astype(np.int32)

    skeleton = gaussian(skel_c2, edge_smoothing) 
    skeleton = (skeleton > threshold_otsu(skeleton)).astype(np.int32)
    edges = (skeleton - node_out_smooth > 0).astype(np.int32)
    labels = dilation(label(edges), footprint = disk(1))

    c1_shifted['condition'] = condition_1
    c2['condition'] = condition_2
    combined_conditions = c1_shifted.append(c2)
    combined_conditions['label'] = 0
    combined_conditions['xsq'] = combined_conditions.x // reduction
    combined_conditions['ysq'] = combined_conditions.y // reduction

    edx, edy = np.where(edges==1)

    count = 1
    for region in tqdm(regionprops(labels), leave=False, desc='Selection ROIs'):
        if region.area > area_threshold:
            xx, yy = np.where(labels == region.label)
            lbx = xx*reduction
            lby = yy*reduction
            xy = np.stack([xx, yy], axis=1)
            for x_, y_ in xy:
                combined_conditions['label'] = np.where(
                    (combined_conditions.xsq == x_) & (combined_conditions.ysq == y_), 
                    count, 
                    combined_conditions['label']
                )
            count += 1
    return combined_conditions