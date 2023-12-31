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
    """ 
    function to evaluate if two points are closely located depending on a distance threshold. 
    ----------
    target_item : tuple
        first set of x,y-coordinates for comparison
    item : tuple
        second set of x,y-coordinates for comparison
    thresh : float
        distance threshold, if true the points are considered closely located
    """
    return math.dist(target_item, item)<=thresh
    
def remove_close_neighbors(input_list, thresh, position):
    """ 
    function to detech closely located points. 
    ----------
    input_list : list
        list of all x,y-coordinates
    thresh : float
        distance threshold
    position : int
        one set of x,y-coordinates for which the distance is determined
    """
    target_item = input_list[position]
    return [item for i, item in enumerate(input_list) if i == position or not is_close(target_item, item, thresh)]

def clean_up_results(input_list, threshold):
    """ 
    Function to clean up very closely located points. 
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
    Function that calculates the difference between two arrays in positions after applying a shift in x,y-direction on one of the arrays. 
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
    Account for the shift in one of the images and extracting Regions of Interest within the image. 
    ----------
    fname : string
        file path to image files
    reduction : int
        reduction of the dimensions of the images
    image_smoothing : float
        sigma for gaussian smoothing of the image
    image_threshold : float
        image threshold for binary thresholding of the image
    displacement : float
        max displacement in pixels that is considered for determining the optimal shift
    node_smoothing : float
        sigma for gaussian smoothing of the detected nodes
    edge_smoothing : float
        sigma for gaussian smoothing of the detected edges
    area_threshold : int
        Threshold for filtering out very small region of interest sections
    """
    # Reading the files
    condition_1 = fname_c1[:-4]
    condition_2 = fname_c2[:-4]

    c1 = pd.read_csv(fname_c1).dropna()
    c1 = c1[c1.precisionz.notnull()][['x', 'y', 'precisionx', 'photon-count']]
    c2 = pd.read_csv(fname_c2).dropna()
    c2 = c2[c2.precisionz.notnull()][['x', 'y', 'precisionx', 'photon-count']]
    
    # determining max values
    xmax, ymax = max(c2.x.max(), c1.x.max()), max(c2.y.max(), c1.y.max())
    xmax, ymax = xmax.astype(np.int64), ymax.astype(np.int64)

    # Transform coordinates into arrays (images         
    img_c2 = np.zeros((xmax//reduction+1, ymax//reduction+1))
    img_c1 = np.zeros((xmax//reduction+1, ymax//reduction+1))

    for x, y in np.array(c1[['x', 'y']]).astype(np.int64):
        img_c1[x//reduction, y//reduction] +=1

    for x, y in np.array(c2[['x', 'y']]).astype(np.int64):
        img_c2[x//reduction, y//reduction] +=1

    # Gaussian smoothing and thresholding of both images
    img_c1 = gaussian(img_c1, image_smoothing)
    img_c1 = (img_c1 > image_threshold * threshold_otsu(img_c1)).astype(np.int64)

    img_c2 = binary_erosion(img_c2)
    img_c2 = gaussian(img_c2, image_smoothing)
    img_c2 = (img_c2 > image_threshold * threshold_otsu(img_c2)).astype(np.int64)

    # Determining the shift in x,y-direction between the two images
    s = list(product(range(-displacement, displacement, 1), range(-displacement, displacement, 1)))
    s_min = [superimpose_after_rolling(img_c1, img_c2, s_) for s_ in tqdm(s, leave=False, desc='Superimposing images')]
    shift = np.array(s[np.argmin(s_min)])
    shift *= reduction

    # Correct for shift in image
    c1_shifted = c1.copy()                         
    c1_shifted.x = c1_shifted.x - shift[0]
    c1_shifted.y = c1_shifted.y - shift[1]             

    # connectivity parameters
    dist_thresh = 2.
    connectivity_thresh = 3.
    end_thresh = 1

    # Create skeleton image of image
    skel_c2 = skeletonize(img_c2)

    # Extract nodes and edges from skeleton
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

    # Clean op closely located nodes
    input_list = np.array(pc_[['x', 'y']])
    processed_list = clean_up_results(input_list, 10)
    pc_rm = pd.DataFrame(processed_list, columns={'x', 'y'})

    node_out = np.zeros(skel_c2.shape)

    for _, row in pc_rm.iterrows():
        xi, yi = row[['x', 'y']]
        node_out[xi, yi] +=1

    # Filter out edges only and label seperately
    node_out_smooth = gaussian(node_out, node_smoothing)
    node_out_pl = gaussian(node_out, 3)
    node_out_smooth = (node_out_smooth > threshold_otsu(node_out_smooth)).astype(np.int32)
    node_out_pl = (node_out_pl > threshold_otsu(node_out_pl)).astype(np.int32)

    skeleton = gaussian(skel_c2, edge_smoothing) 
    skeleton = (skeleton > threshold_otsu(skeleton)).astype(np.int32)
    edges = (skeleton - node_out_smooth > 0).astype(np.int32)
    edx, edy = np.where(edges==1)        
    labels = dilation(label(edges), footprint = disk(1))

    # Initialize dataframe
    c1_shifted['condition'] = condition_1
    c2['condition'] = condition_2
    combined_conditions = c1_shifted.append(c2)
    combined_conditions['label'] = 0
    combined_conditions['xsq'] = combined_conditions.x // reduction
    combined_conditions['ysq'] = combined_conditions.y // reduction

    # Label the coordinates in the original dataframe based on the labels array
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
