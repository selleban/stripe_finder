# packages
import pandas as pd
import numpy as np
import time
import os
import math
from tqdm import *
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from scipy.ndimage import laplace

def add_rotated_shifted_gaussian(theta, x0, y0, sig_x, sig_y, x, y):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    theta : numeric
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    x, y : numeric
        Ratio for at which point to make the split, default is 33%
    x0, y0 : numeric
        Ratio for at which point to make the split, default is 33%
    sig_x, sig_y : numeric
        Ratio for at which point to make the split, default is 33%
    """
    A = np.mat([[1.0 / (np.sqrt(2.0) * sig_x), 0.0], [0.0, 1.0 / (np.sqrt(2.0) * sig_y)]])
    R = np.mat([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    
    X0 = np.mat([[x0], [y0]])
    X0_r = R * X0

    orig_shape = x.shape
    Ap = A * R.T
    
    X = np.mat(np.stack(((x - X0_r[0, 0]).flatten(), (y - X0_r[1, 0]).flatten())))
    prefactor = 1.0
    
    Y = np.asarray(Ap * X)
    result = prefactor * np.exp(-np.sum(Y * Y, axis=0))
    return result.reshape(orig_shape)

def make_filter(x, y, theta, n_gaussians, sig_x, sig_y, r0):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    x, y : numeric
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    theta : numeric
        Ratio for at which point to make the split, default is 33%
    n_gaussians : numeric
        Ratio for at which point to make the split, default is 33%
    sig_x, sig_y : numeric
        Ratio for at which point to make the split, default is 33%
    r0 : numeric
        Ratio for at which point to make the split, default is 33%    
    """
    filter_img = np.zeros_like(x)
    
    alt = -1
    for n in range(n_gaussians):
        
        m = (n - n_gaussians // 2 + 0.5)
                
        filter_img += alt * add_rotated_shifted_gaussian(theta, r0 * m, 0.0, sig_x, sig_y, x, y)
        
        alt *= -1

    return filter_img

def conv_at_theta(fimg, theta, n_gaussian, dx, dy, sig_x, sig_y, r0):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    fimg : numeric
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    theta : numeric
        Ratio for at which point to make the split, default is 33%
    n_gaussians : numeric
        Ratio for at which point to make the split, default is 33%
    dx, dy : numeric
        Ratio for at which point to make the split, default is 33%
    sig_x, sig_y : numeric
        Ratio for at which point to make the split, default is 33%
    r0 : numeric
        r0
    """
    filter_img = make_filter(dx, dy, theta, n_gaussian, sig_x, sig_y, r0)
    return np.real(np.fft.ifft2(fimg * np.fft.fft2(np.fft.fftshift(filter_img))))   

def gaussian_kernel_convolution(df, point_thresh=1000, 
         img_size=1000, clip_crit=1., sig_y_=1000, 
         actin_width=190, n_gaussian=7):
    """ 
    Create a timeseries appropriate to feed through the machine learning models.
    The function acts as a rolling window where every time point prediction x(t+1) 
    follows the time series x(t - n). 
    ----------
    df : numeric
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    point_thresh : numeric
        Ratio for at which point to make the split, default is 33%
    img_size : numeric
        Ratio for at which point to make the split, default is 33%
    clip_crit : numeric
        Default value for the different time dimension, in this case it's 4 for PEF, FEV1, FVC and x
    sig_y_ : numeric
        Ratio for at which point to make the split, default is 33%
    actin_width : numeric
        Ratio for at which point to make the split, default is 33%
    n_gaussian : numeric
        Ratio for at which point to make the split, default is 33%
    """
    selection = df.groupby('label').count() > point_thresh
    df_ = df[df.label.isin(selection[selection.x == True].index)]
    iter_list = np.delete(np.unique(df_.label), 0)
    df_convolved = pd.DataFrame()

    for ii in tqdm(iter_list, leave=False, desc='Gaussian kernel convolution'):
        df_crop = df[df.label == ii]
        a, b = np.polyfit(df_crop.x, df_crop.y, 1)
        err = np.sum((np.polyval(np.polyfit(df_crop.x, df_crop.y, 1), df_crop.x) - df_crop.y)**2)
        theta = np.arctan(a) + (.5 * np.pi)
        pixels_ = np.array(df_crop[['x', 'y']])
        pixels = np.array(df_crop[['x', 'y']])

        pixels[:, 0] = pixels_[:, 0] - pixels_[:, 0].min()
        pixels[:, 1] = pixels_[:, 1] - pixels_[:, 1].min()

        xmin, xmax = np.amin(pixels[:, 0]), np.amax(pixels[:, 0])
        ymin, ymax = np.amin(pixels[:, 1]), np.amax(pixels[:, 1])
        lmax = np.amax(pixels)

        img = np.zeros((img_size, img_size))
        scale = (img_size - 1.0) / lmax
        x = (np.round(pixels[:, 0] * scale)).astype(np.int32) 
        y = (np.round(pixels[:, 1] * scale)).astype(np.int32)

        for _x, _y in zip(x, y):
            img[_x, _y] += 1.0 #_phc

        clip_ceil = clip_crit * np.std(img)
        clipped_img = np.clip(img, 0.0, clip_ceil)
        clipped_img = clipped_img / np.amax(clipped_img)
        fimg = np.fft.fft2(clipped_img)

        xi = np.arange(0, img.shape[0])
        xx, yy = np.meshgrid(xi, xi)
        dx = xx - 0.5 * img.shape[0]
        dy = yy - 0.5 * img.shape[1]

        r0 = 0.5 * actin_width * scale
        sig_x = 20.0 * scale
        sig_y = sig_y_ * scale
#         conv_img = laplace(conv_at_theta(fimg, theta, n_gaussian, dx, dy, sig_x, sig_y, r0))
        conv_img = conv_at_theta(fimg, theta, n_gaussian, dx, dy, sig_x, sig_y, r0)
        conv_array = []
        
        for i_, (x_, y_) in enumerate(zip(x, y)):
            val = conv_img[x_, y_]
            c_x, c_y = pixels_[i_]
            conv_array.append([ii, c_x, c_y, val])
            
        df_conv = pd.DataFrame(conv_array)
        df_convolved = df_convolved.append(df_conv)
    df_convolved = df_convolved.rename(columns={
        0:'label',
        1:'x',
        2:'y',
        3:'value'
    })
    df = df.merge(df_convolved, on=['x', 'y', 'label'], how='left')
    return df